import argparse
import numpy
import pandas
import lightgbm
import math
import matplotlib
import os

from matplotlib import pyplot
from secrets import randbits
from sklearn import metrics, model_selection
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from ppml_utils import *

average_types = ["micro", "macro", "weighted"]

binary_score_types = ["accuracy", "average_precision", "f1", "recall", "auc"]
multi_score_types = ["accuracy","average_precision", "f1", "recall", "auc_ovo", "auc_ovr"]

def single_run_binary(train_x, train_y, test_x, test_y, classifier):
    classifier.fit(train_x, train_y)

    predictions = classifier.predict(test_x)
    prob_preds = classifier.predict_proba(test_x)

    scores = pandas.DataFrame(index=binary_score_types, 
                              columns=average_types, dtype=float)
    
    for avg in average_types:
        scores.loc["accuracy", avg] = metrics.accuracy_score(test_y, predictions)
        scores.loc["average_precision", avg] = metrics.average_precision_score(test_y, prob_preds, average=avg)
        scores.loc["f1", avg] = metrics.f1_score(test_y, predictions, average=avg)
        scores.loc["recall", avg] = metrics.recall_score(test_y, predictions, average=avg)
        try:
            scores.loc["auc", avg] = metrics.roc_auc_score(test_y, predictions, average=avg)
        except:
            scores.loc["auc", avg] = -1.0
    
    confusion_matrix = metrics.confusion_matrix(test_y, predictions)
    return (scores, confusion_matrix)

def run_binary(train, test, classifier,args, iterations):
    scores = pandas.DataFrame(index=binary_score_types, 
                              columns=average_types, 
                              dtype=object
                              )
    confusion_matrices = []
    label = args.label_column

    for col in scores.columns:
        for row in scores.index:
            scores.loc[row, col] = numpy.atleast_1d(numpy.empty(iterations, dtype=float))

    if iterations > 1:
        for i in range(iterations):
            shuffle_train = train.sample(frac=1)
            shuffle_test = test.sample(frac=1)

            train_x = shuffle_train.drop(columns=[label])
            train_y = shuffle_train[label]
            test_x = shuffle_test.drop(columns=[label])
            test_y = shuffle_test[label]

            i_scores, i_cm = single_run_binary(train_x, train_y, test_x, test_y, classifier)
            for col in scores.columns:
                for row in scores.index:
                    scores.loc[row, col][i] = i_scores[col][row]
            confusion_matrices.append(i_cm)
    else:
        shuffle_train = train.sample(frac=1)
        shuffle_test = test.sample(frac=1)

        train_x = shuffle_train.drop(columns=[label])
        train_y = shuffle_train[label]
        test_x = shuffle_test.drop(columns=[label])
        test_y = shuffle_test[label]

        scores, i_cm = single_run_binary(train_x, train_y, test_x, test_y, classifier)
        confusion_matrices.append(i_cm)

    return (scores, confusion_matrices)

def run_binary_cv(train_x, train_y, classifier, iterations, cv_folds=5):
    scores = pandas.DataFrame(index=binary_score_types, 
                              columns=average_types, 
                              dtype=object
                              )

    confusion_matrices = []
    for col in scores.columns:
        for row in scores.index:
            scores.loc[row, col] = numpy.empty(iterations * cv_folds, dtype=float)

    for i in range(iterations):
        i_base = i * cv_folds
        folds = model_selection.StratifiedKFold(cv_folds, shuffle=True).split(train_x, train_y)
        for i_fold, (train_fold, test_fold) in enumerate(folds):
            i_scores, i_cm = single_run_binary(train_x.iloc[train_fold],
                                               train_y.iloc[train_fold], 
                                               train_x.iloc[test_fold], 
                                               train_y.iloc[test_fold], 
                                               classifier)
            for col in scores.columns:
                for row in scores.index:
                    scores.loc[row, col][i_base + i_fold] = i_scores[col][row]
            confusion_matrices.append(i_cm)

    return (scores, confusion_matrices)

def single_run_multi(train_x, train_y, test_x, test_y, classifier):
    classifier.fit(train_x, train_y)

    predictions = classifier.predict(test_x)
    prob_preds = classifier.predict_proba(test_x)

    binarizer = LabelBinarizer()
    binarizer.fit(predictions)
    encoded_test_y = binarizer.fit_transform(test_y)
    encoded_preds = binarizer.transform(predictions)

    if (prob_preds.shape[1] < encoded_test_y.shape[1]):
        pad = numpy.zeros((prob_preds.shape[0], encoded_test_y.shape[1] - prob_preds.shape[1]))
        prob_preds = numpy.hstack((prob_preds, pad))

    print(encoded_test_y)
    print(encoded_test_y.size, prob_preds.size)
    print(encoded_test_y.shape, prob_preds.shape)

    scores = pandas.DataFrame(index=multi_score_types, 
                              columns=average_types, dtype=float)
    for avg in average_types:
        scores.loc["accuracy", avg] = metrics.accuracy_score(test_y, predictions)
        scores.loc["average_precision", avg] = metrics.average_precision_score(encoded_test_y, encoded_preds, average=avg)
        try:
            scores.loc["f1", avg] = metrics.f1_score(
                    test_y, predictions, average=avg)
        except:
            scores.loc["f1", avg] = -1.0

        try:
            scores.loc["recall", avg] = metrics.recall_score(
                    test_y, predictions, average=avg)
        except:
            scores.loc["recall", avg] = -1.0

        if avg != "micro":
            try:
                scores.loc["auc_ovo", avg] = metrics.roc_auc_score(
                        test_y, prob_preds, average=avg, multi_class="ovo")
            except:
                scores.loc["auc_ovo", avg] = -1.0
        else:
            scores.loc["auc_ovo", avg] = -1.0
        try:
            scores.loc["auc_ovr", avg] = metrics.roc_auc_score(
                    test_y, prob_preds, average=avg, multi_class="ovr")
        except:
            scores.loc["auc_ovr", avg] = -1.0

    confusion_matrix = metrics.confusion_matrix(test_y, predictions)
    return (scores, confusion_matrix)

def run_multi(train, test, classifier, args, iterations):
    scores = pandas.DataFrame(index=multi_score_types, 
                              columns=average_types,
                              dtype=object)
    confusion_matrices = []
    label = args.label_column

    for col in scores.columns:
        for row in scores.index:
            scores.loc[row, col] = numpy.empty(iterations, dtype=float)

    if isinstance(train, pandas.DataFrame):
        train_x = train.drop(columns=[label]).values
        train_y = train[label].values
    else:
        train_x = train[:, 1:]
        train_y = train[:, 0]
    
    if isinstance(test, pandas.DataFrame):
        test_x = test.drop(columns=[label]).values
        test_y = test[label].values
    else:
        test_x = test[:, 1:]
        test_y = test[:, 0]

    if iterations > 1:
        for i in range(iterations):
            i_scores, i_cm = single_run_multi(train_x, train_y, test_x, test_y, classifier)
            for col in scores.columns:
                for row in scores.index:
                    scores.loc[row, col][i] = i_scores[col][row]
            confusion_matrices.append(i_cm)
    else:
        i_scores, i_cm = single_run_multi(train_x, train_y, test_x, test_y, classifier)
        for col in scores.columns:
            for row in scores.index:
                scores.loc[row, col] = numpy.array(i_scores[col][row], ndmin=1)
        confusion_matrices.append(i_cm)

    return (scores, confusion_matrices)

def run_multi_cv(train_x, train_y, classifier, iterations, cv_folds=5):
    scores = pandas.DataFrame(index=multi_score_types, 
                              columns=average_types, 
                              dtype=object
                              )

    confusion_matrices = []
    for col in scores.columns:
        for row in scores.index:
            scores.loc[row, col] = numpy.empty(iterations * cv_folds, dtype=float)

    for i in range(iterations):
        i_base = i * cv_folds
        folds = model_selection.StratifiedKFold(cv_folds, shuffle=True).split(train_x, train_y)
        for i_fold, (train_fold, test_fold) in enumerate(folds):
            i_scores, i_cm = single_run_multi(train_x.iloc[train_fold],
                                               train_y.iloc[train_fold], 
                                               train_x.iloc[test_fold], 
                                               train_y.iloc[test_fold], 
                                               classifier)
            for col in scores.columns:
                for row in scores.index:
                    scores.loc[row, col][i_base + i_fold] = i_scores[col][row]
            confusion_matrices.append(i_cm)

    return (scores, confusion_matrices)

def average_confusion_matrix(matrices):
    average_cm = numpy.empty(matrices[0].shape, dtype=float)

    iter = numpy.nditer(average_cm, flags=['multi_index'])
    for i in iter:
        average_cm[iter.multi_index] = float(sum([j[iter.multi_index] for j in matrices])) / len(matrices)
    return average_cm

def std_confusion_matrix(matrices):
    std_dev_cm = numpy.empty(matrices[0].shape, dtype=float)

    iter = numpy.nditer(std_dev_cm, flags=['multi_index'])
    for i in iter:
        std_dev_cm[iter.multi_index] = numpy.std(
                numpy.array([j[iter.multi_index] for j in matrices]).astype(float))
    return std_dev_cm

def average_scores(scores):
    averages = pandas.DataFrame(index=scores.index, columns=scores.columns, dtype=float)

    for avg in scores.columns:
        for row in scores.index:
            averages.loc[row, avg] = sum(scores[avg][row]) / len(scores[avg][row])
    return averages

def std_scores(scores):
    std_devs = pandas.DataFrame(index=scores.index, columns=scores.columns, dtype=float)

    for avg in scores.columns:
        for row in scores.index:
            std_devs.loc[row, avg] = numpy.std(scores[avg][row]) 
    return std_devs

def std_results(scores, matrices):
    return (std_scores(scores), std_confusion_matrix(matrices))

def average_results(scores, matrices):
    return (average_scores(scores), average_confusion_matrix(matrices))

def print_results(scores, matrix):
    print(scores)
    print(matrix)

def save_cm_img(matrix, title, file, **kwargs):
    cm_display = metrics.ConfusionMatrixDisplay(matrix, **kwargs)
    cm_display.plot()
    pyplot.title(title)
    pyplot.savefig(file)

def save_hist_img(values, title, file, **kwargs):
    pyplot.clf()
    pyplot.hist(values, bins="auto", density=True, **kwargs)
    pyplot.title(title)
    pyplot.savefig(file)

def save_multi_hist_img(values, title, file, **kwargs):
    pyplot.clf()
    pyplot.hist(values, **kwargs, bins="auto", range=(0.0, 1.0), density=True,
                            fill=False, histtype="step")
    pyplot.title(title)
    pyplot.legend()
    pyplot.savefig(file)

measure_args = [
        "train_data",
        "test_data",
        "label_column",
        "binary",
        "multi_class",
        "verbose"
        ]

def measure_lgbm(train, test, args):
    binary = args.binary
    label = args.label_column
    verbose = args.verbose
    iterations = args.iterations


    if binary:
        print("Using Binary Labels\n")
    else:
        assert args.multi_class != None,  "Must set either binary (-b) or multi-class (-m [# classes])!"
        num_classes = args.multi_class
        print("Using Multi-Class Labels, with", num_classes, "classes.\n")

    # Setup classifier
    lgb_params = {
            "verbose": -1
            }
    if binary:
        lgb_params["objective"] = "binary"
        lgb_params["metric"] = "binary_logloss"
    else:
        lgb_params["objective"] = "multiclass"
        lgb_params["num_class"] = num_classes
        lgb_params["metric"] = "multi_logloss"

    print("LGBM parameters: ", lgb_params)

    classifier = lightgbm.LGBMClassifier(**lgb_params)

    if binary:
        scores, confusion_matrices = run_binary(train, test, classifier, args, iterations)
    else:
        scores, confusion_matrices = run_multi(train, test, classifier, args, iterations)
        
    return scores, confusion_matrices

def measure_linearSVC(train, test, args):
    binary = args.binary
    label = args.label_column
    verbose = args.verbose
    iterations = args.iterations

    classifier = SVC(**{
        "C": 0.01, "kernel": "linear", "probability": True})

    if binary:
        scores, confusion_matrices = run_binary(train, test, classifier, args, iterations)
    else:
        scores, confusion_matrices = run_multi(train, test, classifier, args, iterations)
        
    return scores, confusion_matrices

def measure_skl_lr(train, test, args):
    binary = args.binary
    label = args.label_column
    verbose = args.verbose
    iterations = args.iterations

    if binary:
        classifier = LogisticRegression(max_iter=10000)
        scores, confusion_matrices = run_binary(train, test, classifier, args, iterations)
    else:
        classifier = OneVsRestClassifier(LogisticRegression(max_iter=10000))
        scores, confusion_matrices = run_multi(train, test, classifier, args, iterations)
        
    return scores, confusion_matrices

measure_models = {
        "lgbm": measure_lgbm,
        "linearSVC": measure_linearSVC,
        "skl_LogReg": measure_skl_lr
        }

def measure(train, test, args):
    binary = args.binary
    label = args.label_column
    verbose = args.verbose
    iterations = args.iterations

    print("Train data:")
    print(train)
    print(train.shape)
    print("Test data:")
    print(test)
    print(test.shape)

    out = args.output_dir
    scores_json = {}

    for model in measure_models:
        model_scores = {}
        
        scores, confusion_matrices = measure_models[model](train, test, args)

        if iterations > 1:
            avg_scores, avg_matrix = average_results(scores, confusion_matrices)
            std_scores, std_matrix = std_results(scores, confusion_matrices)

        for score_type in scores.index:
            this_score = {}
            for avg in scores.columns:
                this_score[avg] = scores.loc[score_type, avg].tolist()
            model_scores[score_type] = this_score
                
        scores_json[model] = model_scores

        if iterations > 1:
            write_file(avg_scores, os.path.join(out, model + "_avg_scores.csv"))
            write_file(std_scores, os.path.join(out, model + "_std_scores.csv"))

        numpy.save(os.path.join(out, model + "_all_cms.npy"), numpy.asarray(confusion_matrices))
        if iterations > 1:
            numpy.savetxt(os.path.join(out, model + "_avg_cm.txt"), avg_matrix)
            numpy.savetxt(os.path.join(out, model + "_std_cm.txt"), std_matrix)
        else:
            numpy.savetxt(os.path.join(out, model + "_cm.txt"), confusion_matrices[0])

        if iterations > 1:
            save_cm_img(avg_matrix, f"Averaged Confusion Matrix ({model})", os.path.join(out, model + "_avg_cm.png"))
            save_cm_img(std_matrix, f"Std. Dev. for Confusion Matrix ({model})", os.path.join(out, model + "_std_cm.png"))
        else:
            save_cm_img(confusion_matrices[0], f"Confusion Matrix ({model})", os.path.join(out, model + "_cm.png"))

        if iterations > 1:
            print(model + "Averages over", iterations, "runs:")
            print_results(avg_scores, avg_matrix)
            print(model + "Std. Dev:")
            print_results(std_scores, std_matrix)
        else:
            print(model + "Results:")
            print_results(scores, confusion_matrices)

    write_file(scores_json, os.path.join(out, "all_scores.json"))

    return (scores, confusion_matrices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "Model Measurement")
    parser.add_argument("-train", "--train-data", required=True)
    parser.add_argument("-test", "--test-data", required=True)
    parser.add_argument("-l", "--label-column", default="label")
    parser.add_argument("-b", "--binary", action="store_true")
    parser.add_argument("-m", "--multi-class", type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-i", "--iterations", type=int, default=100)
    parser.add_argument("-o", "--output-dir")
    args = parser.parse_args()

    measure(read_file(args.train_data), read_file(args.test_data), args)



    
    

