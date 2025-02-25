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
from ppml_utils import *

average_types = ["micro", "macro", "weighted"]

binary_score_types = ["f1", "recall", "auc"]
multi_score_types = ["f1", "recall", "auc_ovo", "auc_ovr"]

def single_run_binary(train_x, train_y, test_x, test_y, lgb_params):
    lgb_params["seed"] = randbits(64)
    classifier = lightgbm.LGBMClassifier(**lgb_params)
    classifier.fit(train_x, train_y)

    predictions = classifier.predict(test_x)
    prob_preds = classifier.predict_proba(test_x)

    scores = pandas.DataFrame(index=binary_score_types, 
                              columns=average_types, dtype=float)
    for avg in average_types:
        scores.loc["f1", avg] = metrics.f1_score(test_y, predictions, average=avg)
        scores.loc["recall", avg] = metrics.recall_score(test_y, predictions, average=avg)
        try:
            scores.loc["auc", avg] = metrics.roc_auc_score(test_y, predictions, average=avg)
        except:
            scores.loc["auc", avg] = -1.0
    
    confusion_matrix = metrics.confusion_matrix(test_y, predictions)
    return (scores, confusion_matrix)

def run_binary(train, test, lgb_params, args, iterations):
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

            i_scores, i_cm = single_run_binary(train_x, train_y, test_x, test_y, lgb_params)
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

        scores, i_cm = single_run_binary(train_x, train_y, test_x, test_y, lgb_params)
        confusion_matrices.append(i_cm)

    return (scores, confusion_matrices)

def run_binary_cv(train_x, train_y, lgb_params, iterations, cv_folds=5):
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
                                               lgb_params)
            for col in scores.columns:
                for row in scores.index:
                    scores.loc[row, col][i_base + i_fold] = i_scores[col][row]
            confusion_matrices.append(i_cm)

    return (scores, confusion_matrices)

def single_run_multi(train_x, train_y, test_x, test_y, lgb_params):
    lgb_params["seed"] = randbits(64)
    classifier = lightgbm.LGBMClassifier(**lgb_params)
    classifier.fit(train_x, train_y)

    predictions = classifier.predict(test_x)
    prob_preds = classifier.predict_proba(test_x)

    scores = pandas.DataFrame(index=multi_score_types, 
                              columns=average_types, dtype=float)
    for avg in average_types:
        scores.loc["f1", avg] = metrics.f1_score(
                test_y, predictions, average=avg)
        scores.loc["recall", avg] = metrics.recall_score(
                test_y, predictions, average=avg)
        if avg != "micro":
            scores.loc["auc_ovo", avg] = metrics.roc_auc_score(
                    test_y, prob_preds, average=avg, multi_class="ovo")
        else:
            scores.loc["auc_ovo", avg] = -1.0
        scores.loc["auc_ovr", avg] = metrics.roc_auc_score(
                test_y, prob_preds, average=avg, multi_class="ovr")

    confusion_matrix = metrics.confusion_matrix(test_y, predictions)
    return (scores, confusion_matrix)

def run_multi(train, test, lgb_params, args, iterations):
    scores = pandas.DataFrame(index=multi_score_types, 
                              columns=average_types,
                              dtype=object)
    confusion_matrices = []
    label = args.label_column
    print(iterations, "measure iterations")

    for col in scores.columns:
        for row in scores.index:
            scores.loc[row, col] = numpy.empty(iterations, dtype=float)

    if iterations > 1:
        for i in range(iterations):
            shuffle_train = train.sample(frac=1)
            shuffle_test = test.sample(frac=1)

            train_x = shuffle_train.drop(columns=[label])
            train_y = shuffle_train[label]
            test_x = shuffle_test.drop(columns=[label])
            test_y = shuffle_test[label]

            i_scores, i_cm = single_run_multi(train_x, train_y, test_x, test_y, lgb_params)
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

        i_scores, i_cm = single_run_multi(train_x, train_y, test_x, test_y, lgb_params)
        for col in scores.columns:
            for row in scores.index:
                scores.loc[row, col] = numpy.array(i_scores[col][row], ndmin=1)
        confusion_matrices.append(i_cm)

    return (scores, confusion_matrices)

def run_multi_cv(train_x, train_y, lgb_params, iterations, cv_folds=5):
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
                                               lgb_params)
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

def measure(train, test, args):
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

    if binary:
        scores, confusion_matrices = run_binary(train, test, lgb_params, args, iterations)
    else:
        scores, confusion_matrices = run_multi(train, test, lgb_params, args, iterations)

    if iterations > 1:
        avg_scores, avg_matrix = average_results(scores, confusion_matrices)
        std_scores, std_matrix = std_results(scores, confusion_matrices)

    out = args.output_dir

    scores_json = {}
    for score_type in scores.index:
        this_score = {}
        for avg in scores.columns:
            this_score[avg] = scores.loc[score_type, avg].tolist()
        scores_json[score_type] = this_score
            

    write_file(scores_json, os.path.join(out, "all_scores.json"))
    if iterations > 1:
        write_file(avg_scores, os.path.join(out, "avg_scores.csv"))
        write_file(std_scores, os.path.join(out, "std_scores.csv"))

    numpy.save(os.path.join(out, "all_cms.npy"), numpy.asarray(confusion_matrices))
    if iterations > 1:
        numpy.savetxt(os.path.join(out, "avg_cm.txt"), avg_matrix)
        numpy.savetxt(os.path.join(out, "std_cm.txt"), std_matrix)
    else:
        numpy.savetxt(os.path.join(out, "cm.txt"), confusion_matrices[0])

    if iterations > 1:
        save_cm_img(avg_matrix, "Averaged Confusion Matrix", os.path.join(out, "avg_cm.png"))
        save_cm_img(std_matrix, "Std. Dev. for Confusion Matrix", os.path.join(out, "std_cm.png"))
    else:
        save_cm_img(confusion_matrices[0], "Confusion Matrix", os.path.join(out, "cm.png"))

    if iterations > 1:
        print("Averages over", iterations, "runs:")
        print_results(avg_scores, avg_matrix)
        print("Std. Dev:")
        print_results(std_scores, std_matrix)
    else:
        print("Results:")
        print_results(scores, confusion_matrices)
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



    
    

