import argparse
import numpy
import pandas
import lightgbm
from secrets import randbits
from sklearn import metrics

file_readers = {
        "csv": lambda file: pandas.read_csv(file),
        "tsv": lambda file: pandas.read_csv(file, sep='\t'),
        "xlsx": lambda file: pandas.read_excel(file)
        }

file_writers = {
        "csv": lambda data, file, **kwargs: data.to_csv(file, **kwargs),
        "tsv": lambda data, file, **kwargs: data.to_csv(file, **kwargs, sep='\t'),
        "xlsx": lambda data, file, **kwargs: data.to_excel(file, **kwargs),
        "joblib": lambda data, file, **kwargs: joblib.dump(data, file, **kwargs),
        "json": lambda data, file, **kwargs: json.dump(data, open(file, 'w'), **kwargs, indent=4)
        }

def read_file(path):
    _, _, suffix = path.rpartition('.')
    return file_readers[suffix](path)

def write_file(data, path, **kwargs):
    _, _, suffix = path.rpartition('.')
    file_writers[suffix](data, path, **kwargs)

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
        scores.loc["auc", avg] = metrics.roc_auc_score(test_y, predictions, average=avg)
    
    confusion_matrix = metrics.confusion_matrix(test_y, predictions)
    return (scores, confusion_matrix)

def run_binary(train_x, train_y, test_x, test_y, lgb_params, iterations):
    scores = pandas.DataFrame(index=binary_score_types, 
                              columns=average_types, 
                              dtype=object
                              )
    confusion_matrices = []

    for col in scores.columns:
        for row in scores.index:
            scores.loc[row, col] = numpy.empty(iterations, dtype=float)

    for i in range(iterations):
        i_scores, i_cm = single_run_binary(train_x, train_y, test_x, test_y, lgb_params)
        for col in scores.columns:
            for row in scores.index:
                scores.loc[row, col][i] = i_scores[col][row]
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

def run_multi(train_x, train_y, test_x, test_y, lgb_params, iterations):
    scores = pandas.DataFrame(index=multi_score_types, 
                              columns=average_types,
                              dtype=object)
    confusion_matrices = []

    for col in scores.columns:
        for row in scores.index:
            scores.loc[row, col] = numpy.empty(iterations, dtype=float)

    for i in range(iterations):
        i_scores, i_cm = single_run_multi(train_x, train_y, test_x, test_y, lgb_params)
        for col in scores.columns:
            for row in scores.index:
                scores.loc[row, col][i] = i_scores[col][row]
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

def save_cm_img(matrix, title, file):
    import matplotlib
    cm_display = metrics.ConfusionMatrixDisplay(matrix)
    cm_display.plot()
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.savefig(file)


measure_args = [
        "train_data",
        "test_data",
        "label_column",
        "binary",
        "multi_class",
        "verbose"
        ]

def measure(train, test, args):
    model_data = train
    test_data = test
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

    # Split label column off
    train_x = model_data.drop(columns=[label])
    train_y = model_data[label]
    test_x = test_data.drop(columns=[label])
    test_y = test_data[label]

    if verbose:
        print("Train X: ", train_x)
        print("Train Y: ", train_y)
        print("Test X: ", test_x)
        print("Test Y: ", test_y)

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
        scores, confusion_matrices = run_binary(train_x, train_y, test_x, test_y, lgb_params, iterations)
    else:
        scores, confusion_matrices = run_multi(train_x, train_y, test_x, test_y, lgb_params, iterations)

    avg_scores, avg_matrix = average_results(scores, confusion_matrices)
    std_scores, std_matrix = std_results(scores, confusion_matrices)

    if args.save_all_scores != None:
        write_file(scores, args.save_all_scores)
    if args.save_avg_scores != None:
        write_file(avg_scores, args.save_avg_scores)
    if args.save_std_scores != None:
        write_file(std_scores, args.save_std_scores)

    if args.save_all_cms != None:
        numpy.save(args.save_all_cms, numpy.asarray(confusion_matrices))
    if args.save_avg_cm != None:
        numpy.savetxt(args.save_avg_cm, avg_matrix)
    if args.save_std_cm != None:
        numpy.savetxt(args.save_std_cm, std_matrix)

    if args.avg_cm_img != None:
        save_cm_img(avg_matrix, "Averaged Confusion Matrix", args.avg_cm_img)
    if args.std_cm_img != None:
        save_cm_img(std_matrix, "Std. Dev. for Confusion Matrix", args.std_cm_img)

    print("Averages over", iterations, "runs:")
    print_results(avg_scores, avg_matrix)
    print("Std. Dev:")
    print_results(std_scores, std_matrix)
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
    parser.add_argument("--save-all-scores")
    parser.add_argument("--save-all-cms")
    parser.add_argument("--save-avg-scores")
    parser.add_argument("--save-avg-cm")
    parser.add_argument("--save-std-scores")
    parser.add_argument("--save-std-cm")
    parser.add_argument("--avg-cm-img")
    parser.add_argument("--std-cm-img")
    args = parser.parse_args()

    measure(read_file(args.train_data), read_file(args.test_data), args)



    
    

