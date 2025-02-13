import argparse
import numpy
import pandas
import measure
import glob
import os
import json

from sklearn import metrics
from ppml_utils import *

file_paths = {
        "all_scores": "all_scores.json",
        "avg_scores": "avg_scores.csv",
        "all_cms": "all_cms.npy",
        }

def collate(directories, title=None):

    if title != None:
        title = '(' + title + ')'
    else:
        title = ""

    scores = []
    matrices = []
    for dir in directories:
        print(dir)
        try:
            file = read_file(os.path.join(dir, file_paths["all_scores"]))
            scores.append(file)
        except:
            print("\tScores file not found in", dir)

        try:
            file = read_file(os.path.join(dir, file_paths["all_cms"]))
            matrices.extend(file)
        except:
            print("\tConfusion matrices file not found in", dir)

    for i in range(len(scores)):
        for score in scores[i].keys():
            for avg, values in dict(scores[i][score]).items():
                if values[0] < 0:
                    scores[i][score].pop(avg)

    averages = {}
    for score in scores[0]:
        averages[score] = {}
        for avg in scores[0][score]:
            sum = 0.0
            for i in scores:
                sum += numpy.mean(numpy.asarray(i[score][avg]))
            averages[score][avg] = sum / len(scores)

    print(averages)
    write_file(averages, "avg_scores_collated.json")

    for score in scores[0]:
        for avg in scores[0][score]:
            measure.save_multi_hist_img(numpy.concatenate([i[score][avg] for i in scores]), 
                                        f"{score} - {avg}\n{title}", 
                                        f"{score}_{avg}_collated.png")

    average_cms = measure.average_confusion_matrix(matrices)
    std_cms = measure.std_confusion_matrix(matrices)
    print("Average Confusion Matrix", title, ":\n", average_cms)
    print("Std. Dev. Confusion Matrix", title, ":\n", std_cms)

    measure.save_cm_img(average_cms, f"Average Confusion Matrix of {len(scores)} iterations \n{title}", "avg_cms.png")
    measure.save_cm_img(std_cms, f"Std. Dev. Confusion Matrix of {len(scores)} iterations\n{title}", "std_cms.png")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "Result Collation")
    parser.add_argument("directory", action="extend", nargs="+", default=["./"])
    parser.add_argument("-t", "--title")
    args = parser.parse_args()

    collate(args.directory, title=args.title)






