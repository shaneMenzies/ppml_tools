import argparse
import os
import measure

from ppml_utils import *

file_paths = {
        "all_scores": "all_scores.json",
        "avg_scores": "avg_scores.csv",
        "all_cms": "all_cms.npy",
        }

def series_hist_graph(data, labels, title):
    for score in data.keys():
        i_data = data[score]
        i_labels = labels[score]
        measure.save_multi_hist_img(i_data, f"{title}\n{score}", f"{score}_comparison.png", label=i_labels, stacked=False)

graph_funcs = {
        "series_hist": series_hist_graph,
        }

def compare(directories, labels, graphs, title):
    if labels == None:
        labels = [str(i) for i in range(len(directories))]
    directories = zip(directories, labels)

    data = {}
    data_labels = {}
    for dir, label in directories:
        print(dir)
        dir_data = {}

        iteration_dirs = list(os.walk(dir))[1:]
        for iteration in iteration_dirs:
            i_scores = read_file(os.path.join(iteration[0], 
                                              file_paths["all_scores"]))
            for score in i_scores.keys():
                for avg_type in i_scores[score].keys():
                    if i_scores[score][avg_type][0] >= 0:
                        score_key = f"{score}_{avg_type}"
                        if score_key not in dir_data:
                            dir_data[score_key] = []
                        dir_data[score_key].extend(i_scores[score][avg_type])

        
        for score in dir_data.keys():
            if score not in data:
                data[score] = []
                data_labels[score] = []
            data[score].append(dir_data[score])
            data_labels[score].append(label)

    for graph in graphs:
        graph_funcs[graph](data, data_labels, title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "Result Comparison")
    parser.add_argument("-d", "--directories", action="extend", nargs="+", required=True)
    parser.add_argument("-l", "--labels", action="extend", nargs="+")
    parser.add_argument("-g", "--graphs", action="extend", nargs="+", default=["series_hist"])
    parser.add_argument("-t", "--title")
    args = parser.parse_args()

    compare(args.directories, args.labels, args.graphs, args.title)
