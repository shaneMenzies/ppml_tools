import argparse
import pandas
import numpy 
import glob
import json
import re
import joblib

from secrets import randbits
from sklearn.model_selection import train_test_split
from ppml_utils import *

def jhu_id_transform(id):
    return '-'.join(id.split('.')[:4])

def filter_data(data, spec):
    if "filter" in spec:
        filter = read_file(spec["filter"])[["Symbol"]].values.flatten()
        common_values = [val for val in filter if val in data.iloc[0]]
        for val in filter:
            if val not in common_values:
                print("Warning: Value '", val, "' in filter not found in data!")
        return data[common_values]
    else:
        return data

def load_data(spec):
    data_sources = []
    for file in spec["data"]:
        print("Loading from ", file)
        next_df = read_file(file).transpose()
        next_headers = next_df.iloc[1]
        next_df = next_df.iloc[2:]
        next_df.columns = list(next_headers)
        next_df = filter_data(next_df, spec)
        data_sources.append(next_df)
    return pandas.concat(data_sources, axis=0).dropna(axis=1)

def label_data(data, spec):
    label_file = read_file(spec["label_file"])
    id_column = spec["identifier_column"]
    label_column = spec["label_column"]
    mapping = spec["label_mapping"]

    labels = {}
    for source_id in data.index:
        id = jhu_id_transform(source_id)
        label = label_file[label_file[id_column] == id][label_column].values[0]
        if label in mapping:
            label = mapping[label]
        elif "default_label" in spec:
            print("Did not find mapping for ", label, 
                  ", using default label ", spec["default_mapping"])
            label = spec["default_mapping"]
        else:
            print("Warning: Unable to determine correct label for ", label)
        labels[source_id] = label
    labels = pandas.Series(data=labels, name="label")
    return pandas.concat([labels, data], axis=1)

def discretize_data(data, spec):
    if "discretize" in spec:
        alphas = spec["discretize"]["quantiles"]
        bin_number = len(alphas) + 1
        data_copy = data.copy()
        data_quantile = numpy.quantile(data, alphas, axis=0)

        statistic_dict = {}
        mean_dict = {}
        quantile_dict = {}
        for col in data.columns:
            if col != 'label':
                col_quantiles = data_quantile[:, data.columns.get_loc(col)]
                discrete_col = numpy.digitize(data[col], col_quantiles)
                data[col] = discrete_col
                quantile_dict[col] = col_quantiles

                statistic_dict[col] = []
                mean_dict[col] = []
                for bin_idx in range(bin_number):
                    bin_arr = data_copy[col][discrete_col == bin_idx]
                    statistic_dict[col].append(len(bin_arr))
                    mean_dict[col].append(numpy.mean(bin_arr) if len(bin_arr) > 0 else numpy.nan)
        
        if "save_statistics" in spec["discretize"]:
            write_file(statistic_dict, spec["discretize"]["save_statistics"])
            print("Statistics data written to ", spec["discretize"]["save_statistics"])
        if "save_means" in spec["discretize"]:
            write_file(mean_dict, spec["discretize"]["save_means"])
            print("Means data written to ", spec["discretize"]["save_means"])
        if "save_quantiles" in spec["discretize"]:
            write_file(quantile_dict, spec["discretize"]["save_quantiles"])
            print("Quantile data written to ", spec["discretize"]["save_quantiles"])
    return data

def preprocess(spec):
    data = load_data(spec)
    data = data.apply(pandas.to_numeric, errors='coerce')
    data = discretize_data(data, spec)
    data = label_data(data, spec)

    print("Shape of combined, labelled data (rows, columns): ", data.shape)

    results = {}
    if "domain" in spec["output"]:
        domain = {}
        for col in data.columns:
            domain[str(col)] = data[col].nunique()
        results["domain"] = domain
        write_file(domain, spec["output"]["domain"])
        print("Domain written to ", spec["output"]["domain"])
    
    if "processed_data" in spec["output"]:
        results["combined_data"] = data
        write_file(data, spec["output"]["processed_data"], index=False)
        print("Data written to ", spec["output"]["processed_data"])

    train, test = split_data(data, spec)
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    results["train"] = train
    results["test"] = test
    return results

def split_data(data, spec):
    if "split" in spec:
        if "seed" in spec["split"]:
            print("Using", spec["split"]["seed"], 
                  "as seed for train/test split")
            seed = spec["split"]["seed"]
        else:
            seed = randbits(32)
            print("Using random seed", seed, "for train/test split")
        train, test = train_test_split(
                data, 
                test_size=spec["split"]["test_ratio"], shuffle=True, 
                random_state=seed, stratify=data['label']
                )
        print("Shape of training data (rows, columns): ", train.shape)
        print("Shape of test data (rows, columns): ", test.shape)
        if "train_data" in spec["split"]:
            write_file(train, spec["split"]["train_data"], index=False)
            print("Training data written to ", spec["split"]["train_data"])
        if "test_data" in spec["split"]:
            write_file(test, spec["split"]["test_data"], index=False)
            print("Testing data written to ", spec["split"]["test_data"])

        return (train, test)
    else:
        return (data, None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "Data Preprocesser")
    parser.add_argument('-s', '--spec', required=True)

    args = parser.parse_args()
    spec = json.load(open(args.spec))
    preprocess(spec)




    
    



