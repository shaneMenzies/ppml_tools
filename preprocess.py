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
        common_values = [val for val in filter if val in data.columns]
        for val in filter:
            if val not in common_values:
                print("Warning: Value '", val, "' in filter not found in data!")
        return data[common_values]
    else:
        return data

def load_data(spec):

    if isinstance(spec["data"], list):
        data_sources = []
        for file in spec["data"]:
            print("Loading from ", file, "(list)")
            next_df = read_file(file)
            if "transpose" in spec and spec["transpose"]:
                next_df = next_df.transpose()
                next_headers = next_df.iloc[0]
                next_df = next_df.iloc[1:]
                next_df.columns = list(next_headers)
            next_df = filter_data(next_df, spec)
            data_sources.append(next_df)
        return pandas.concat(data_sources, axis=0).dropna(axis=1)
    elif isinstance(spec["data"], dict):
        data = pandas.DataFrame()
        if "train" in spec["data"]:

            # Collect train
            for file_x, file_y in zip(spec["data"]["train"]["x"], 
                                      spec["data"]["train"]["y"]):
                print("Loading from ", file_x, "and", file_y)
                next_df = pandas.concat([read_file(file_y), 
                                        read_file(file_x)],
                                        axis=1)
                next_df = filter_data(next_df, spec)
                data = pandas.concat([data, next_df], axis=0)

            data = data.dropna(axis=1)
            if "test" in spec["data"]:
                # Collect test data
                spec["train_test_merge_row"] = len(data.index)
                for file_x, file_y in zip(spec["data"]["test"]["x"], 
                                          spec["data"]["test"]["y"]):
                    print("Loading from ", file_x, "and", file_y)
                    next_df = pandas.concat([read_file(file_y), 
                                            read_file(file_x)],
                                            axis=1)
                    next_df = filter_data(next_df, spec)
                    data = pandas.concat([data, next_df], axis=0)
                data = data.dropna(axis=1)

        else:
            for file_x, file_y in zip(spec["data"]["x"], 
                                      spec["data"]["y"]):
                print("Loading from ", file_x, "and", file_y)
                next_df = pandas.concat([read_file(file_y), 
                                        read_file(file_x)],
                                        axis=1)
                next_df = filter_data(next_df, spec)
                data = pandas.concat([data, next_df], axis=0)
            data = data.dropna(axis=1)
        return data
    else:
        print("Invalid data specifier in specification!!!")



def label_data(data, spec):
    if "label_file" in spec:
        label_file = read_file(spec["label_file"])
        id_column = spec["identifier_column"]
        label_column = spec["label_column"]
        mapping = spec["label_mapping"]
        print(f"Matching inputs {data.index} to labels from {label_file[id_column]}, and applying mapping {mapping}")

        labels = {}
        for source_id in data.index:
            if "use_jhu_id_transform" in spec and spec["use_jhu_id_transform"]:
                id = jhu_id_transform(id)
            else:
                id = source_id
            try:
                label = label_file[label_file[id_column] == id][label_column].values[0]
            except:
                print(f"Could not find match in label file")
                label = None
            if label in mapping:
                label = mapping[label]
            elif "default_label" in spec:
                label = spec["default_label"]
            else:
                print("Warning: Unable to determine correct label for ", label)
            labels[source_id] = label

        print(numpy.unique(numpy.array(list(labels.values())), return_counts=True))

        labels = pandas.Series(data=labels, name="label")
        return pandas.concat([labels, data], axis=1)
    elif "label_mapping" in spec:
        label_column = spec["label_column"]
        mapping = spec["label_mapping"]
        labels = data[label_column].to_list()
        data = data.drop(columns=spec["label_column"])
        for i in range(len(labels)):
            label = labels[i]
            if label in mapping:
                label = mapping[label]
            elif "default_label" in spec:
                print("Did not find mapping for ", label, 
                      ", using default label ", spec["default_mapping"])
                label = spec["default_mapping"]
            else:
                print("Warning: Unable to determine correct label for ", label)
            labels[i] = label
        data.insert(0, "label", labels)
    return data

def discretize_data(data, spec):
    if "discretize" in spec:
        if spec["label_column"] in data:
            had_labels = True
            label_col = data[spec["label_column"]]
            data = data.drop(columns=spec["label_column"])
        else:
            had_labels = False
        alphas = spec["discretize"]["quantiles"]
        bin_number = len(alphas) + 1
        data_copy = data.copy()
        data = data.apply(pandas.to_numeric, errors='coerce')
        data_quantile = numpy.quantile(data, alphas, axis=0)

        statistic_dict = {}
        mean_dict = {}
        quantile_dict = {}
        for col in data.columns:
            col_quantiles = data_quantile[:, data.columns.get_loc(col)]
            try:
                discrete_col = numpy.digitize(data[col], col_quantiles)
            except:
                print("Could not discretize column", col, data[col].values)
            data[col] = discrete_col
            quantile_dict[col] = col_quantiles

            statistic_dict[col] = []
            mean_dict[col] = []
            for bin_idx in range(bin_number):
                bin_arr = data_copy[col][discrete_col == bin_idx]
                statistic_dict[col].append(len(bin_arr))
                next_mean = numpy.mean(bin_arr)
                if numpy.isnan(next_mean):
                    # Estimate mean 
                    if bin_idx == 0:
                        next_mean = (numpy.min(data_copy[col]) + col_quantiles[0]) / 2
                    elif bin_idx == (bin_number - 1):
                        next_mean = (numpy.max(data_copy[col]) + col_quantiles[-1]) / 2
                    else:
                        next_mean = (col_quantiles[bin_idx] + col_quantiles[bin_idx + 1]) / 2
                mean_dict[col].append(next_mean)

        if had_labels:
            data = pandas.concat([label_col, data], axis=1)
        
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
    print("Loaded data:\n", data)
    data = discretize_data(data, spec)
    print("Data after discretization:\n", data)
    data = label_data(data, spec)
    print("Labelled data:\n", data)

    print("Shape of combined, labelled data (rows, columns): ", data.shape)

    results = {}
    if "domain" in spec["output"]:
        domain = {}
        for col in data.columns:
            if col == "label" or ("label_column" in spec and col == spec["label_column"]):
                domain[str(col)] = data[col].nunique()
            elif "discretize" in spec:
                domain[str(col)] = len(spec["discretize"]["quantiles"]) + 1
            elif "non_label_domain" in spec:
                domain[str(col)] = spec["non_label_domain"]
            else:
                domain[str(col)] = data[col].nunique()
                
        results["domain"] = domain
        write_file(domain, spec["output"]["domain"])
        print("Domain written to ", spec["output"]["domain"])
    
    if "processed_data" in spec["output"]:
        results["combined_data"] = data
        write_file(data, spec["output"]["processed_data"], index=False)
        print("Data written to ", spec["output"]["processed_data"])

    train, test = split_data(data, spec)
    results["train"] = train
    results["test"] = test
    return results

def split_data(data, spec):
    if "split" in spec:
        if "test_ratio" in spec["split"]:
            if "seed" in spec["split"]:
                print("Using", spec["split"]["seed"], 
                      "as seed for train/test split")
                seed = spec["split"]["seed"]
            else:
                seed = 4000 #randbits(32)
            print("Using random seed", seed, "for train/test split")
            train, test = train_test_split(
                    data, 
                    test_size=spec["split"]["test_ratio"], #shuffle=True, 
                    random_state=seed #,stratify=data['label']
                    )
        elif "train_test_merge_row" in spec:
            train_test_merge_first_test = spec["train_test_merge_row"]
            train = data.iloc[:train_test_merge_first_test]
            test = data.iloc[train_test_merge_first_test:]
        else:
            train = data
            test = None

        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        print("Shape of training data (rows, columns): ", train.shape)
        print("Shape of test data (rows, columns): ", test.shape)
        if "train_data" in spec["split"]:
            write_file(train, spec["split"]["train_data"], index=False)
            print("Training data written to ", spec["split"]["train_data"])
        if "test_data" in spec["split"] and isinstance(test, pandas.DataFrame):
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




    
    



