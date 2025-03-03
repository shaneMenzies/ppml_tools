import argparse
import os
import numpy
import pandas

from ppml_utils import *

deprocess_args = [
        "spec",
        "output_dir"
        ]

def dediscretize(synth, pp_spec, output_dir):
    mean_dict = read_file(pp_spec["discretize"]["save_means"])
    exclude_columns = ["label", pp_spec["label_column"]]
    data_processed = pandas.DataFrame(dtype=float).reindex_like(synth)
    
    for col_i in range(len(synth.columns)):
        col = synth.columns[col_i]
        if col in exclude_columns:
            data_processed.loc[:, col] = synth.loc[:, col]
        else:
            # Retrieve the mean values for each bin in this column
            mean_labels = numpy.array(mean_dict[col])

            def dedisc_element(bin):
                if bin < len(mean_labels):
                    if numpy.isnan(mean_labels[bin]):
                        print(mean_labels)
                        print(f"Error! Column {col} has NaN for bin {bin}!!")
                        exit()
                    return mean_labels[bin]
                else:
                    print(f"Error! Column {col} doesn't have mean for bin {bin}!!")
        
    
            # Map each bin to its mean value for reconstruction
            data_processed.loc[:, col] = synth.loc[:, col].apply(dedisc_element)
    
    return data_processed

def relabel(synth, pp_spec):
    if "label_file" in pp_spec:
        pass
    elif "label_mapping" in pp_spec:
        original_label = pp_spec["label_column"]
        mapping = pp_spec["label_mapping"]
        labels = synth["label"].to_list()
        synth = synth.drop(columns="label")

        for i in range(len(labels)):
            label = labels[i]
            for (key, value) in mapping.items():
                if label == value:
                    labels[i] = key
                    break
        synth.insert(0, original_label, labels)
    return synth

def deprocess(synth, pp_spec, output_dir):
    synth = dediscretize(synth, pp_spec, output_dir)
    synth = relabel(synth, pp_spec)
    synth.reset_index(drop=True, inplace=True)
    write_file(synth, os.path.join(output_dir, "synthetic_deprocessed.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "Data Deprocesser")
    parser.add_argument('-s', '--spec', required=True)
    parser.add_argument('-o', '--output-dir', required=True)
    parser.add_argument('-d', '--synth-data')

    args = vars(parser.parse_args())

    pp_spec = read_file(args["spec"])

    if args["synth_data"] != None:
        synth = read_file(args["synth_data"])
    else:
        synth = read_file(os.path.join(args["output_dir"], "synthetic_data.csv"))
    deprocess(synth, pp_spec, args["output_dir"])
    
