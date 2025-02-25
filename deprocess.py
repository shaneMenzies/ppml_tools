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
    print(mean_dict)
    exclude_columns = ["label", pp_spec["label_column"]]
    data_processed = []
    
    for col in synth.columns:
        if col in exclude_columns:
            # Append the unchanged column
            data_processed.append(synth[col])
        else:
            # Retrieve the mean values for each bin in this column
            mean_labels = numpy.array(mean_dict[col])
    
            # Map each bin to its mean value for reconstruction
            col_inverse = synth[col].apply(
                    lambda x: mean_labels[x] if x < len(mean_labels) 
                                             else numpy.nan)
            data_processed.append(col_inverse)
    
    # Combine all columns back into a DataFrame
    data_processed_df = pandas.DataFrame(data_processed).T
    data_processed_df.columns = synth.columns
    
    return data_processed_df 

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
    pp_spec = read_file(pp_spec)

    synth = relabel(synth, pp_spec)
    synth = dediscretize(synth, pp_spec, output_dir)
    synth.reset_index(drop=True, inplace=True)
    write_file(synth, os.path.join(output_dir, "synthetic_deprocessed.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "Data Deprocesser")
    parser.add_argument('-s', '--spec', required=True)
    parser.add_argument('-o', '--output-dir', required=True)
    parser.add_argument('-d', '--synth-data')

    args = vars(parser.parse_args())
    if args["synth_data"] != None:
        synth = read_file(args["synth_data"])
    else:
        synth = read_file(os.path.join(args["output_dir"], "synthetic_data.csv"))
    deprocess(synth, args["spec"], args["output_dir"])
    
