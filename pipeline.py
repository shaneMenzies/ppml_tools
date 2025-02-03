import argparse
import time
import pandas
import numpy
import json
import joblib
import types
import os

from preprocess import preprocess
from generate import run_mechanism
from measure import measure

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

def pipeline(args):
    # Preprocess
    pp_spec = json.load(open(args.preprocess_spec))
    pp_result = preprocess(pp_spec)
    train, test, domain = pp_result["train"], pp_result["test"], pp_result["domain"]

    # Generate
    mechanism_args = {
            "dataset": train,
            "domain": domain,
            "epsilon": args.epsilon,
            }
    if args.delta != None:
        mechanism_args["delta"] = args.delta

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.process_time()
    synth = run_mechanism(args.mechanism_dir, args.mechanism, mechanism_args)
    elapsed_time = time.process_time() - start_time
    print("Synthetic dataset generated in", elapsed_time, "s")
    write_file(synth, os.path.join(args.output_dir, "synthetic_data.csv"), index=False)

    # Measure
    measure_args = types.SimpleNamespace(
            label_column = "label",
            verbose = False,
            iterations = args.measure_iterations,
            save_all_scores = os.path.join(args.output_dir, "all_scores.csv"),
            save_all_cms = os.path.join(args.output_dir, "all_cms.npy"),
            save_avg_scores = os.path.join(args.output_dir, "avg_scores.csv"),
            save_avg_cm = os.path.join(args.output_dir, "avg_cm.txt"),
            save_std_scores = os.path.join(args.output_dir, "std_scores.csv"),
            save_std_cm = os.path.join(args.output_dir, "std_cm.txt"),
            avg_cm_img = os.path.join(args.output_dir, "avg_cm.png"),
            std_cm_img = os.path.join(args.output_dir, "std_cm.png"),
            )

    if domain["label"] == 2:
        measure_args.binary = True
    else:
        measure_args.binary = False
        measure_args.multi_class = domain["label"]
    
    scores, confusion_matrices = measure(synth, test, measure_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "Pipeline")
    parser.add_argument("-pps", "--preprocess-spec", required=True)
    parser.add_argument("--mechanism-dir", default="mechanisms")
    parser.add_argument("-m", "--mechanism", required=True)
    parser.add_argument("-e", "--epsilon", required=True, type=float)
    parser.add_argument("-d", "--delta", type=float)
    parser.add_argument("--seed")
    parser.add_argument("-i", "--measure-iterations", default=2500, type=int)
    parser.add_argument("-o", "--output-dir", required=True)

    args = parser.parse_args()

    pipeline(args)

