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
from deprocess import deprocess
from ppml_utils import *

def pipeline(args):
    # Preprocess
    os.makedirs(args.output_dir, exist_ok=True)
    pp_spec = json.load(open(args.preprocess_spec))
    pp_result = preprocess(pp_spec)
    train, test, domain = pp_result["train"], pp_result["test"], pp_result["domain"]
    write_file(pp_result["train"], os.path.join(args.output_dir, "train_data.SENSITIVE.csv"), index=False)
    write_file(pp_result["test"], os.path.join(args.output_dir, "test_data.SENSITIVE.csv"), index=False)

    test_label_dist = test["label"]
    # clear up test data for generation
    del test

    # Generate
    mechanism_args = {
            "dataset": train,
            "domain": domain,
            "epsilon": args.epsilon,
            "output_dir": args.output_dir,
            "label_dist": test_label_dist
            }
    if args.delta != None:
        mechanism_args["delta"] = args.delta

    start_time = time.process_time()
    synth, error = run_mechanism(args.mechanism_dir, args.mechanism, mechanism_args)
    elapsed_time = time.process_time() - start_time
    print("Synthetic dataset generated in", elapsed_time, "s")
    write_file(synth, os.path.join(args.output_dir, "synthetic_data.csv"), index=False)
    write_file(numpy.array(error, ndmin=1), 
               os.path.join(args.output_dir, "workload_error.txt"),
               fmt="%1.8f")

    # Optional deprocess
    if args.deprocess:
        deprocess(synth, pp_spec, args.output_dir)

    if not args.skip_measure:
        # Measure
        measure_args = types.SimpleNamespace(
                label_column = "label",
                verbose = False,
                iterations = args.measure_iterations,
                output_dir = args.output_dir
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
    parser.add_argument("-e", "--epsilon", type=float)
    parser.add_argument("-d", "--delta", type=float)
    parser.add_argument("--seed")
    parser.add_argument("-i", "--measure-iterations", default=1, type=int)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--deprocess", action="store_true")
    parser.add_argument("--skip-measure", action="store_true")

    args = parser.parse_args()

    pipeline(args)

