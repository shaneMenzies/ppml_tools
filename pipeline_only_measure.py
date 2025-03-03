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

    os.makedirs(args.output_dir, exist_ok=True)

    if args.data != None:
        synth = read_file(args.data)
    else:
        synth = read_file(os.path.join(args.output_dir, "synthetic_data.csv"))

    if args.domain != None:
        domain = read_file(args.domain)
    elif args.spec != None:
        pp_spec = read_file(args.spec)
        domain = read_file(pp_spec["output"]["domain"])
    test = read_file(os.path.join(args.output_dir, "test_data.SENSITIVE.csv"))

    # Optional deprocess
    if args.deprocess:
        pp_spec = read_file(args.spec)
        deprocess(synth, pp_spec, args.output_dir)

    # Measure
    measure_args = types.SimpleNamespace(
            label_column = "label",
            verbose = False,
            iterations = args.measure_iterations,
            output_dir = args.output_dir,
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
    parser = argparse.ArgumentParser(prog = "Pipeline (Only Measurement)")
    parser.add_argument("--data")
    parser.add_argument("--domain")
    parser.add_argument("-i", "--measure-iterations", default=1, type=int)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("-s", "--spec")
    parser.add_argument("--deprocess", action="store_true")

    args = parser.parse_args()

    pipeline(args)

