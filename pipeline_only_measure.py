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
from ppml_utils import *

def pipeline(args):

    os.makedirs(args.output_dir, exist_ok=True)
    synth = read_file(args.data)
    domain = read_file(args.domain)
    test = read_file(os.path.join(args.output_dir, "test_data.SENSITIVE.csv"))

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
    parser.add_argument("--data", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("-i", "--measure-iterations", default=1, type=int)
    parser.add_argument("-o", "--output-dir", required=True)

    args = parser.parse_args()

    pipeline(args)

