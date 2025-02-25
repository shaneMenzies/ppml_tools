import argparse
import itertools
import importlib
import numpy
import pandas
import json
import sys
import os
import traceback

from secrets import randbits
from ppml_utils import *

mechanism_args = [
        "dataset",
        "domain",
        "label_col",
        "epsilon",
        "delta",
        "rng_seed",
        "degree",
        "max_cells",
        "num_marginals",
        "output_dir"
        ]

default_args = {
        "dataset": None,
        "domain": None,
        "label_col": "label",
        "epsilon": 1.0,
        "delta": 1e-8,
        "rng_seed": None,
        "degree": 2,
        "max_cells": 10000,
        "num_marginals": None,
        "output_dir": "."
        }

def fill_with_defaults(args):
    for key in default_args:
        if key not in args:
            args[key] = default_args[key]

    if args["rng_seed"] is None:
        args["rng_seed"] = randbits(64)
        print("Generate step using", args["rng_seed"], "for rng seed")
    return args

def determine_workload(args, data):
    workload = list(itertools.combinations(data.domain, args["degree"]))
    workload = [cl for cl in workload if data.domain.size(cl) <= args["max_cells"]]
    if args["num_marginals"] is not None:
        workload = [
            workload[i]
            for i in prng.choice(len(workload), args["num_marginals"], replace=False)
        ]
    return workload

def make_dataset(data, domain):
    from mbi import Dataset, Domain
    # Should already be loaded from a file
    return Dataset(data, Domain(domain.keys(), domain.values()))

def workload_error(real, synth, workload):
    error = -1.0
    try:
        errors = []
        for proj, wgt in workload:
            X = real.project(proj).datavector()
            Y = synth.project(proj).datavector()
            X_sum = X.sum()
            Y_sum = Y.sum()
            if X_sum != 0:
                X = X / X_sum
            if Y_sum != 0:
                Y = Y / Y_sum
            e = 0.5 * numpy.linalg.norm(X - Y, 1)
            errors.append(e)
        error = numpy.mean(numpy.asarray(errors))
        print("Average Workload Error:", error)
    except Exception as exc:
        print("An exception occured in calculating workload error.")
        print(type(exc))
        print(exc)

    return error


def mech_aim(args): 
    from mbi import Dataset, Domain
    aim = importlib.import_module("aim")
    data = make_dataset(args["dataset"], args["domain"])
    workload = determine_workload(args, data)
    aim_workload = [(cl, 1.0) for cl in workload]
    model, synth = aim.AIM(args["epsilon"], args["delta"]).run(data, aim_workload)
    error = workload_error(data, synth, workload)
    return synth.df, error 

def mech_mst(args):
    from mbi import Dataset, Domain
    mst = importlib.import_module("mst")
    data = make_dataset(args["dataset"], args["domain"])
    workload = determine_workload(args, data)
    synth = mst.MST(data, args["epsilon"], args["delta"]) 
    error = workload_error(data, synth, workload)
    return synth.df, error 

def mech_mwem(args):
    from mbi import Dataset, Domain
    mwem = importlib.import_module("mwem+pgm")
    data = make_dataset(args["dataset"], args["domain"])
    workload = determine_workload(args, data)
    synth = mwem.mwem_pgm(data, args["epsilon"], args["delta"],
                              workload=workload)
    error = workload_error(data, synth, workload)
    return synth.df, error

def mech_pgg(args):
    pgg = importlib.import_module("model")
    pgg_model = pgg.Private_PGM(args["label_col"], True, 
                            args["epsilon"], args["delta"])
    pgg_model.train(args["dataset"], args["domain"], num_iters=2500)
    synth = pgg_model.generate(num_rows=args["dataset"].shape[0])
    # pgg always puts label column at the end
    synth[:, 0], synth[:, -1] = synth[:, -1].copy(), synth[:, 0].copy()
    synth = pandas.DataFrame(synth, columns=args["dataset"].columns)

    data = make_dataset(args["dataset"], args["domain"])
    synth_ds = make_dataset(synth, args["domain"])
    workload = determine_workload(args, data)
    error = workload_error(data, synth_ds, workload)
    return synth, error

def mech_pgg_nodp(args):
    pgg = importlib.import_module("model")
    pgg_model = pgg.Private_PGM(args["label_col"], False, 8, 1e-5)
                            
    pgg_model.train(args["dataset"], args["domain"], num_iters=2500)
    synth = pgg_model.generate(num_rows=args["dataset"].shape[0])
    # pgg always puts label column at the end
    synth[:, 0], synth[:, -1] = synth[:, -1].copy(), synth[:, 0].copy()
    synth = pandas.DataFrame(synth, columns=args["dataset"].columns)

    data = make_dataset(args["dataset"], args["domain"])
    synth_ds = make_dataset(synth, args["domain"])
    workload = determine_workload(args, data)
    error = workload_error(data, synth_ds, workload)
    return synth, error

def mech_real(args):
    real_data = args["dataset"]
    return real_data, -1.0

def mech_tabddpm(tabddpm_dir, args):
    import tomli
    import tomli_w
    import subprocess
    output_dir = args["output_dir"]
    train = args["dataset"]
    train_y = train["label"].values
    train_x = train.drop(columns=["label"]).values
    os.makedirs(os.path.join(output_dir, "Data/ppml_pipeline"), exist_ok=True)
    numpy.save(os.path.join(output_dir, "Data/ppml_pipeline/X_num_train.npy"), train_x)
    numpy.save(os.path.join(output_dir, "Data/ppml_pipeline/y_train.npy"), train_y)
    os.makedirs(os.path.join(output_dir, "Results/ppml_pipeline"), exist_ok=True)
    numpy.save(os.path.join(output_dir, "Results/ppml_pipeline/X_num_train.npy"), train_x)
    numpy.save(os.path.join(output_dir, "Results/ppml_pipeline/y_train.npy"), train_y)

    # dummy test and validation files, won't be using tabddpm's evaluation
    data_out_dir = os.path.join(output_dir, "Data/ppml_pipeline/")
    numpy.save(data_out_dir + "y_val.npy", train_y)
    numpy.save(data_out_dir + "y_test.npy", train_y)
    numpy.save(data_out_dir + "X_num_val.npy", train_x)
    numpy.save(data_out_dir + "X_num_test.npy", train_x)

    data_info = {
            "name": "ppml_pipeline",
            "id": "ppml_pipeline",
            "task_type": "binclass",
            "n_num_features": train_x.shape[1],
            "n_cat_features": 0,
            "train_size": len(train_y)
            }
    write_file(data_info, os.path.join(output_dir, "Data/ppml_pipeline/info.json"))

    # Load default config
    tabddpm_config = {}
    with open("tabddpm_config.toml", "rb") as f:
        tabddpm_config = tomli.load(f)

    tabddpm_config["parent_dir"] = os.path.join(output_dir, "exp/ppml_pipeline/results")
    tabddpm_config["real_data_path"] = os.path.join(output_dir, "Data/ppml_pipeline/")
    tabddpm_config["num_numerical_features"] = train_x.shape[1]
    tabddpm_config["model_params"]["num_classes"] = len(numpy.unique(train_y))
    tabddpm_config["sample"]["num_samples"] = len(train_y)

    os.makedirs(os.path.join(output_dir, "exp/ppml_pipeline"), exist_ok=True)
    with open(os.path.join(output_dir, "exp/ppml_pipeline/config.toml"), "wb") as f:
        tomli_w.dump(tabddpm_config, f)

    # Run tabddpm
    command = ["python3 " + os.path.join(tabddpm_dir, "scripts/pipeline.py")
                    + " --config " 
                    + os.path.join(output_dir, "exp/ppml_pipeline/config.toml")
                    + " --train --sample"]
    print("Running TabDDPM, Command:", command)
    subprocess.run(command, shell=True)

    # Load synthetic data
    synth_x = read_file(os.path.join(output_dir, "exp/ppml_pipeline/results/X_num_train.npy"))
    synth_y = read_file(os.path.join(output_dir, "exp/ppml_pipeline/results/y_train.npy"))

    synth = pandas.DataFrame(
            numpy.hstack([synth_y, synth_x]), 
            columns=args["dataset"].columns)

    return synth, -1.0

mechanisms = {
        "aim": mech_aim,
        "mst": mech_mst, 
        "mwem": mech_mwem, 
        "pro_gene_gen": mech_pgg,
        "pro_gene_gen_nodp": mech_pgg_nodp,
        "real": mech_real,
        "tabddpm": mech_tabddpm,
        }

def run_mechanism(directory, mechanism, args):
    sys.path.insert(0, directory)
    if mechanism == "tabddpm":
        return mech_tabddpm(directory, args)
    else:
        if mechanism == "pro_gene_gen" or mechanism == "pro_gene_gen_nodp":
            sys.path.insert(0, os.path.join(directory, "../.."))
    return mechanisms[mechanism](fill_with_defaults(args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "Mechanism Wrapper")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--epsilon", required=True, type=float)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--rng-seed", type=int)
    parser.add_argument("--mechanism-dir", default="mechanisms")
    parser.add_argument("mechanism")
    parser.add_argument("--save")
    args = parser.parse_args()

    mech_args = {
            "dataset": read_file(args.dataset),
            "domain": read_file(args.domain),
            "label_col": args.label_col,
            "epsilon": args.epsilon,
            }
    if args.delta != None:
        mech_args["delta"] = args.delta
    if args.rng_seed != None:
        mech_args["rng_seed"] = args.rng_seed

    synth_ds = run_mechanism(args.mechanism_dir, args.mechanism, mech_args)
    if args.save != None:
        write_file(synth_ds, args.save, index=False)



