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
            X = X / X_sum
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
    model, synth = aim.AIM(args["epsilon"], args["delta"]).run(data, workload)
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

def mech_real(args):
    real_data = args["dataset"]
    return real_data, -1.0

def mech_tabddpm(args):
    train = args["dataset"]
    train_x = train["label"].values
    train_y = train.drop(columns=["label"]).values
    numpy.save("x_train.npy", train_x)
    numpy.save("y_train.npy", train_y)
    return 0, 0

mechanisms = {
        "aim": mech_aim,
        "mst": mech_mst, 
        "mwem": mech_mwem, 
        "pro_gene_gen": mech_pgg,
        "real": mech_real,
        "tabddpm": mech_tabddpm,
        }

def run_mechanism(directory, mechanism, args):
    sys.path.insert(0, directory)
    if mechanism == "pro_gene_gen":
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



