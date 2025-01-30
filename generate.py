import argparse
import itertools
import importlib
import numpy
import pandas
import json
import sys
import os
from secrets import randbits

file_readers = {
        "csv": lambda file: pandas.read_csv(file),
        "tsv": lambda file: pandas.read_csv(file, sep='\t'),
        "xlsx": lambda file: pandas.read_excel(file),
        "json": lambda file: json.load(open(file)) 
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
        "delta": 1e-9,
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

def calculate_error(realDS, synthDS, workload):
    errors = []
    for proj in workload:
        x = realDS.project(proj).datavector()
        y = synthDS.project(proj).datavector()
        e = 0.5 * numpy.linalg.norm(y / y.sum() - y / y.sum(), 1)
        errors.append(e)
    return numpy.mean(numpy.asarray(errors))

def make_dataset(data, domain):
    from mbi import Dataset, Domain
    # Should already be loaded from a file
    return Dataset(data, Domain(domain.keys(), domain.values()))

def mech_aim(args): 
    from mbi import Dataset, Domain
    aim = importlib.import_module("aim")
    data = make_dataset(args["dataset"], args["domain"])
    workload = determine_workload(args, data)
    model, synth = aim.AIM(args["epsilon"], args["delta"]).run(data, workload)
    workload = [(cl, 1.0) for cl in workload]
    return (synth.df, calculate_error(data, synth, workload))

def mech_mst(args):
    from mbi import Dataset, Domain
    mst = importlib.import_module("mst")
    data = make_dataset(args["dataset"], args["domain"])
    workload = determine_workload(args, data)
    synth = mst.MST(data, args["epsilon"], args["delta"]) 
    return (synth.df, calculate_error(data, synth, workload))

def mech_mwem(args):
    from mbi import Dataset, Domain
    mwem = importlib.import_module("mwem+pgm")
    data = make_dataset(args["dataset"], args["domain"])
    workload = determine_workload(args, data)
    synth = mwem.mwem_pgm(data, args["epsilon"], args["delta"],
                              workload=workload)
    return (synth.df, calculate_error(data, synth, workload))

def mech_pgg(args):
    pgg = importlib.import_module("model")
    pgg_model = pgg.Private_PGM(args["label_col"], True, 
                            args["epsilon"], args["delta"])
    pgg_model.train(args["dataset"], args["domain"], num_iters=2500)
    synth = pgg_model.generate(num_rows=args["dataset"].shape[0])
    # pgg always puts label column at the end
    synth[:, 0], synth[:, -1] = synth[:, -1].copy(), synth[:, 0].copy()
    synth = pandas.DataFrame(synth, columns=args["dataset"].columns)
    realDS = make_dataset(args["dataset"], args["domain"])
    synthDS = make_dataset(synth, args["domain"])
    error = calculate_error(realDS, synthDS, determine_workload(args, realDS))
    return (synth, error)

mechanisms = {
        "aim": mech_aim,
        "mst": mech_mst, 
        "mwem": mech_mwem, 
        "pro_gene_gen": mech_pgg
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

    synth_ds, error = run_mechanism(args.mechanism_dir, args.mechanism, mech_args)
    print("Error:", error)
    if args.save != None:
        write_file(synth_ds, args.save, index=False)



