import numpy
import pandas
import joblib
import json

file_readers = {
        "csv": lambda file, **kwargs: pandas.read_csv(file, **kwargs),
        "tsv": lambda file, **kwargs: pandas.read_csv(file, sep='\t', **kwargs),
        "json": lambda file, **kwargs: json.load(open(file, **kwargs)),
        "npy": lambda file, **kwargs: numpy.load(file, **kwargs),
        "txt": lambda file, **kwargs: numpy.loadtxt(file, **kwargs)
        }

file_writers = {
        "csv": lambda data, file, **kwargs: data.to_csv(file, **kwargs),
        "tsv": lambda data, file, **kwargs: data.to_csv(file, **kwargs, sep='\t'),
        "joblib": lambda data, file, **kwargs: joblib.dump(data, file, **kwargs),
        "json": lambda data, file, **kwargs: json.dump(data, open(file, 'w'), **kwargs, indent=4),
        "npy": lambda data, file, **kwargs: numpy.save(file, data, **kwargs),
        "txt": lambda data, file, **kwargs: numpy.savetxt(file, data, **kwargs)
        }

def read_file(path, **kwargs):
    _, _, suffix = path.rpartition('.')
    return file_readers[suffix](path, **kwargs)

def write_file(data, path, **kwargs):
    _, _, suffix = path.rpartition('.')
    file_writers[suffix](data, path, **kwargs)
