import argparse
import pylauncher
parser = argparse.ArgumentParser()
parser.add_argument("target")
args = parser.parse_args()

pylauncher.ClassicLauncher(args.target, debug="host+job", delay=60.0, queuestate=(args.target + ".queuestate"), workdir=(args.target + ".jobs.d"))
