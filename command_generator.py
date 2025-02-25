import argparse
import os

command_values = [
        "prefix",
        "pipeline_loc",
        "spec_file",
        "spec_id",
        "mech_dir",
        "mechanism",
        "epsilon",
        "iteration",
        "output_dir"
        ]

# puts together a single command
def single_command(values):
    return f"{values["prefix"]} {values["pipeline_loc"]} -pps {values["spec_file"]} --mechanism-dir {values["mech_dir"]} -m {values["mechanism"]} -e {values["epsilon"]} -o {values["output_dir"]} {values["suffix"]}\n"

pattern_operators = {
        "%spec_file": lambda values, output: output.replace("%spec_file", values["spec_file"]),
        "%spec_id": lambda values, output: output.replace("%spec_id", values["spec_id"]),
        "%mechanism": lambda values, output: output.replace("%mechanism", values["mechanism"]),  
        "%epsilon": lambda values, output: output.replace("%epsilon", values["epsilon"]),  
        "%iteration": lambda values, output: output.replace("%iteration", str(values["iteration"])),  
        }

def parse_pattern(values, pattern):
    output = pattern
    remaining = True
    while remaining:
        remaining = False
        for pattern in pattern_operators.keys():
            if pattern in output:
                output = pattern_operators[pattern](values, output)
                remaining = True
    return output

def collate_command(values):
    collate_loc = os.path.join(values["pipeline_loc"], "../collate.py")
    collation_dir = os.path.join(values["output_dir"], "..")
    title = parse_pattern(values, "%spec_id - %mechanism - %epsilon")
    iterations = ""
    for i in range(values["iteration"] + 1):
        iterations += f"./{i}/" 
    return f"cd {collation_dir} && {values["prefix"]} {collate_loc} {iterations} -t {title} \n"

command_generator_args = [
        "prefix",
        "pipeline_loc",
        "spec_files",
        "spec_ids",
        "mech_dir",
        "mechanisms",
        "epsilons",
        "iterations",
        "output_dir_pattern",
        "command_file_pattern",
        "suffix",
        "auto_collate"
        ]

def generate_commands(args):
    opened_files = {}

    for (spec_file, spec_id) in zip(args["spec_files"], args["spec_ids"]):
        for mechanism in args["mechanisms"]:
            for epsilon in args["epsilons"]:
                values = {
                        "prefix": args["prefix"],
                        "pipeline_loc": args["pipeline_loc"],
                        "spec_file": spec_file,
                        "spec_id": spec_id,
                        "mech_dir": args["mech_dir"],
                        "mechanism": mechanism,
                        "epsilon": str(epsilon),
                        "suffix": args["suffix"]
                        }
                for iteration in range(args["iterations"]):
                    values["iteration"] = iteration
                    values["output_dir"] = parse_pattern(values, args["output_dir_pattern"])
                    target_file = parse_pattern(values, args["command_file_pattern"])
                    if target_file not in opened_files:
                        opened_files[target_file] = open(target_file, "w")
                    opened_files[target_file].write(single_command(values))
                if args["auto_collate"]:
                    target_file = parse_pattern(values, args["command_file_pattern"])
                    opened_files[target_file].write(collate_command(values))

    for file in opened_files.values():
        file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "PyLauncher command generator")
    parser.add_argument("--prefix", default="python")
    parser.add_argument("--pipeline-loc", required=True)
    parser.add_argument("--spec-files", required=True, action="extend", nargs="+", type=str)
    parser.add_argument("--spec-ids", action="extend", nargs="+", type=str)
    parser.add_argument("--mech-dir", required=True)
    parser.add_argument("--mechanisms", required=True, action="extend", nargs="+", type=str)
    parser.add_argument("--epsilons", required=True, action="extend", nargs="+", type=str)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--out-dir-format", required=True)
    parser.add_argument("--cmd-file-format", required=True)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--auto-collate", action="store_true")
    args = vars(parser.parse_args())
    args["output_dir_pattern"] = args["out_dir_format"]
    args["command_file_pattern"] = args["cmd_file_format"]

    generate_commands(args)

    

    
