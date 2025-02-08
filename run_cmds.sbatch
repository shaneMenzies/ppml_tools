#!/bin/bash
#SBATCH   -p small
#SBATCH   -J run_cmds
#SBATCH   -o run_cmds.o%j
#SBATCH   -e run_cmds.o%j
#SBATCH   -N 1
#SBATCH   -t 48:00:00
#SBATCH   --mail-user=smenzies@uw.edu
#SBATCH   --mail-type=all

module load python3/3.9.2
module load pylauncher
module load tacc-apptainer
python3 run_cmds.py $1
