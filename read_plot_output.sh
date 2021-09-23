#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p konings,normal,owners
#SBATCH -t 2:00:00
#SBATCH --begin=now+3hours
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cfamigli@stanford.edu
#SBATCH --array=0-1

# define the location of the command list file
CMD_LIST=./read_plot_output.txt

# get the command list index from Slurm
CMD_INDEX=$SLURM_ARRAY_TASK_ID

# execute the command
$(sed "${CMD_INDEX}q;d" "$CMD_LIST")