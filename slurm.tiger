#!/bin/bash
#SBATCH --job-name=do_tasks
#SBATCH -N 1
#SBATCH -p shared
#SBATCH -n 20
#SBATCH --ntasks-per-node=28
#SBATCH -t 00:30:00
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=user@email
#SBATCH --error=do_tasks_%j.err
#SBATCH --output=do_tasks_%j.out
#

NPROCS=20

MODULE="pyathena.tigress_ncr.do_tasks"

export MATPLOTLIBRC="$HOME/.config/matplotlib"
export TERM="xterm-256color"
export PATH="$HOME/miniconda3/bin:$HOME/local/bin:$PATH"
export PYTHONSTARTUP="$HOME/.pythonrc.py"
export PYTHONPATH="$PYTHONPATH:/tigress/jk11/slug2/"

module load intel-mpi

echo "Starting:"

srun -n $NPROCS --mpi=pmi2 python -m $MODULE

date

echo "do_tasks finished"
