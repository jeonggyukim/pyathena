#!/bin/bash
# Princeton HPC specific setups

# clone repository

git clone https://github.com/jeonggyukim/pyathena

# load anaconda

module load anaconda3/2022.5

# create an environment

conda env create -f env.yml

conda activate pyathena

# installing mpi4py on stellar
# https://researchcomputing.princeton.edu/support/knowledge-base/mpi4py

module load openmpi/gcc/4.1.6
export MPICC=$(which mpicc)
pip install mpi4py

# installing ffmpeg
conda install ffmpeg
