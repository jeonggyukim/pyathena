# Installation

Below is an example of how you can set up pyathena. It assumes that you have already installed [miniconda](https://docs.conda.io/en/latest/miniconda.html) or anaconda on your system.

1. Clone the pyathena repo
   ```sh
   git clone git@github.com:jeonggyukim/pyathena.git
   ```
3. Create an environment from the env.yml file
   ```sh
   cd pyathena/
   conda env create -f env.yml
   ```
4. Activate the pyathena environment
   ```sh
   conda activate pyathena
   ```
5. Add pyathena directory to your python startup file (optional but recommended).

# MPI setup for stellar Princeton cluster

After setting up the environment as above,
```sh
# installing mpi4py on stellar
# https://researchcomputing.princeton.edu/support/knowledge-base/mpi4py
module load openmpi/gcc/4.1.0
export MPICC=$(which mpicc)
pip install mpi4py
```