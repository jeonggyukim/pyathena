# Installation

Below is an example of how you can set up pyathena. It assumes that you have already installed [miniconda](https://docs.conda.io/en/latest/miniconda.html) or anaconda on your system.

1. Clone the pyathena repo
   ```sh
   git clone https://github.com/jeonggyukim/pyathena.git
   ```
3. Create an environment from the env.yml file
   ```sh
   conda update conda
   conda env create -f path_to_pyathena/env.yml
   ```
4. Activate the pyathena environment
   ```sh
   conda activate pyathena
   ```
5. Install pyathena
   ```sh
   pip install .
   ```

Sometimes `yt` and other installed packages (e.g., numpy) may have compatibility issues. In this case, you can downgrade packages to more stable, older versions. For example,
```sh
conda install -c conda-forge numpy=1.26.4
```

To update the existing pyathena environment with an updated env.yml file
```sh
conda activate pyathena
conda env update --file env.yml --prune
```

# MPI setup for stellar Princeton cluster

After setting up the environment as above,
```sh
# installing mpi4py on stellar
# https://researchcomputing.princeton.edu/support/knowledge-base/mpi4py
module load openmpi/gcc/4.1.0
export MPICC=$(which mpicc)
pip install mpi4py
```
