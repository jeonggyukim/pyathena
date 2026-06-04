## About

pyathena is a set of python scripts for reading and analyzing simulation data produced by the Athena-TIGRESS and TIGRIS codes.

## Requirement

Python version **3.10** or higher

## Installation

Below is an example of how you can set up pyathena. It assumes that you have already installed [miniforge](https://github.com/conda-forge/miniforge) (recommended), [miniconda](https://docs.conda.io/en/latest/miniconda.html), or anaconda on your system. Miniforge defaults to the `conda-forge` channel and avoids Anaconda Inc.'s commercial-use licensing on the `defaults` channel. Miniforge also ships [`mamba`](https://github.com/mamba-org/mamba), a faster drop-in replacement for `conda` — swap `conda` for `mamba` in any command below.

```sh
git clone https://github.com/jeonggyukim/pyathena.git
cd pyathena
conda env create -f env.yml
conda activate pyathena
pip install -e .                                         # drop the -e for a non-editable install
python -c "import pyathena; print(pyathena.__file__)"    # sanity check
```

Use `pip install -e .` (editable) if you plan to edit the source — changes take effect without reinstalling.

If conda cannot install the environment because two packages need different versions of the same thing, install an older version of one of them, for example:
```sh
conda install -c conda-forge <package-name>=<version>
```

To update the existing pyathena environment with an updated env.yml file
```sh
conda activate pyathena
conda env update --file env.yml --prune
```

To remove pyathena environment
```sh
conda remove --name pyathena --all
```

## Example Usage

See example [notebooks](notebook) and [documentation](https://jeonggyukim.github.io/pyathena/intro.html).

## Contributing

Fork the repo, follow the [Installation](#installation) steps with `pip install -e .`, then submit a pull request from a feature branch.

## License

See [LICENSE.md](LICENSE.md).
