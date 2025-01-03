<div id="top"></div>

## About

pyathena is a set of python scripts for reading and analyzing simulation data produced by the Athena-TIGRESS and TIGRIS codes.

## Requirement

Python version **3.10** or higher

## Installation

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

To remove pyathena environment
```sh
conda remove --name pyathena --all
```

## Example Usage

See example [notebooks](notebook) and [documentation](https://jeonggyukim.github.io/pyathena/intro.html).

## Contributing

If you have a suggestion that would make pyathena better, please fork the repo and create a pull request.
Don't forget to give the project a star! Thanks again!

1. Fork pyathena
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
