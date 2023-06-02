# pyathena

[<img src="https://img.shields.io/badge/DOI-doinumber-blue">](https://doi.org/)
[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://changgoo.github.io/pyathena-1/intro.html)

<div id="top"></div>
<!--
*** README.md template Shamelessly taken from
*** https://raw.githubusercontent.com/othneildrew/Best-README-Template/master/README.md
-->

## About

pyathena is a set of python scripts for reading and analyzing simulation data produced by the Athena-TIGRESS code.

## Installation

Below is an example of how you can set up pyathena. It assumes that you have already installed [miniconda](https://docs.conda.io/en/latest/miniconda.html) or anaconda on your system.

1. Clone the pyathena repo
   ```sh
   git clone https://github.com/jeonggyukim/pyathena.git
   ```
3. Create an environment from the env.yml file
   ```sh
   conda update conda # if you haven't already
   conda env create -f path_to_pyathena/env.yml
   ```
4. Activate the pyathena environment
   ```sh
   conda activate pyathena
   ```
5. Add pyathena directory to your python startup file (optional but recommended).

## Example Usage

See example [notebooks](../notebook).

## Contributing

If you have a suggestion that would make pyathena better, please fork the repo and create a pull request.
Don't forget to give the project a star! Thanks again!

1. Fork pyathena
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
