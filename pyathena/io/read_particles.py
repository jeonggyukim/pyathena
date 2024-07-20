"""Module containing particle output readers"""


from .athena_read import partab,parbin
from pandas import read_csv

def read_partab(filename, **kwargs):
    """Read particle output

    Parameters
    ----------
    filename : str
        Data filename
    **kwargs : dict, optional
        Extra arguments passed to partab. Refer to partab documentation for
        a list of all possible arguments.

    Returns
    -------
    ds : pandas.DataFrame
        Particle data
    """
    ds = partab(filename, **kwargs)
    ds.set_index('pid', inplace=True)

    return ds

def read_parbin(filename, **kwargs):
    """Read particle output

    Parameters
    ----------
    filename : str
        Data filename
    **kwargs : dict, optional
        Extra arguments passed to partab. Refer to partab documentation for
        a list of all possible arguments.

    Returns
    -------
    ds : xarray.Dataset
        Particle data
    """
    ds = parbin(filename, **kwargs)

    return ds

def read_parhst(filename, **kwargs):
    """Read individual particle history

    Parameters
    ----------
    filename : str
        Data filename
    **kwargs : dict, optional
        Extra arguments passed to pandas.read_csv.

    Returns
    -------
    ds : pandas.DataFrame
        Individual particle history
    """
    ds = read_csv(filename, **kwargs)
    # TODO(SMOON) set index

    return ds
