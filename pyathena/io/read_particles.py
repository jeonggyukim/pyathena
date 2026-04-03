"""Module containing particle output readers"""


from .athena_read import partab, parbin
from pandas import read_csv

def read_partab(filename, **kwargs):
    """Read Athena++ particle tab file (`par?.tab`) into a DataFrame.

    Columns are read from the file header. The DataFrame index is set to
    the particle id (``pid``) and sorted.

    Parameters
    ----------
    filename : str
        Path to the `.tab` particle output file.
    **kwargs : dict, optional
        Extra arguments passed to :func:`~pyathena.io.athena_read.partab`
        (e.g., ``raw=True`` to skip column-name parsing).

    Returns
    -------
    ds : pandas.DataFrame
        Particle data indexed by ``pid``.

    Examples
    --------
    >>> from pyathena import read_partab
    >>> df = read_partab("/path/to/par0.tab")
    >>> df.head()
    """
    ds = partab(filename, **kwargs)
    ds.set_index('pid', inplace=True)
    ds.sort_index(inplace=True)

    return ds

def read_parbin(filename, **kwargs):
    """Read Athena++ particle binary file (`par?.bin`) into a DataFrame.

    The DataFrame index is set to the particle id (``pid``) and sorted.

    Parameters
    ----------
    filename : str
        Path to the `.bin` particle output file.
    **kwargs : dict, optional
        Extra arguments passed to :func:`~pyathena.io.athena_read.parbin`.

    Returns
    -------
    ds : pandas.DataFrame
        Particle data indexed by ``pid``.
    """
    ds = parbin(filename, **kwargs)
    ds.set_index('pid', inplace=True)
    ds.sort_index(inplace=True)

    return ds

def read_parhst(filename, **kwargs):
    """Read an individual particle history file into a DataFrame.

    The file is expected to be a CSV-style text file produced by Athena++
    for tracking individual particle trajectories over time.

    Parameters
    ----------
    filename : str
        Path to the particle history file.
    **kwargs : dict, optional
        Extra arguments passed to :func:`pandas.read_csv`
        (e.g., ``sep``, ``header``, ``names``).

    Returns
    -------
    ds : pandas.DataFrame
        Individual particle history.
    """
    ds = read_csv(filename, **kwargs)
    # TODO(SMOON) set index

    return ds
