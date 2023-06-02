"""
Read athena++ particle outputs
"""


from .athena_read import partab
from pandas import read_csv

def read_partab(filename, **kwargs):
    ds = partab(filename, **kwargs)
    ds.set_index('pid', inplace=True)

    return ds

def read_parhst(filename, **kwargs):
    ds = read_csv(filename, **kwargs)
    # TODO(SMOON) set index

    return ds
