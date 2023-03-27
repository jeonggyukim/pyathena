"""
Read athena++ parhst file
"""


from pandas import read_csv

def read_parhst(filename, **kwargs):
    ds = read_csv(filename, **kwargs)
    # TODO(SMOON) set index
    
    return ds
