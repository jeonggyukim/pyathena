"""
Read athena++ partab file
"""


from .athena_read import partab

def read_partab(filename, **kwargs):
    ds = partab(filename, **kwargs)
    ds.set_index('pid', inplace=True)
    
    return ds
