"""Module containing particle output readers"""


from .athena_read import partab, parbin
import numpy as np
from pandas import DataFrame, read_csv

def read_partab(filename, *, header_only=False, **kwargs):
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
    with open(filename) as stream:
        first = stream.readline()
    metadata = {}
    if 'time =' in first:
        metadata['time'] = float(first.split('time =', 1)[1].strip())
    if header_only:
        return metadata
    ds = partab(filename, **kwargs)
    ds.attrs.update(metadata)
    ds.set_index('pid', inplace=True)
    ds.sort_index(inplace=True)

    return ds

def read_parbin(filename, **kwargs):
    """Read an Athena++ or native AthenaK particle snapshot into a DataFrame.

    The DataFrame index is set to the particle id (``pid``) and sorted.

    Parameters
    ----------
    filename : str
        Path to an Athena++ `.parbin` or native AthenaK `.parbin` snapshot.
    **kwargs : dict, optional
        Extra arguments passed to :func:`~pyathena.io.athena_read.parbin`.

    Returns
    -------
    ds : pandas.DataFrame
        Particle data indexed by ``pid``.
    """
    with open(filename, 'rb') as stream:
        signature = stream.readline()
        if signature.startswith(b'Athena particle output version='):
            if signature != b'Athena particle output version=1.0\n':
                raise ValueError('Unsupported particle output version')
            header = {}
            for line in stream:
                if line == b'data:\n':
                    break
                key, separator, value = line.decode('ascii').strip().partition('=')
                if not separator:
                    raise ValueError('Invalid particle output header')
                header[key] = value
            else:
                raise ValueError('Missing particle data marker')
            try:
                metadata = dict(time=float(header['time']), cycle=int(header['cycle']),
                                output_number=int(header['output number']),
                                nparticles=int(header['number of particles']),
                                real_size=int(header['size of real']),
                                byte_order=header['byte order'])
            except (KeyError, ValueError) as error:
                raise ValueError('Invalid particle output header') from error
            count = metadata['nparticles']
            size = metadata['real_size']
            order = metadata['byte_order']
            if count < 0 or size not in (4, 8) or order not in ('little', 'big'):
                raise ValueError('Invalid particle count, precision, or byte order')
            if kwargs.get('header_only', False):
                return metadata
            payload = stream.read()
            if len(payload) != count*(4 + 7*size):
                raise ValueError('Invalid particle payload length')
            endian = '<' if order == 'little' else '>'
            ids = np.frombuffer(payload, dtype=endian+'i4', count=count).astype(np.int32)
            state = np.frombuffer(payload, dtype=endian+f'f{size}',
                                  offset=4*count).reshape(count, 7)
            ds = DataFrame(state, columns=['mass', 'x1', 'x2', 'x3', 'v1', 'v2', 'v3'],
                           index=ids)
            ds.index.name = 'pid'
            ds.sort_index(inplace=True)
            ds.attrs.update(metadata)
            return ds
    ds = parbin(filename, **kwargs)
    if kwargs.get('header_only', False):
        return ds
    ds.set_index('pid', inplace=True)
    ds.sort_index(inplace=True)

    return ds

def read_parhst(filename, **kwargs):
    """Read an individual particle history file into a DataFrame.

    The file is expected to be a CSV-style text file produced by Athena++ or AthenaK
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
