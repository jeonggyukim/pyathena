import logging

def create_logger(name: str, verbose) -> logging.Logger:
    """Function to initialize logger and set default verbosity.

    Parameters
    ----------
    name : str
        Name of the logger
    verbose : bool or str or int
        Set logging level to "INFO"/"WARNING" if True/False.
    """

    levels = ['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

    if verbose is True:
        level = 'INFO'
    elif verbose is False:
        level = 'WARNING'
    elif verbose in levels + [l.lower() for l in levels]:
        level = verbose.upper()
    elif isinstance(verbose, int):
        level = verbose
    else:
        raise ValueError('Cannot recognize option {0:s}.'.format(verbose))

    l = logging.getLogger(name)
    if not l.hasHandlers():
        h = logging.StreamHandler()
        f = logging.Formatter('[%(name)s-%(levelname)s] %(message)s')
        h.setFormatter(f)
        l.addHandler(h)

    l.setLevel(level)

    return l
