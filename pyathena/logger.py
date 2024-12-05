import logging

def _verbose_to_level(value):
    """Convert verbose to valid logging level
    """

    levels = ['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

    if value is True:
        level = 'INFO'
    elif value is False:
        level = 'WARNING'
    elif isinstance(value, str) and value.upper() in levels:
        level = value.upper()
    elif isinstance(value, int):
        level = value
    else:
        raise ValueError('Cannot recognize verbose {0:s}. '.format(value) + \
                         'Should be one of logging levels or True/False')

    return level


def create_logger(name: str, verbose) -> logging.Logger:
    """Function to initialize logger and set logging level.

    Parameters
    ----------
    name : str
        Name of the logger
    verbose : bool or str or int
        If True/False, set logging level to 'INFO'/'WARNING'.
        One of logging levels
        ('NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        or their numerical values (0, 10, 20, 30, 40, 50)
        (see https://docs.python.org/3/library/logging.html#logging-levels)
    """

    l = logging.getLogger(name)
    if not l.hasHandlers():
        h = logging.StreamHandler()
        f = logging.Formatter('[%(name)s-%(levelname)s] %(message)s')
        h.setFormatter(f)
        l.addHandler(h)

    l.setLevel(_verbose_to_level(verbose))

    return l
