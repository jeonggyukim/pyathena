import os
import pandas as pd

def cool_grackle(Z=1.0):
    """
    Function to read equilibrium cooling function in grackle

    Parameters
    ----------
    Z : float
        Metallicity
    """

    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, '../../data/microphysics')
    fname = os.path.join(data_dir, 'Grackle_equillibrium_cooling_{0:6.4f}Z.dat'.format(Z))
    df = pd.read_csv(fname, sep=' ')

    return df
