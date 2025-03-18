import pathlib
import os
import os.path as osp
import numpy as np
import astropy.units as au
import astropy.constants as ac

class CollIonRate(object):

    def __init__(self):
        self._read_data()

    def _read_data(self, max_rows=465):

        basedir = osp.join(pathlib.Path(__file__).parent.absolute(),
                           '../../data/microphysics/cloudy')
        self.fname = os.path.join(basedir, 'coll_ion.dat')
        lines = np.loadtxt(self.fname, unpack=True, skiprows=1, max_rows=max_rows)

        # In cloudy, both eletron and atomic numbers are on C scale (starting from 0), so
        # need to add 1.
        self.N = lines[0].astype(int) + 1
        self.Z = lines[1].astype(int) + 1
        self.dE_Kel = lines[2]*(1.0*au.eV/ac.k_B).cgs.value
        self.P = lines[3] # 0 or 1
        self.A = lines[4]
        self.X = lines[5]
        self.K = lines[6]

    def get_ci_rate(self, Z, N, T):
        iZ = Z == self.Z
        iN = N == self.N
        i = np.where(iZ & iN)[0][0]

        U = self.dE_Kel[i]/T
        rate = np.where(U > 80.0,
                        0.0,
                        self.A[i]*(1.0 + self.P[i]*U**0.5)/\
                        (self.X[i] + U)*U**(self.K[i])*np.exp(-U))

        return rate
