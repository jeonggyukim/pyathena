"""Read Draine (2003)'s dust extinction cross section
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pathlib

class DustDraine(object):
    """Class to read dust extinction, scattering, and absorption
    properties computed for carbonaceous-silicate model for
    interstellar dust with R_V=3.1 (Draine 2003)

    Tabulated quantities in DataFrame:
    lwav  = wavelength in vacuo (micron)
    lwavAA = wavelength in vacuo (Anstrom)
    albedo  = (scattering cross section)/(extinction cross section)
    cos   = <cos(theta)> for scattered light
    cossq = <cos^(theta)> for scattered light
    Cext = extinction cross section per H nucleon (cm^2/H)
    K_abs = absorption cross section per mass of dust (cm^2/gram)
    kappa_abs = absorption cross section per mass of gas (cm^2/gram)

    https://www.astro.princeton.edu/~draine/dust/dustmix.html

    Example
    -------
    dfa (dictionary) has all data.

    >>> d = DustDraine()
    >>> print(d.dfa.keys())

    Print gas-to-dust ratio
    >>> d = DustDraine()
    >>> print(d.GTD)
    """

    def __init__(self, basedir='/tigress/jk11/pyathena'):
        # read data

        local = pathlib.Path(__file__).parent.absolute()

        # Milky Way exctinction
        self.fname_Rv31 = os.path.join(local,'../../data/kext_albedo_WD_MW_3.1A_60_D03.dat')
        self.fname_Rv40 = os.path.join(local, '../../data/kext_albedo_WD_MW_4.0A_40_D03.dat')
        self.fname_Rv55 = os.path.join(local, '../../data/kext_albedo_WD_MW_5.5A_30_D03.dat')
        self.fname_LMCavg = os.path.join(local, '../../data/kext_albedo_WD_LMCavg_20.dat')
        self.fname_SMCbar = os.path.join(local, '../../data/kext_albedo_WD_SMCbar_0.dat')

        # Gas to dust ratio
        self.GTD = {'Rv31':1.236e2,
                    'Rv40':1.174e2,
                    'Rv55':1.051e2,
                    'LMCavg':4.281E+02,
                    'SMCbar':6.669E+02,
                    }

        self.Mdust_per_H = {'Rv31':1.870E-26,
                            'Rv40':1.969E-26,
                            'Rv55':2.199E-26,
                            'LMCavg':5.462E-27,
                            'SMCbar':3.506E-27,
                            }

        self.dfa = dict()
        self.dfa['Rv31'] = self._read_data(self.fname_Rv31, self.GTD['Rv31'])
        self.dfa['Rv40'] = self._read_data(self.fname_Rv40, self.GTD['Rv40'])
        self.dfa['Rv55'] = self._read_data(self.fname_Rv55, self.GTD['Rv55'])
        self.dfa['LMCavg'] = self._read_data(self.fname_LMCavg, self.GTD['LMCavg'], MW=False)
        self.dfa['SMCbar'] = self._read_data(self.fname_SMCbar, self.GTD['SMCbar'], MW=False)

    @staticmethod
    def _read_data(fname, GTD, MW=True):

        # Read dielectronic recombination rate data
        with open(fname, 'r') as fp:
            lines = fp.readlines()

        nline = len(lines)

        lwav = np.zeros(nline,)
        albedo = np.zeros(nline,)
        cos = np.zeros(nline,)
        Cext = np.zeros(nline,)
        kappa_abs = np.zeros(nline,)
        if MW:
            cossq = np.zeros(nline,)
            idx_comment = 6
        else:
            idx_comment = 5

        # Get maximum string length of comments
        maxlen = 0
        for i, line in enumerate(lines):
            l = line.split()
            if len(l[idx_comment:]) != 0:
                maxlen = max(maxlen, len(' '.join(l[idx_comment:])))
        comment = np.chararray(nline, itemsize=maxlen)

        for i, line in enumerate(lines):
            l = line.split()
            lwav[i] = float(l[0])
            albedo[i] = float(l[1])
            cos[i] = float(l[2])
            Cext[i] = float(l[3])
            kappa_abs[i] = float(l[4])
            if MW:
                cossq[i] = float(l[5])
            if len(l[idx_comment:]) != 0:
                comment[i] = ' '.join(l[idx_comment:])
            else:
                comment[i] = ''

        if MW:
            df = pd.DataFrame({'lwav':lwav, 'albedo':albedo, 'cos':cos,
                               'Cext':Cext, 'K_abs':kappa_abs, 'cossq':cossq,
                               'comment':comment})
        else:
            df = pd.DataFrame({'lwav':lwav, 'albedo':albedo, 'cos':cos,
                               'Cext':Cext, 'K_abs':kappa_abs,
                               'comment':comment})

        df['Cabs'] = df['Cext']*(1 - df['albedo'])

        df['lAA'] = df['lwav']*1e4
        df['kappa_abs'] = df['K_abs']/GTD

        return df
