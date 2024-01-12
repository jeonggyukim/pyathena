import numpy as np
import os
import os.path as osp
import pathlib

import astropy.units as au
import astropy.constants as ac

__all__ = ['PhotX']

class PhotX(object):
    """
    Computes photoionization cross-sections as described in Verner 96
    http://adsabs.harvard.edu/abs/1996ApJ...465..487V
    Original python code from Rabacus implementation
    https://github.com/galtay/rabacus/tree/master/rabacus/atomic/verner/photox
    (Released under GNU GPL3 v.30;
    JKIM: is it okay to copy snippets of their code? what to do with license?)
    """

    def __init__(self, datadir=None):
        # Read Verner et al. 1996 photoionization cross-section table
        if datadir is None:
            basedir = osp.join(pathlib.Path(__file__).parent.absolute(),
                               '../../data/microphysics')
            fname = os.path.join(basedir, 'verner96_photx.dat')

        dat = np.loadtxt(fname, unpack=True)
        self._dat = dat

        # Organize data
        self.Z = self._dat[0]
        self.N = self._dat[1]
        self.Eth = self._dat[2]
        self.Emax = self._dat[3]
        self.E0 = self._dat[4]
        self.sigma0 = self._dat[5] * 1.0e-18
        self.ya = self._dat[6]
        self.P = self._dat[7]
        self.yw = self._dat[8]
        self.y0 = self._dat[9]
        self.y1 = self._dat[10]
        del self._dat

    def get_Eth(self, Z, N, unit='eV'):
        """
        Threshold ionization energy in eV for ions defined by Z and N.

        Parameters
        ----------
        Z : int
            Atomic number (number of protons)
        N : int
            Electron number

        Returns
        -------
        Eth: float
            Threshold ionization energy in eV
        """

        c1 = self.Z == Z
        c2 = self.N == N
        indx = np.where(c1 & c2)
        indx = indx[0][0]
        Eth = self.Eth[indx]

        if unit == 'eV':
            return Eth
        elif unit == 'Angstrom':
            return ((ac.h*ac.c)/(Eth*au.eV)).to('Angstrom').value

    def get_sigma(self, Z, N, E):
        """Returns a photo-ionization cross-section for an ion defined by
        Z and N at energies E in eV.

        Parameters
        ----------
        Z : int
            Atomic number (number of protons)
        N : int
            Electron number (number of electrons)
        E : array of floats
            Calculate cross-section at these energies [eV]

        Returns
        -------
        sigma: array of floats
            Photoionization cross-sections [cm^-2]
        """

        # calculate fit
        c1 = self.Z == Z
        c2 = self.N == N
        indx = np.where(c1 & c2)
        indx = indx[0][0]
        Z = self.Z[indx]
        N = self.N[indx]
        Eth = self.Eth[indx]
        Emax = self.Emax[indx]
        E0 = self.E0[indx]
        sigma0 = self.sigma0[indx]
        ya = self.ya[indx]
        P = self.P[indx]
        yw = self.yw[indx]
        y0 = self.y0[indx]
        y1 = self.y1[indx]

        x = E / E0 - y0
        y = np.sqrt(x*x + y1*y1)

        sigma = sigma0 * ((x-1)*(x-1) + yw*yw) * y**(0.5*P - 5.5) * \
            (1 + np.sqrt(y/ya))**(-P)

        # zero cross-section below threshold
        indx = np.where(E < Eth)
        if indx[0].size > 0:
            sigma[indx] = 0.0

        return sigma

def get_sigma_pi_H2(E):
    """H2 photoionization cross-section [cm^-2]
    Table 1 in Baczynski et al. (2015)
    Piecewise constant cross-section to the analytical results
    of Liu & Shemansky (2012)

    E : array of floats
        Photon energy in eV
    """
    return 1e-18*np.piecewise(E, [E < 15.2,
                            ((E >= 15.2) & (E < 15.45)),
                            ((E >= 15.45) & (E < 15.70)),
                            ((E >= 15.7) & (E < 15.95)),
                            ((E >= 15.95) & (E < 16.20)),
                            ((E >= 16.2) & (E < 16.40)),
                            ((E >= 16.4) & (E < 16.65)),
                            ((E >= 16.65) & (E < 16.85)),
                            ((E >= 16.85) & (E < 17.0)),
                            ((E >= 17.0) & (E < 17.2)),
                            ((E >= 17.2) & (E < 17.65)),
                            ((E >= 17.65) & (E < 18.1)),
                            E >= 18.1],
                            [0.0,0.09,1.15,3.0,5.0,6.75,8.0,9.0,9.5,9.8,10.1,9.85,
                             lambda E: 9.85/(E/18.1)**3])
