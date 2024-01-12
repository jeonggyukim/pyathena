"""Computes recombination rate
"""

import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import pathlib

class RecRate(object):
    """Class to compute Badnell (radiative/dielectronic) recombination rates,
    Draine (2011)'s recombination rates
    """

    def __init__(self):
        # read data
        self._read_data()

    def _read_data(self):

        basedir = osp.join(pathlib.Path(__file__).parent.absolute(),
                           '../../data/microphysics')

        self.fname_dr_C = os.path.join(basedir, 'badnell_dr_C.dat')
        self.fname_dr_E = os.path.join(basedir, 'badnell_dr_E.dat')
        self.fname_rr = os.path.join(basedir, 'badnell_rr.dat')

        # Read dielectronic recombination rate data
        with open(self.fname_dr_C, 'r') as fp:
            lines1 = fp.readlines()

        with open(self.fname_dr_E, 'r') as fp:
            lines2 = fp.readlines()

        i0 = 4
        nline = len(lines1) - i0
        if len(lines1) != len(lines2):
            print('Check data file (lines1, lines2) = {0:d}, {1:d}',
                  len(lines1), len(lines2))
            raise

        self.Zd = np.zeros(nline, dtype='uint8')
        self.Nd = np.zeros(nline, dtype='uint8')
        self.Md = np.zeros(nline, dtype='uint8')
        self.Wd = np.zeros(nline, dtype='uint8')
        self.Cd = np.zeros((nline, 9))
        self.Ed = np.zeros((nline, 9))
        self.nd = np.zeros(nline, dtype='uint8')

        for i, (l1, l2) in enumerate(zip(lines1[i0:i0 + nline],
                                         lines2[i0:i0 + nline])):
            l1 = l1.split()
            l2 = l2.split()
            # Make sure that Z, N, M, W all match
            if int(l1[0]) == int(l2[0]) and \
               int(l1[1]) == int(l2[1]) and \
               int(l1[2]) == int(l2[2]) and \
               int(l1[3]) == int(l2[3]):
                self.Zd[i] = int(l1[0])
                self.Nd[i] = int(l1[1])
                self.Md[i] = int(l1[2])
                self.Wd[i] = int(l1[3])
                for j, l1_ in enumerate(l1[4:]):
                    self.Cd[i, j] = float(l1_)
                for j, l2_ in enumerate(l2[4:]):
                    self.Ed[i, j] = float(l2_)
                self.nd[i] = j + 1
            else:
                print("Columns do not match!")
                raise

        del lines1, lines2

        # Read radiative recombination rate data
        with open(self.fname_rr, 'r') as fp:
            lines = fp.readlines()

        i0 = 4
        nline = len(lines) - i0

        self.Zr = np.zeros(nline, dtype='uint8')
        self.Nr = np.zeros(nline, dtype='uint8')
        self.Mr = np.zeros(nline, dtype='uint8')
        self.Wr = np.zeros(nline, dtype='uint8')
        self.Ar = np.zeros(nline)
        self.Br = np.zeros(nline)
        self.T0r = np.zeros(nline)
        self.T1r = np.zeros(nline)
        self.Cr = np.zeros(nline)
        self.T2r = np.zeros(nline)
        # Use modifed B for low-charge ions
        self.modr = np.zeros(nline, dtype='bool')

        for i, l1 in enumerate(lines[i0:i0 + nline]):
            l1 = l1.split()
            self.Zr[i] = int(l1[0])
            self.Nr[i] = int(l1[1])
            self.Mr[i] = int(l1[2])
            self.Wr[i] = int(l1[3])
            self.Ar[i] = float(l1[4])
            self.Br[i] = float(l1[5])
            self.T0r[i] = float(l1[6])
            self.T1r[i] = float(l1[7])
            try:
                self.Cr[i] = float(l1[8])
                self.T2r[i] = float(l1[9])
                self.modr[i] = True
            except IndexError:
                self.modr[i] = False

    def get_rr_rate(self, Z, N, T, M=1):
        """
        Calculate radiative recombination rate coefficient

        Parameters
        ----------
        Z : int
            Nuclear Charge
        N : int
            Number of electrons of the initial target ion
        T : array of floats
            Temperature [K]
        M : int
            Initial metastable levels (M=1 for the ground state) of the ground
            and metastable terms. The defualt value is 1.

        Returns
        -------
        rr: array of floats
            Radiative recombination coefficients [cm^3 s^-1]
        """

        c1 = self.Zr == Z
        c2 = self.Nr == N
        c3 = self.Mr == M
        idx = np.where(c1 & c2 & c3)
        i = idx[0][0]
        sqrtTT0 = np.sqrt(T/self.T0r[i])
        sqrtTT1 = np.sqrt(T/self.T1r[i])
        if self.modr[i]:
            B = self.Br[i] + self.Cr[i]*np.exp(-self.T2r[i]/T)
        else:
            B = self.Br[i]

        rr = self.Ar[i] / (sqrtTT0 * (1.0 + sqrtTT0)**(1.0 - B) * \
            (1.0 + sqrtTT1)**(1.0 + B))

        return rr

    def get_dr_rate(self, Z, N, T, M=1):
        """
        Calculate dielectronic recombination rate coefficient

        Parameters
        ----------
        Z : int
            Nuclear Charge
        N : int
            Number of electrons of the initial target ion (before recombination)
        T : array of floats
            Temperature [K]
        M : int
            Initial metastable levels (M=1 for the ground state) of the ground
            and metastable terms. The defualt value is 1.

        Returns
        -------
        rr: array of floats
            Dielectronic recombination coefficients [cm^3 s^-1]
        """

        c1 = self.Zd == Z
        c2 = self.Nd == N
        c3 = self.Md == M

        idx = np.where(c1 & c2 & c3)

        i = idx[0][0]
        dr = 0.0
        for m in range(self.nd[i]):
            dr += self.Cd[i, m]*np.exp(-self.Ed[i, m]/T)

        dr *= T**(-1.5)
        return dr

    def get_rec_rate(self, Z, N, T, M=1, kind='badnell'):
        """
        Calculate radiative + dielectronic recombination rate coefficient

        Parameters
        ----------
        Z : int
            Nuclear Charge
        N : int
            Number of electrons of the initial target ion (before recombination)
        T : array of floats
            Temperature [K]
        M : int
            Initial metastable levels (M=1 for the ground state) of the ground
            and metastable terms. The defualt value is 1.
        kind : str
            Set to 'badnell' to use fits Badnell fits or 'dr11' to use
            Draine (2011)'s formula.

        Returns
        -------
        rrate: array of floats
            Recombination rate coefficient [cm^3 s^-1]
        """

        if kind == 'badnell':
            if Z == 1: # No dielectronic recombination
                return self.get_rr_rate(Z, N, T, M=M)
            else:
                return self.get_rr_rate(Z, N, T, M=M) + \
                  self.get_dr_rate(Z, N, T, M=M)
        elif kind == 'dr11':
            if Z == 1:
                return self.get_rec_rate_H_caseA(T)
            else:
                print('Z > 1 is not supported for dr11 recombination rate.')
                raise

    @staticmethod
    def get_rec_rate_H_caseA_Dr11(T):
        """Compute case A recombination rate coefficient for H
        Table 14.1 in Draine (2011)
        """
        T4 = T*1e-4
        return 4.13e-13*T4**(-0.7131 - 0.0115*np.log(T4))

    @staticmethod
    def get_rec_rate_H_caseB_Dr11(T):
        """Compute case B recombination rate coefficient for H
        Table 14.1 in Draine (2011)
        """
        T4 = T*1e-4
        return 2.54e-13*T4**(-0.8163 - 0.0208*np.log(T4))

    @staticmethod
    def get_rec_rate_H_caseB(T):
        """Compute case B recombination rate coefficient for H
        This is what we use in Athena-TIGRESS (fit to Ferland)
        """
        Tinv = 1.0/T
        bb = 315614.0*Tinv
        cc = 115188.0*Tinv
        dd = 1.0 + np.power(cc, 0.407)
        return 2.753e-14*np.power(bb, 1.5)*np.power(dd, -2.242)

    @staticmethod
    def get_alpha_gr(T, psi, Z):

        # Parameters for Fit (14.37) to Grain Recombination Rate coefficients
        # alpha_gr(X +) for selected ions. (Draine 2011)
        C = dict()
        C['H'] = np.array([12.25, 8.074e-6, 1.378, 5.087e2, 1.586e-2, 0.4723, 1.102e-5])
        C['He']= np.array([5.572, 3.185e-7, 1.512, 5.115e3, 3.903e-7, 0.4956, 5.494e-7])
        C['C'] = np.array([45.58, 6.089e-3, 1.128, 4.331e2, 4.845e-2, 0.8120, 1.333e-4])
        C['Mg']= np.array([2.510, 8.116e-8, 1.864, 6.170e4, 2.169e-6, 0.9605, 7.232e-5])
        C['S'] = np.array([3.064, 7.769e-5, 1.319, 1.087e2, 3.475e-1, 0.4790, 4.689e-2])
        C['Ca']= np.array([1.636, 8.208e-9, 2.289, 1.254e5, 1.349e-9, 1.1506, 7.204e-4])

        if Z == 1:
            e = 'H'
        elif Z == 2:
            e = 'He'
        elif Z == 6:
            e = 'C'
        elif Z == 12:
            e = 'Mg'
        elif Z == 16:
            e = 'S'
        elif Z == 20:
            e = 'Ca'

        return 1e-14*C[e][0]/(1.0 + C[e][1]*psi**C[e][2]*\
                (1.0 + C[e][3]*T**C[e][4]*psi**(-C[e][5]-C[e][6]*np.log(T))))

    @staticmethod
    def get_rec_rate_grain(ne, G0, T, Z):
        """Compute grain assisted recombination coefficient
        Ch 14.8 in Draine (2011)
        """

        psi = G0*T**0.5/ne
        return RecRate.get_alpha_gr(T, psi, Z)

    def plt_rec_rate(self, Z, N, M=1):

        T = np.logspace(3, 6)
        # Z = ct.EnumAtom.He.value
        # N = Z - 1
        # M = M
        plt.loglog(T, self.get_rec_rate(Z, N, T, M=M), '-')
        plt.loglog(T, self.get_rr_rate(Z, N, T, M=M), ':')
        plt.loglog(T, self.get_dr_rate(Z, N, T, M=M), '--')
        plt.ylim(1e-14, 1e-10)
        return plt.gca()
