import pathlib
import os
import os.path as osp
import pandas as pd
import copy
import numpy as np
import astropy.units as au
import astropy.constants as ac

class ChargeTransferRate(object):

    def __init__(self):
        self.basedir = osp.join(pathlib.Path(__file__).parent.absolute(),
                                '../../data/microphysics')

        self.name_to_Z = dict(H=1, He=2, Li=3, Be=4, B=5, C=6, N=7, O=8, F=9,
                              Ne=10, Na=11, Mg=12, Al=13, Si=14, P=15, S=16, Cl=17, Ar=18,
                              K=19, Ca=20, Sc=21, Ti=22, V=23, Cr=24, Mn=25, Fe=26, Co=27,
                              Ni=28, Cu=29, Zn=30)

        # read data
        self._read_data_cloudy()
        self._read_data_uga()

    def _read_data_uga(self):
        self.dfi = self._read_cti_hyd()
        self.dfr = self._read_ctr_hyd()

    def _read_ctr_hyd(self):
        fname = os.path.join(self.basedir, 'ugacxdb/ctr_hyd.dat')
        df = pd.read_csv(fname, skiprows=22, skipfooter=2, delimiter=r'\s+',
                         names=['ion','a','b','c','d','tlo','thi','R'], engine='python')

        name = [ion.split('^')[0] for ion in df['ion']]
        df['Z'] = np.array([self.name_to_Z[name_] for name_ in name])
        # N = Z - charge
        df['N'] = df['Z'] - np.array([int(ion.split('+')[-1]) for ion in df['ion']])
        cols = df.columns.tolist()
        df = df[cols[-2:] + cols[:-2]]
        return df

    def _read_cti_hyd(self):
        fname = os.path.join(self.basedir, 'ugacxdb/cti_hyd.dat')
        df = pd.read_csv(fname, skiprows=24, skipfooter=6, delimiter=r'\s+',
                         names=['ion','a','b','c','d','tlo','thi','Ek','R'], engine='python')
        name = [ion.split('^')[0] for ion in df['ion']]
        df['Z'] = np.array([self.name_to_Z[name_] for name_ in name])
        # N = Z - charge
        df['N'] = df['Z'] - np.array([int(ion.split('+')[-1]) for ion in df['ion']])
        cols = df.columns.tolist()
        df = df[cols[-2:] + cols[:-2]]
        return df

    def _read_data_cloudy(self, nelem=30):

        # basedir = osp.join(pathlib.Path(__file__).parent.absolute(),
        #                    '../data/microphysics/cloudy')
        self.fname_ion = os.path.join(self.basedir, 'cloudy/ctiondata.dat')
        self.fname_rec = os.path.join(self.basedir, 'cloudy/ctrecombdata.dat')

        # CT Ionization
        # Note: First parameter is in units of 1e-9 [cm^3/s]
        # Note: Seventh parameter is in units of 1e4 K

        # 120 = 4 (ion charge) * 30 elements
        # 8 x 120
        lines_ion = np.loadtxt(self.fname_ion, unpack=True, skiprows=1)
        # 7 x 120
        lines_rec = np.loadtxt(self.fname_rec, unpack=True, skiprows=1)

        self.Z1 = np.repeat(np.arange(1,31),4)
        self.Z2 = np.repeat(np.arange(1,31),4)
        self.N1 = []
        self.N2 = []
        for Z in range(1, 31):
            for q in range(0, 4):
                self.N1.append(np.maximum(Z - q, 1))
                self.N2.append(np.maximum(Z - q, 1))

        self.N1 = np.array(self.N1)
        self.N2 = np.array(self.N2)

        self.a1 = lines_ion[0,:]
        self.b1 = lines_ion[1,:]
        self.c1 = lines_ion[2,:]
        self.d1 = lines_ion[3,:]
        self.Tmin1 = lines_ion[4,:]
        self.Tmax1 = lines_ion[5,:]

        # dE_rate : Energy deficit in 10^4 K for rate calculation (Kingdon & Ferland 1996)
        # dE_heat : Energy deficit in eV for heating calculation (Kingdon & Ferland 1999)
        # Why are they different?
        # maybe because there are multiple states (see Kingdon+99).
        self.dE_rate1 = lines_ion[6,:]
        self.dE_heat1 = lines_ion[7,:]

        self.a2 = lines_rec[0,:]
        self.b2 = lines_rec[1,:]
        self.c2 = lines_rec[2,:]
        self.d2 = lines_rec[3,:]
        self.Tmin2 = lines_rec[4,:]
        self.Tmax2 = lines_rec[5,:]
        self.dE_heat2 = lines_rec[6,:]

        # Modify coefficient for CT recombination of HeII with HI
        # Cloudy uses the value in Table 1 of Kingdon+96.
        # Confusingly, however, they use "+=" operator...
        # see line 93 in atmdat_char_tran.cpp
        i = self._get_index2(2, 2)
        self.a2[4] = 7.47e-6

        # Modify coefficient for CT ionization of
        # FeI with HII (26)
        # AlI with HII (13)
        # PI with HII (15)
        # ClI with HII (17)
        # TiI with HII (22)
        # MnI with HII (25)
        # NiI with HII (28)
        # NaI with HII (11) # Sum of channels that deposit into HI(n=1) and HI(n=2)
        # KI with HII (19)
        # SI with HII (16)
        # See cloudy source file for details
        Zs = [26, 13, 15, 17, 22, 25, 28, 11, 19, 16]
        Ns = [26, 13, 15, 17, 22, 25, 28, 11, 19, 16]
        # Note the unit 1e-9 [cm^3/s]
        a1s = [5.4, 3.0, 1.0, 1.0, 3.0, 3.0, 7.7e-3, 1.25e-3, 1e-5]

        for Z, N, a1 in zip(Zs, Ns, a1s):
            i = self._get_index1(Z, N)
            self.a1[i] = a1
            self.b1[i] = 0.0
            self.c1[i] = 0.0
            self.d1[i] = 0.0
            self.Tmin1[i] = 0.0
            self.Tmax1[i] = 1e10
            self.dE_rate1[i] = 0.0
            self.dE_heat1[i] = 0.0

    def get_ct_rec_rate_uga(self, Z, N, T):
        iZ = Z == self.dfr['Z']
        iN = N == self.dfr['N']
        idx = np.where(iZ & iN)[0][0]
        d = self.dfr.iloc[idx]

        T = np.where(T >= d.tlo, T, d.tlo)
        T = np.where(T <= d.thi, T, d.thi)
        T4 = T*1e-4
        # Eq. 7 in Kingdon & Ferland (1996)

        return 1e-9*d.a*T4**(d.b)*(1.0 + d.c*np.exp(d.d*T4))

    def get_ct_ion_rate_uga(self, Z, N, T):
        iZ = Z == self.dfi['Z']
        iN = N == self.dfi['N']
        idx = np.where(iZ & iN)[0][0]
        d = self.dfi.iloc[idx]

        T_in = copy.deepcopy(T)
        T = np.where(T >= d.tlo, T, d.tlo)
        T = np.where(T <= d.thi, T, d.thi)
        T4 = T*1e-4
        # Eq. 7 in Kingdon & Ferland (1996)
        return 1e-9*d.a*T4**(d.b)*(1.0 + d.c*np.exp(d.d*T4))*np.exp(-d.Ek*1e4/T_in)

    def get_ct_ion_rate(self, Z, N, T):
        """
        Charge transfer ionization rate coefficient for
        Ion(Z,q) + H -> Ion(Z,q+1) + H+
        where q = Z - N is the ion charge. That is, (Z,N) is the reactant.
        """

        # Charge
        q = Z - N
        if q < 0:
            raise ValueError('Check Z and N')
        if q > 3:
            return np.zeros_like(T)

        # Catch exceptions
        if (Z == 8) and (N == 8):
            #return ChargeTransferRate.get_ct_ion_OI_HII(T)
            return ChargeTransferRate.get_ct_ion_OI_HII_Draine11(T)
        elif (Z == 12) and (N == 12):
            return ChargeTransferRate.get_ct_ion_MgI_HII(T)
        elif (Z == 14) and (N == 14):
            return ChargeTransferRate.get_ct_ion_SiI_HII(T)

        i = self._get_index1(Z, N)
        if self.a1[i] == 0.0:
            return np.zeros_like(T)
        else:
            T_in = copy.deepcopy(T)
            T = np.where(T >= self.Tmin1[i], T, self.Tmin1[i])
            T = np.where(T <= self.Tmax1[i], T, self.Tmax1[i])
            T4 = T*1e-4
            # Recombination rate multiplied by the Boltzmann factor
            # Use input T for Boltzman factor calculation
            rate = 1e-9*self.a1[i]*T4**(self.b1[i])*(1.0 + self.c1[i]*np.exp(self.d1[i]*T4))*\
                np.exp(-self.dE_rate1[i]*1e4/T_in)

        return rate

    def get_ct_rec_rate(self, Z, N, T):
        """
        Charge transfer recombination rate coefficient for
        Ion(Z,q+1) + H+ -> Ion(Z,q) + H
        where q = Z - N is the ion charge. That is, (Z,N) is the product.
        """

        # Charge
        q = Z - N
        rate_Dalgarno = 1.92e-9
        if q < 0:
            raise ValueError('Check Z and N')
        if q > 3:
            return np.full_like(T, rate_Dalgarno)*(q + 1)

        # Catch exceptions
        if (Z == 8) and (N == 8):
            # return ChargeTransferRate.get_ct_rec_OII_HI(T)
            return ChargeTransferRate.get_ct_rec_HI_OII_Draine11(T)

        i = self._get_index2(Z, N)

        if self.a2[i] == 0.0:
            return np.zeros_like(T)
        else:
            T = np.where(T >= self.Tmin2[i], T, self.Tmin2[i])
            T = np.where(T <= self.Tmax2[i], T, self.Tmax2[i])
            T4 = T*1e-4
            # Eq. 7 in Kingdon & Ferland (1996)
            rate = 1e-9*self.a2[i]*T4**(self.b2[i])*(1.0 + self.c2[i]*np.exp(self.d2[i]*T4))

        return rate

    def _get_index1(self, Z, N):
        iZ = Z == self.Z1
        iN = N == self.N1
        return np.where(iZ & iN)[0][0]

    def _get_index2(self, Z, N):
        iZ = Z == self.Z2
        iN = N == self.N2
        return np.where(iZ & iN)[0][0]

    @staticmethod
    def get_ct_rec_HI_OII(T):
        lnT = np.log(T)
        rate = np.where(T <= 10.0,
                        3.744e-10,
                        ((((1.1963502e-13*lnT - 2.8577012e-12)*lnT + 2.9979994e-11)*lnT
                            - 1.3146803e-10)*lnT + 2.3651505e-10)*lnT + 2.3344302e-10)
        return rate

    @staticmethod
    def get_ct_ion_HII_OI(T):
        lnT = np.log(T)
        rate = np.where(T <= 10.0,
                        4.749e-20,
                        np.where(T <= 190.0,
                                 np.exp(-21.134531 - 242.06831/T + 84.761441/(T*T)),
                                 np.where(T <= 200.0,\
                2.18733e-12*(T - 190.0) + 1.85823e-10,
                (((((1.1580844e-14*lnT - 2.6139493e-13)*lnT + 2.0699463e-12)*lnT
                   - 3.6606214e-12)*lnT - 1.488594e-12)*lnT - 3.7282001e-13)*lnT
                                          - 7.6767404e-14)))
        return rate

    @staticmethod
    def get_ct_ion_MgI_HII(T):
        # valid for temp 5e3 to 3e4
        T4 = T*1e-4
        T4 = np.where(T4 > 10.0, 10.0, T4)
        return 9.76e-12*(T4**3.14)*(1.0 + 55.54*np.exp(-1.12*T4))

    @staticmethod
    def get_ct_ion_SiI_HII(T):
        T4 = T*1e-4
        T4 = np.where(T4 > 10.0, 10.0, T4)
        return 0.92e-12*(T4**1.15)*(1.0 + 0.80*np.exp(-0.24*T4))

    @staticmethod
    def get_ct_rec_HI_OII_Draine11(T, sum=True):
        Tinv = 1.0/T
        T4 = 1.0e-4*T
        lnT4 = np.log(T4)
        # Charge exchange rate for recombination
        k0r = 1.14e-9*T4**(0.4 + 0.018*lnT4)
        k1r = 3.44e-10*T4**(0.451 + 0.036*lnT4)
        k2r = 5.33e-10*T4**(0.384 + 0.024*lnT4)*np.exp(-97.0*Tinv)
        if sum:
            return k0r + k1r + k2r
        else:
            return k0r, k1r, k2r

    @staticmethod
    def get_ct_ion_HII_OI_Draine11(T, sum=True):
        k0r, k1r, k2r = ChargeTransferRate.get_ct_rec_HI_OII_Draine11(T, sum=False)
        Tinv = 1.0/T
        k0i = 8.0/5.0*k0r*np.exp(-229.0*Tinv)
        k1i = 8.0/3.0*k1r*np.exp(-1.0*Tinv)
        k2i = 8.0*k2r*np.exp(97.0*Tinv)
        if sum:
            return k0i + k1i + k2i
        else:
            return k0i, k1i, k2i
