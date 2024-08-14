import os
import os.path as osp
import pandas as pd
import pathlib
import astropy.units as au
import astropy.constants as ac
import numpy as np

to_Myr = (1.0*au.pc/au.km*au.s).to('Myr').value
to_Avir = (1.0*au.pc*(au.km/au.s)**2/(ac.G*au.M_sun)).to('')

class ReadObs():

    def __init__(self):

        local = pathlib.Path(__file__).parent.absolute()

        self.files = dict()
        self.df = dict()
        self.files['Sun18'] = os.path.join(local,'../../data/Sun18-Table3.txt')

        self.files['Sun20a'] = os.path.join(local,'../../data/Sun20a-Table3.txt')
        self.files['Sun20bTA'] = os.path.join(local,'../../data/Sun20b-TableA1.txt')
        self.files['Sun20bTB'] = os.path.join(local,'../../data/Sun20b-TableB1.txt')
        self.files['Lee16'] = os.path.join(local,'../../data/Lee16-Table3.txt')
        self.files['Ochsendorf17T4'] = os.path.join(local,'../../data/Ochsendorf17-Table4.txt')
        self.files['Ochsendorf17T5'] = os.path.join(local,'../../data/Ochsendorf17-Table5.txt')
        self.files['VE16T2'] = os.path.join(local,'../../data/Vutisalchavakul16-Table2.txt')
        self.files['VE16T3'] = os.path.join(local,'../../data/Vutisalchavakul16-Table3.txt')
        self.files['Evans14'] = os.path.join(local,'../../data/Evans14-Table1.txt')
        self.files['MD17'] = os.path.join(local,'../../data/MD17-Table1.txt')

        self.files['Bigiel102'] = os.path.join(local,'../../data/Bigiel10-Table2.txt')
        self.files['Bigiel103'] = os.path.join(local,'../../data/Bigiel10-Table3.txt')
        self.files['Leroy08'] = os.path.join(local,'../../data/Leroy08-Table7.txt')

        self.df['Sun18'] = self._read_Sun18()
        self.df['Sun20a'] = self._read_Sun20a()
        self.df['Sun20bTA'], self.df['Sun20bTB'] = self._read_Sun20b()
        self.df['Lee16'] = self._read_Lee16()
        self.df['Ochsendorf17T4'] = self._read_Ochsendorf17T4()
        self.df['Ochsendorf17T5'] = self._read_Ochsendorf17T5()
        self.df['VE16T2'] = self._read_VE16T2()
        self.df['VE16T3'] = self._read_VE16T3()
        self.df['Evans14'] = self._read_Evans14()
        self.df['MD17'] = self._read_MD17()

        self.df['Bigiel10_inner'] = self._read_Bigiel10_inner()
        self.df['Bigiel10_outer'] = self._read_Bigiel10_outer()
        self.df['Leroy08'] = self._read_Leroy08()

    def _read_MD17(self):
        df = pd.read_csv(self.files['MD17'], sep=r'\s+', skiprows=0,
                         names=['Cloud','Ncomp','Npix','A','l','e_l','b','e_b',
                                'theta','WCO','NH2','Sigma','vcent','sigmav','Rmax',
                                'Rmin','Rang','Rgal','INF','Dn','Df','zn','zf','Sn',
                                'Sf','Rn','Rf','Mn','Mf'])

        # Determine mass and radius based on which distance is more likely
        df['D'] = np.where(df['INF'] == 0, df['Dn'], df['Df'])
        df['M'] = np.where(df['INF'] == 0, df['Mn'], df['Mf'])
        df['R'] = np.where(df['INF'] == 0, df['Rn'], df['Rf'])
        # Virial parameter
        df['avir'] = (5.0*(df['sigmav'].values*au.km/au.s)**2\
                      *(df['R'].values*au.pc)/(ac.G*df['M'].values*au.M_sun)).to('')

        return df

    def get_MD17(self):
        return self.df['MD17']

    def _read_Evans14(self):
        df = pd.read_csv(self.files['Evans14'], sep=r'\s+', skiprows=2,
                         names=['Dist','Rcloud','SFR','Mcloud','M_dense','Sigma_SFR',
                                'Sigma_gas','t_ff','sigmav','t_cross'])
        df['SFEff'] = df['SFR']*df['t_ff']/df['Mcloud']

        return df

    def get_Evans14(self):
        return self.df['Evans14']

    def _read_Sun18(self):
        df = pd.read_fwf(self.files['Sun18'], skiprows=28,
                         names=['Name','Res','Tpeak','Sigma','sigma',
                                'avir','Pturb','Mask1','Mask2'])
        return df

    def get_Sun18(self):
        return self.df['Sun18']

    def get_Sun18(self):
        return self.df['Sun18']

    def get_Sun18_Antenna(self):
        return self.df['Sun18'].query('Name == "Antenna"')

    def get_Sun18_M31M33(self):
        return self.df['Sun18'].query('Name == "M31" or Name == "M33"')

    def get_Sun18_main_sample(self):
        return self.df['Sun18'].query('Name != "M31" and Name != "M33" and Name != "Antenna"')

    def _read_Sun20a(self):
        df = pd.read_fwf(self.files['Sun20a'], skiprows=27,
                         names=['Galaxy','inDisk','fCO120pc','Pturb120pc','PDE120pc',
                                'fCO60pc','Pturb60pc','PDE60pc','PDEkpc','PDEkpc11',
                                'SigSFRkpc','Rmolkpc'])
        return df

    def _read_Sun20b(self):
        dfA = pd.read_fwf(self.files['Sun20bTA'], skiprows=46,
                          names=['Galaxy','f_Galaxy','Bar','Arm','Dist','i',
                                 'PA','Mstar','SFR','Reff','Tnoise','rch','fCO','f_fCO','Nlos'])
        dfB = pd.read_fwf(self.files['Sun20bTB'], skiprows=27,
                          names=['Galaxy','scale','rgal','Center','Arm',
                                 'Interarm','ICO21','Sigma','Vdisp','Pturb','alphavir'])
        return dfA,dfB

    def get_Sun20a(self):
        return self.df['Sun20a']

    def get_Sun20b(self):
        return self.df['Sun20bTA'],self.df['Sun20bTB']

    # def _read_Sun20b(self):
    #     df = pd.read_fwf(self.files['Sun20b'], skiprows=27,
    #                      names=['Galaxy','scale','rgal','Center','Arm',
    #                             'Interarm','ICO21','Sigma','Vdisp','Pturb',
    #                             'alphavir'])
    #     return df

    # def get_Sun20b(self):
    #     return self.df['Sun20b']

    def _read_Lee16(self):
        df = pd.read_fwf(self.files['Lee16'], skiprows=30,
                         names=['SFCno','SigV','Dist','Rad','Q',
                                'Mgas','Sigma','SFEbr','SFRbr','tff','Avir'])
        df['tdyn_1d'] = df['Rad']/df['SigV']*to_Myr
        df['tdyn_3d'] = df['Rad']/(np.sqrt(3.0)*df['SigV'])*to_Myr
        # SFEff = Lum/Psi/Mgas*tff/t_*
        # For Halpha, t_*=3.9; see Eq 14. in Lee+16
        df['SFEff'] = df['SFEbr']/3.9*df['tff']

        return df

    def get_Lee16(self):
        return self.df['Lee16']

    def _read_Ochsendorf17T4(self):
        df = pd.read_fwf(self.files['Ochsendorf17T4'], skiprows=25,
                         names=['Name','RAdeg','DEdeg','Type','Mass',
                                'Rad','SFR','SFE','SFEff','tff','sigV'])
        df['tdyn_1d'] = df['Rad']/df['sigV']*to_Myr
        df['tdyn_3d'] = df['Rad']/(np.sqrt(3.0)*df['sigV'])*to_Myr
        df['Avir'] = (5.0*df['Rad'].values*df['sigV'].values**2)/\
                     (df['Mass'].values)*to_Avir
        return df

    def _read_Ochsendorf17T5(self):
        df = pd.read_fwf(self.files['Ochsendorf17T5'], skiprows=25,
                         names=['Name','RAdeg','DEdeg','Type','Mass',
                                'Rad','SFR','SFE','SFEff','tff','sigV'])
        df['tdyn_1d'] = df['Rad']/df['sigV']*to_Myr
        df['tdyn_3d'] = df['Rad']/(np.sqrt(3.0)*df['sigV'])*to_Myr
        df['Avir'] = (5.0*df['Rad'].values*df['sigV'].values**2)/\
                     (df['Mass'].values)*to_Avir
        return df

    def get_Ochsendorf17(self):
        return self.df['Ochsendorf17T4'], self.df['Ochsendorf17T5']

    def _read_VE16T2(self):
        df = pd.read_fwf(self.files['VE16T2'], skiprows=14,
                         names=['ID','SFR-Rad','e_SFR-Rad','SFR-MIR','e_SFR-MIR'])

        return df

    def _read_VE16T3(self):
        df = pd.read_fwf(self.files['VE16T3'], skiprows=32,
                         names=['ID','Type','rCloud','e_rCloud','delv','e_delv','MCloud','e_MCloud',
                                'MVir','e_MVir','alpha','e_alpha','nCloud','e_nCloud','tff','e_tff'])
        # Convert from kMsun to Msun
        for c in ('MCloud','e_MCloud','MVir','e_MVir'):
            df[c] = 1e3*df[c]

        # Dynamical time scale and efficiency per free-fall time
        df['tdyn_3d'] = df['rCloud']/(np.sqrt(3.0)*df['delv'])*to_Myr
        df['tdyn_1d'] = df['rCloud']/df['delv']*to_Myr

        return df

    def get_VE16(self):
        df = pd.merge(self.df['VE16T2'], self.df['VE16T3'], on='ID')
        df['SFEff'] = df['SFR-MIR']*df['tff']/df['MCloud']

        return df

    def _read_Bigiel10_inner(self):
        df = pd.read_fwf(self.files['Bigiel102'], skiprows=49,
                         names=['Sample','Name','logHI','e_logHI',
                                'logH2','e_logH2','logSFR','elogSFR'])
        df['SigmaHI']=10.**(df['logHI'])
        df['SigmaH2']=10.**(df['logH2'])
        df['Sigma_SFR']=10.**(df['logSFR'])

        return df

    def _read_Bigiel10_outer(self):
        df = pd.read_fwf(self.files['Bigiel103'], skiprows=47,
                         names=['Sample','Name','logHI','e_logHI',
                                'SFR','eSFR'])
        df['SigmaHI']=10.**(df['logHI'])
        df['Sigma_SFR']=df['SFR']*1.e-5
        return df

    def get_Bigiel_inner(self):
        return self.df['Bigiel10_inner']

    def get_Bigiel_outer(self):
        return self.df['Bigiel10_outer']

    def _read_Leroy08(self):
        df = pd.read_fwf(self.files['Leroy08'], skiprows=34,
                         names=['Name','Rad','NormRad','SigmaHI','e_SigmaHI',
                                'SigmaH2','e_SigmaH2','Sigma_star','e_Sigma_star',
                                'Sigma_SFR','e_Sigma_SFR','SFR_FUV','SFR_24'])
        df['Sigma_SFR'] *= 1.e-4
        df['e_Sigma_SFR'] *= 1.e-4
        df['SFR_FUV'] *= 1.e-4
        df['SFR_24'] *= 1.e-4
        return df

    def get_Leroy08(self):
        return self.df['Leroy08']
