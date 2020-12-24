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
        self.files['Sun20'] = os.path.join(local,'../../data/Sun20-Table3.txt')
        self.files['Lee16'] = os.path.join(local,'../../data/Lee16-Table3.txt')
        self.files['Ochsendorf17T4'] = os.path.join(local,'../../data/Ochsendorf17-Table4.txt')
        self.files['Ochsendorf17T5'] = os.path.join(local,'../../data/Ochsendorf17-Table5.txt')
        self.files['VE16T2'] = os.path.join(local,'../../data/Vutisalchavakul16-Table2.txt')
        self.files['VE16T3'] = os.path.join(local,'../../data/Vutisalchavakul16-Table3.txt')
        self.files['Evans14'] = os.path.join(local,'../../data/Evans14-Table1.txt')
        self.df['Sun18'] = self._read_Sun18()
        self.df['Sun20'] = self._read_Sun20()
        self.df['Lee16'] = self._read_Lee16()
        self.df['Ochsendorf17T4'] = self._read_Ochsendorf17T4()
        self.df['Ochsendorf17T5'] = self._read_Ochsendorf17T5()
        self.df['VE16T2'] = self._read_VE16T2()
        self.df['VE16T3'] = self._read_VE16T3()
        self.df['Evans14'] = self._read_Evans14()
    
    def _read_Evans14(self):
        df = pd.read_csv(self.files['Evans14'], sep='\s+', skiprows=2,
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
    
    def _read_Sun20(self):
        df = pd.read_fwf(self.files['Sun20'], skiprows=27,
                         names=['Galaxy','inDisk','fCO120pc','Pturb120pc','PDE120pc',
                                'fCO60pc','Pturb60pc','PDE60pc','PDEkpc','PDEkpc11',
                                'SigSFRkpc','Rmolkpc'])
        return df

    def get_Sun20(self):
        return self.df['Sun20']

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
