import os
import os.path as osp
import pandas as pd
import pathlib

class ReadObs():
    
    def __init__(self):
        
        local = pathlib.Path(__file__).parent.absolute()
        
        self.files = dict()
        self.df = dict()
        self.files['Sun18'] = os.path.join(local,'../../data/Sun18-Table3.txt')
        self.files['Sun20'] = os.path.join(local,'../../data/Sun20-Table3.txt')
        self.df['Sun18'] = self._read_Sun18()
        self.df['Sun20'] = self._read_Sun20()
    
    def get_Sun20(self):
        return self.df['Sun20']

    def get_Sun18(self):
        return self.df['Sun18']
    
    def get_Sun18_Antenna(self):
        return self.df['Sun18'].query('Name == "Antenna"')

    def get_Sun18_M31M33(self):
        return self.df['Sun18'].query('Name == "M31" or Name == "M33"')
    
    def get_Sun18_main_sample(self):
        return self.df['Sun18'].query('Name != "M31" and Name != "M33" and Name != "Antenna"')
    
    def _read_Sun18(self):
        df = pd.read_fwf(self.files['Sun18'], skiprows=28,
                         names=['Name','Res','Tpeak','Sigma','sigma',
                                'avir','Pturb','Mask1','Mask2'])
        return df

    def _read_Sun20(self):
        df = pd.read_fwf(self.files['Sun20'], skiprows=27,
                         names=['Galaxy','inDisk','fCO120pc','Pturb120pc','PDE120pc',
                                'fCO60pc','Pturb60pc','PDE60pc','PDEkpc','PDEkpc11',
                                'SigSFRkpc','Rmolkpc'])
        return df
