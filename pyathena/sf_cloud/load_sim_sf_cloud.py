import os
import time
import os.path as osp
import pandas as pd
import numpy as np
from scipy import integrate

from ..util.split_container import split_container
from ..load_sim import LoadSim
from ..util.units import Units
from ..util.cloud import Cloud

from .read_hst import ReadHst
from .slc_prj import SliceProj
from .dust_pol import DustPol

from .compare import Compare
from .virial import Virial

class LoadSimSFCloud(LoadSim, ReadHst, SliceProj, DustPol, Virial):
    """LoadSim class for analyzing sf_cloud simulations.
    """
    
    def __init__(self, basedir, savdir=None, load_method='pyathena',
                 units=Units(kind='LV', muH=1.4271),
                 verbose=False):
        """The constructor for LoadSimSFCloud class

        Parameters
        ----------
        basedir : str
            Name of the directory where all data is stored
        savdir : str
            Name of the directory where pickled data and figures will be saved.
            Default value is basedir.
        load_method : str
            Load vtk using 'pyathena' or 'yt'. Default value is 'pyathena'.
            If None, savdir=basedir. Default value is None.
        verbose : bool or str or int
            Print verbose messages using logger. If True/False, set logger
            level to 'DEBUG'/'WARNING'. If string, it should be one of the string
            representation of python logging package:
            ('NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            Numerical values from 0 ('NOTSET') to 50 ('CRITICAL') are also
            accepted.
        """

        super(LoadSimSFCloud,self).__init__(basedir, savdir=savdir,
                                        load_method=load_method,
                                        units=units,
                                        verbose=verbose)


class LoadSimSFCloudAll(Compare):
    """Class to load multiple simulations"""
    def __init__(self, models=None):

        # Default models
        if models is None:
            models = dict()

        self.models = []
        self.basedirs = dict()
        
        for mdl, basedir in models.items():
            if not osp.exists(basedir):
                print('[LoadSimSFCloudAll]: Model {0:s} doesn\'t exist: {1:s}'.format(
                    mdl,basedir))
            else:
                self.models.append(mdl)
                self.basedirs[mdl] = basedir

    def set_model(self, model, savdir=None, load_method='pyathena', verbose=False):
        
        self.model = model
        self.sim = LoadSimSFCloud(self.basedirs[model], savdir=savdir,
                                  load_method=load_method, verbose=verbose)
        return self.sim

def load_all_alphabeta(force_override=False):

    
    
    models = dict(
        # A series (B=2)
        # A1
        A1S1='/tigress/jk11/GMC/M1E5R20.R.B2.A1.S1.N256',
        A1S2='/tigress/jk11/GMC/M1E5R20.R.B2.A1.S2.N256',
        A1S3='/tigress/jk11/GMC/M1E5R20.R.B2.A1.S3.N256',
        A1S4='/tigress/jk11/GMC/M1E5R20.R.B2.A1.S4.N256',
        A1S5='/tigress/jk11/GMC/M1E5R20.R.B2.A1.S5.N256',

        # A2
        A2S1='/tigress/jk11/GMC/M1E5R20.R.B2.A2.S1.N256',
        A2S2='/tigress/jk11/GMC/M1E5R20.R.B2.A2.S2.N256',
        A2S3='/tigress/jk11/GMC/M1E5R20.R.B2.A2.S3.N256',
        A2S4='/tigress/jk11/GMC/M1E5R20.R.B2.A2.S4.N256',
        A2S5='/tigress/jk11/GMC/M1E5R20.R.B2.A2.S5.N256',

        # A3
        A3S1='/tigress/jk11/GMC/M1E5R20.R.B2.A3.S1.N256',
        A3S2='/tigress/jk11/GMC/M1E5R20.R.B2.A3.S2.N256',
        A3S3='/tigress/jk11/GMC/M1E5R20.R.B2.A3.S3.N256',
        A3S4='/tigress/jk11/GMC/M1E5R20.R.B2.A3.S4.N256',
        A3S5='/tigress/jk11/GMC/M1E5R20.R.B2.A3.S5.N256',
        
        # A4
        A4S1='/tigress/jk11/GMC/M1E5R20.R.B2.A4.S1.N256',
        A4S2='/tigress/jk11/GMC/M1E5R20.R.B2.A4.S2.N256',
        A4S3='/tigress/jk11/GMC/M1E5R20.R.B2.A4.S3.N256',
        A4S4='/tigress/jk11/GMC/M1E5R20.R.B2.A4.S4.N256',
        A4S5='/tigress/jk11/GMC/M1E5R20.R.B2.A4.S5.N256',

        # A5
        A5S1='/tigress/jk11/GMC/M1E5R20.R.B2.A5.S1.N256',
        A5S2='/tigress/jk11/GMC/M1E5R20.R.B2.A5.S2.N256',
        A5S3='/tigress/jk11/GMC/M1E5R20.R.B2.A5.S3.N256',
        A5S4='/tigress/jk11/GMC/M1E5R20.R.B2.A5.S4.N256',
        A5S5='/tigress/jk11/GMC/M1E5R20.R.B2.A5.S5.N256',

        # B series (A=2)
        # B0.5
        B05S1='/perseus/scratch/gpfs/jk11/GMC/M1E5R20.R.B0.5.A2.S1.N256',
        B05S2='/perseus/scratch/gpfs/jk11/GMC/M1E5R20.R.B0.5.A2.S2.N256',
        B05S3='/perseus/scratch/gpfs/jk11/GMC/M1E5R20.R.B0.5.A2.S3.N256',
        B05S4='/perseus/scratch/gpfs/jk11/GMC/M1E5R20.R.B0.5.A2.S4.N256',
        #B05S5='/tigress/jk11/GMC/M1E5R20.R.B0.5.A2.S5.N256',

        # B1
        B1S1='/tigress/jk11/GMC/M1E5R20.R.B1.A2.S1.N256',
        B1S2='/tigress/jk11/GMC/M1E5R20.R.B1.A2.S2.N256',
        B1S3='/tigress/jk11/GMC/M1E5R20.R.B1.A2.S3.N256',
        B1S4='/tigress/jk11/GMC/M1E5R20.R.B1.A2.S4.N256',
        B1S5='/tigress/jk11/GMC/M1E5R20.R.B1.A2.S5.N256',

        # B2
        B2S1='/tigress/jk11/GMC/M1E5R20.R.B2.A2.S1.N256',
        B2S2='/tigress/jk11/GMC/M1E5R20.R.B2.A2.S2.N256',
        B2S3='/tigress/jk11/GMC/M1E5R20.R.B2.A2.S3.N256',
        B2S4='/tigress/jk11/GMC/M1E5R20.R.B2.A2.S4.N256',
        B2S5='/tigress/jk11/GMC/M1E5R20.R.B2.A2.S5.N256',

        # B4
        B4S1='/tigress/jk11/GMC/M1E5R20.R.B4.A2.S1.N256',
        B4S2='/tigress/jk11/GMC/M1E5R20.R.B4.A2.S2.N256',
        B4S3='/tigress/jk11/GMC/M1E5R20.R.B4.A2.S3.N256',
        B4S4='/tigress/jk11/GMC/M1E5R20.R.B4.A2.S4.N256',
        B4S5='/tigress/jk11/GMC/M1E5R20.R.B4.A2.S5.N256',

        # B8
        B8S1='/tigress/jk11/GMC/M1E5R20.R.B8.A2.S1.N256',
        B8S2='/tigress/jk11/GMC/M1E5R20.R.B8.A2.S2.N256',
        B8S3='/tigress/jk11/GMC/M1E5R20.R.B8.A2.S3.N256',
        B8S4='/tigress/jk11/GMC/M1E5R20.R.B8.A2.S4.N256',
        B8S5='/tigress/jk11/GMC/M1E5R20.R.B8.A2.S5.N256',

        # Binf
        BinfS1='/tigress/jk11/GMC/M1E5R20.R.Binf.A2.S1.N256.again',
        BinfS2='/tigress/jk11/GMC/M1E5R20.R.Binf.A2.S2.N256',
        BinfS3='/tigress/jk11/GMC/M1E5R20.R.Binf.A2.S3.N256',
        BinfS4='/tigress/jk11/GMC/M1E5R20.R.Binf.A2.S4.N256',
        BinfS5='/tigress/jk11/GMC/M1E5R20.R.Binf.A2.S5.N256',
        
        # B16
        # B16S1='/tigress/jk11/GMC/M1E5R20.R.B16.A2.S1.N256.old',
        )
    
    sa = LoadSimSFCloudAll(models)

    markers = ['o','v','^','s','*']

    # Check if pickle exists
    fpkl = osp.join('/tigress/jk11/GMC/pickles/alphabeta.p')
    if not force_override and osp.isfile(fpkl):
        r = pd.read_pickle(fpkl)
        return sa, r

    df_list = []

    # Save key results to a single dataframe
    for mdl in sa.models:
        print(mdl, end=' ')
        s = sa.set_model(mdl, verbose=False)
        h = s.read_hst(force_override=False)
        par = s.par
        cl = Cloud(par['problem']['M_cloud'], par['problem']['R_cloud'])
    
        df = dict()
        df['basedir'] = s.basedir
        # Input parameters
        df['mhd'] = par['configure']['gas'] == 'mhd'
        df['Nx'] = par['domain1']['Nx1']
        
        df['M'] = float(par['problem']['M_cloud'])
        df['R'] = float(par['problem']['R_cloud'])
        df['Sigma'] = df['M']/(np.pi*df['R']**2)
        df['seed'] = int(np.abs(par['problem']['rseed']))
        df['alpha_vir'] = float(par['problem']['alpha_vir'])
        df['marker'] = markers[df['seed'] - 1]
        df['vesc'] = cl.vesc.to('km/s').value
        df['sigma1d'] = cl.sigma1d.to('km/s').value
        df['tff'] = cl.tff.to('Myr').value
        if df['mhd']:
            df['mu'] = float(par['problem']['muB'])
            df['label'] = r'B{0:d}.A{1:d}.S{2:d}'.\
                          format(int(df['mu']),int(df['alpha_vir']),int(df['seed']))
        else:
            df['mu'] = np.inf
            df['label'] = r'Binf.A{0:d}.S{1:d}'.\
                          format(int(df['alpha_vir']),int(df['seed']))
            
        df['par'] = par
        df['hst'] = h
        
        # Simulation results
        #Mstar_final = h['Mstar'].iloc[-1]
        Mstar_final = max(h['Mstar'].values)
        df['Mstar_final'] = Mstar_final
        df['SFE'] = Mstar_final/df['M']
        df['t_final'] = h['time'].iloc[-1]
        df['tau_final'] = df['t_final']/df['tff']
        
        idx_SF0, = h['Mstar'].to_numpy().nonzero()
        if len(idx_SF0):
            df['t_*'] = h['time'][idx_SF0[0]]
            df['tau_*'] = df['t_*']/df['tff']
            # df['t_95%'] = h['time'][h.Mstar > 0.95*Mstar_final].values[0]
            # df['tau_95%'] = df['t_95%']/df['tff']
            df['t_90%'] = h['time'][h.Mstar > 0.95*Mstar_final].values[0]
            df['tau_90%'] = df['t_90%']/df['tff']
            df['t_SF'] = df['t_90%'] - df['t_*']
            df['tau_SF'] = df['t_SF']/df['tff']
            df['t_SF2'] = Mstar_final**2 / \
                        integrate.trapz(h['SFR']**2, h.time)
            df['tau_SF2'] = df['t_SF2']/df['tff']
        else:
            df['t_*'] = np.nan
            df['tau_*'] = np.nan
            # df['t_95%'] = np.nan
            # df['tau_95%'] = np.nan
            df['t_90%'] = np.nan
            df['tau_90%'] = np.nan
            df['t_SF'] = np.nan
            df['tau_SF'] = np.nan
            df['t_SF2'] = np.nan
            df['tau_SF2'] = np.nan

            
        df_list.append(pd.DataFrame(pd.Series(df, name=mdl)).T)

    df = pd.concat(df_list).sort_index(ascending=False)

    df.to_pickle(fpkl)

    return sa, df
