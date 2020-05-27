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

from .hst import Hst
from .slc_prj import SliceProj
from .dust_pol import DustPol

from .compare import Compare
from .pdf import PDF
from .virial import Virial
from .sfr import get_SFR_mean

class LoadSimSFCloud(LoadSim, Hst, SliceProj, PDF,
                     DustPol, Virial):
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

    def get_summary(self, as_dict=False):
        markers = ['o','v','^','s','*']

        par = self.par
        cl = Cloud(par['problem']['M_cloud'], par['problem']['R_cloud'])

        df = dict()
        df['par'] = par

        # Read hst, virial analysis
        h = self.read_hst(force_override=False)
        df['hst'] = h
        
        if (par['configure']['gas'] == 'mhd') and \
           (int(par['domain1']['Nx1']) == 256):
            h_vir = self.read_virial_all(force_override=True)
            df['hst_vir'] = h_vir
        else:
            df['hst_vir'] = None
    
        df['basedir'] = self.basedir
        df['domain'] = self.domain
        # Input parameters
        df['mhd'] = par['configure']['gas'] == 'mhd'
        df['Nx'] = int(par['domain1']['Nx1'])
        
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
            df['muB'] = float(par['problem']['muB'])
            df['label'] = r'B{0:d}.A{1:d}.S{2:d}'.\
                          format(int(df['muB']),int(df['alpha_vir']),int(df['seed']))
        else:
            df['muB'] = np.inf
            df['label'] = r'Binf.A{0:d}.S{1:d}'.\
                          format(int(df['alpha_vir']),int(df['seed']))            
        
        # Simulation results
        # Mstar_final = h['Mstar'].iloc[-1]
        Mstar_final = max(h['Mstar'].values)
        df['Mstar_final'] = Mstar_final
        df['SFE'] = Mstar_final/df['M']
        df['t_final'] = h['time'].iloc[-1]
        df['tau_final'] = df['t_final']/df['tff']
        
        idx_SF0, = h['Mstar'].to_numpy().nonzero()
        if len(idx_SF0):
            df['t_*'] = h['time'][idx_SF0[0]-1]
            df['tau_*'] = df['t_*']/df['tff']
            df['t_95%'] = h['time'][h.Mstar > 0.95*Mstar_final].values[0]
            df['tau_95%'] = df['t_95%']/df['tff']
            df['t_90%'] = h['time'][h.Mstar > 0.90*Mstar_final].values[0]
            df['tau_90%'] = df['t_90%']/df['tff']
            df['t_80%'] = h['time'][h.Mstar > 0.80*Mstar_final].values[0]
            df['tau_80%'] = df['t_80%']/df['tff']
            df['t_50%'] = h['time'][h.Mstar > 0.50*Mstar_final].values[0]
            df['tau_50%'] = df['t_50%']/df['tff']
            df['t_SF'] = df['t_90%'] - df['t_*']
            df['tau_SF'] = df['t_SF']/df['tff']
            df['t_SF2'] = Mstar_final**2 / \
                        integrate.trapz(h['SFR']**2, h.time)
            df['tau_SF2'] = df['t_SF2']/df['tff']
            df['SFR_mean'] = get_SFR_mean(h, 0.0, 90.0)['SFR_mean']
        else:
            df['t_*'] = np.nan
            df['tau_*'] = np.nan
            df['t_95%'] = np.nan
            df['tau_95%'] = np.nan
            df['t_90%'] = np.nan
            df['tau_90%'] = np.nan
            df['t_80%'] = np.nan
            df['tau_80%'] = np.nan
            df['t_50%'] = np.nan
            df['tau_50%'] = np.nan
            df['t_SF'] = np.nan
            df['tau_SF'] = np.nan
            df['t_SF2'] = np.nan
            df['tau_SF2'] = np.nan
            df['SFR_mean'] = np.nan

        try:
            df['fesc_cum_PH'] = h['fesc_cum_PH'].iloc[-1] # Lyman Continuum
            df['fesc_cum_FUV'] = h['fesc_cum_FUV'].iloc[-1]
        except KeyError:
            df['fesc_cum_PH'] = np.nan
            df['fesc_cum_FUV'] = np.nan
            
        if as_dict:
            return df
        else:
            return pd.Series(df, name=self.basename)
        
    def get_dt_output(self):

        r = dict()
        r['vtk'] = None
        r['hst'] = None
        r['vtk_sp'] = None
        r['rst'] = None
        r['vtk_2d'] = None
        
        for i in range(self.par['job']['maxout']):
            b = f'output{i+1}'
            if self.par[b]['out_fmt'] == 'vtk' and \
               (self.par[b]['out'] == 'prim' or self.par[b]['out'] == 'cons'):
                r['vtk'] = self.par[b]['dt']
            elif self.par[b]['out_fmt'] == 'hst':
                r['hst'] = self.par[b]['dt']
            elif self.par[b]['out_fmt'] == 'starpar_vtk':
                r['vtk_sp'] = self.par[b]['dt']
            elif self.par[b]['out_fmt'] == 'rst':
                r['rst'] = self.par[b]['dt']
            elif self.par[b]['out_fmt'] == 'vtk' and \
               ('Sigma' in self.par[b]['out']) or ('EM' in self.par[b]['out']):
                r['vtk_2d'] = self.par[b]['dt']

        self.dt_output = r
        
        return r 
        
    def get_nums(self, t_Myr=None, dt_Myr=None, sp_frac=None, rounding=True,
                 output='vtk'):
        """Function to determine output snapshot numbers
        from (1) t_Myr or from (2) dt_Myr relative to the time of first SF,
        or (3) from sp_frac (0 - 100%).

        Parameters
        ----------
        t_Myr : array-like or scalar
            Time of snapshots in Myr
        dt_Myr : array-like or scalar
            (time - time of first SF) of snapshots in Myr
        sp_frac : (sequence of) float
            Fraction of final stellar mass (0 < sp_frac < 1)
        output : str
            Output type: 'vtk', 'starpar_vtk', 'vtk_2d', 'hst', 'rst'
        """

        u = self.u
        # Find time at which xx percent of SF has occurred
        if t_Myr is not None:
            t_Myr = np.atleast_1d(t_Myr)
            t_code = [t_Myr_/u.Myr for t_Myr_ in t_Myr]
        elif dt_Myr is not None:
            dt_Myr = np.atleast_1d(dt_Myr)
            h = self.read_hst()
            idx_SF0, = h['Mstar'].to_numpy().nonzero()
            t0 = h['time_code'][idx_SF0[0] - 1]
            t_code = [t0 + dt_Myr_/u.Myr for dt_Myr_ in dt_Myr]
            # print('time of first SF [Myr]', t0*self.u.Myr)
            # print('time of first SF [code]', t0)
        elif sp_frac is not None:
            sp_frac = np.atleast_1d(sp_frac)
            h = self.read_hst()
            Mstar_final = h['Mstar'].iloc[-1]
            idx = [np.where(h['Mstar'] > sp_frac_*Mstar_final)[0][0] \
                   for sp_frac_ in sp_frac]
            t_code = [h['time_code'].iloc[idx_] for idx_ in idx]

        nums = []
        dt_output = self.get_dt_output()[output]
        for t in t_code:
            if rounding:
                num = int(round(t/dt_output))
            else:
                num = int(t/dt_output)
                
            nums.append(num)

        if len(nums) == 1:
            nums = nums[0]

        return nums
    
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
        B05S1='/tigress/jk11/GMC/M1E5R20.R.B0.5.A2.S1.N256',
        B05S2='/tigress/jk11/GMC/M1E5R20.R.B0.5.A2.S2.N256',
        B05S3='/tigress/jk11/GMC/M1E5R20.R.B0.5.A2.S3.N256',
        B05S4='/tigress/jk11/GMC/M1E5R20.R.B0.5.A2.S4.N256',
        B05S5='/tigress/jk11/GMC/M1E5R20.R.B0.5.A2.S5.N256',

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

        # Low resolution
        B2S4_N128='/perseus/scratch/gpfs/jk11/GMC/M1E5R20.R.B2.A2.S4.N128.again/'
        
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
        df = s.get_summary(as_dict=True)
        df_list.append(pd.DataFrame(pd.Series(df, name=mdl)).T)

    df = pd.concat(df_list).sort_index(ascending=False)

    df.to_pickle(fpkl)

    return sa, df
