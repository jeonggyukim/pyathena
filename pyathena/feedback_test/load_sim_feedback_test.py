import os
import os.path as osp
import pandas as pd
import getpass
import matplotlib as mpl
import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt

from ..load_sim import LoadSim
from ..util.units import Units

from .hst import Hst
from .dust_pol import DustPol
from .profile_1d import Profile1D

class LoadSimFeedbackTest(LoadSim, Hst, DustPol, Profile1D):
    """LoadSim class for analyzing LoadSimFeedbackTest simulations.
    """
    
    def __init__(self, basedir, savdir=None, load_method='pyathena',
                 units=Units(kind='LV', muH=1.4271),
                 verbose=False):
        """The constructor for LoadSimFeedbackTest class

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

        super(LoadSimFeedbackTest,self).__init__(basedir, savdir=savdir,
                                                 load_method=load_method,
                                                 units=units,
                                                 verbose=verbose)

    def get_nums(self, t_Myr=None, rounding=True,
                 output='vtk'):
        """Function to determine output snapshot numbers
        from t_Myr

        Parameters
        ----------
        t_Myr : array-like or scalar
            Time of snapshots in Myr
        output : str
            Output type: 'vtk', 'starpar_vtk', 'vtk_2d', 'hst', 'rst'
        """

        u = self.u
        # Find time at which xx percent of SF has occurred
        if t_Myr is not None:
            t_Myr = np.atleast_1d(t_Myr)
            t_code = [t_Myr_/u.Myr for t_Myr_ in t_Myr]

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
    
    def get_dt_output(self):

        r = dict()
        r['vtk'] = None
        r['hst'] = None
        r['vtk_sp'] = None
        r['rst'] = None
        r['vtk_2d'] = None
        
        for i in range(self.par['job']['maxout']):
            b = f'output{i+1}'                
            try:
                if self.par[b]['out_fmt'] == 'vtk' and \
                   (self.par[b]['out'] == 'prim' or self.par[b]['out'] == 'cons'):
                    r['vtk'] = self.par[b]['dt']
                elif self.par[b]['out_fmt'] == 'hst':
                    r['hst'] = self.par[b]['dt']
                elif self.par[b]['out_fmt'] == 'starpar_vtk':
                    r['vtk_sp'] = self.par[b]['dt']
                elif self.par[b]['out_fmt'] == 'rst':
                    r['rst'] = self.par[b]['dt']
                elif self.par[b]['out_fmt'] == 'vtk':
                    r['vtk_2d'] = self.par[b]['dt']
            except KeyError:
                continue
        self.dt_output = r
        
        return r 

    def get_summary_sn(self, as_dict=False):

        par = self.par
        df = dict()
        df['par'] = par

        h = self.read_hst(force_override=True)
        df['hst'] = h
        df['basedir'] = self.basedir
        df['domain'] = self.domain
        df['Nx'] = int(par['domain1']['Nx1'])

        # Input parameters
        df['mhd'] = par['configure']['gas'] == 'mhd'

        df['iWind'] = par['feedback']['iWind']
        df['iSN'] = par['feedback']['iSN']
        df['irayt'] = par['radps']['irayt']
        try:
            df['iPhot'] = par['radps']['iPhot']
        except:
            df['iPhot'] = par['radps']['iPhotIon']

        df['iRadp'] = par['radps']['apply_force']

        df['n0'] = par['problem']['n0']
        df['Z_gas'] = par['problem']['Z_gas']
        df['Z_dust'] = par['problem']['Z_dust']

        # Initial feedback radius (rinit in Kim & Ostriker 2015)
        df['dx'] = df['domain']['dx'][0]
        df['r_init'] = par['problem']['rblast_over_hdx']*(0.5*df['dx'])

        # Quantities at the time of shell formation
        # Shell formation time
        df['t_sf'] = float(h.loc[(h['Mi']+h['Mh']).max() ==
                                (h['Mi']+h['Mh']), 'time'])
        # Radius at the time of shell formation
        df['r_sf'] = float(h.loc[h['time'] == df['t_sf'], 'Rsh'])
        # Mass of ionized and hot gas (T > 2e4K)
        df['Mhi_sf'] = float(h.loc[h['time'] == df['t_sf'], 'Mhi'])
        # Shell mass
        df['Msh_sf'] = float(h.loc[h['time'] == df['t_sf'], 'Msh'])
        # Momentum
        df['pr_sf'] = float(h.loc[h['time'] == df['t_sf'], 'pr'])
        df['pok_bub_sf'] = float(h.loc[h['time'] == df['t_sf'], 'pok_bub'])
        df['Ethm_sf'] = float(h.loc[h['time'] == df['t_sf'], 'Ethm'])
        df['Ekin_sf'] = float(h.loc[h['time'] == df['t_sf'], 'Ekin'])
        df['vrsh_sf'] = float(h.loc[h['time'] == df['t_sf'], 'vrsh'])
        df['vrbub_sf'] = float(h.loc[h['time'] == df['t_sf'], 'vrbub'])

        df['dx_over_r_sf'] = df['dx']/df['r_sf']
        
        # Momentum at 10*t_sf
        idx = (h['time'] - 10.0*df['t_sf']).abs().argsort()[0]
        df['pr_10t_sf'] = h['pr'].iloc[idx]

        # Plot styles
        df['cmapZ'] = mpl.cm.plasma_r
        df['normZ'] = mpl.colors.LogNorm(0.003,3.0)
        df['linecolorZ'] = df['cmapZ'](df['normZ'](df['Z_gas']))

        df['cmapn'] = cmr.cosmic_r
        df['normn'] = mpl.colors.LogNorm(1e-2,1e2)
        df['linecolorn'] = df['cmapn'](df['normn'](df['n0']))

        
        if as_dict:
            return df
        else:
            return pd.Series(df, name=self.basename)


        
class LoadSimFeedbackTestAll(object):
    """Class to load multiple simulations"""
    def __init__(self, models=None):

        # Default models
        if models is None:
            models = dict()

        # self.models = list(models.keys())
        self.models = []
        self.basedirs = dict()
        
        for mdl, basedir in models.items():
            if not osp.exists(basedir):
                print('[LoadSimFeedbackTestAll]: Model {0:s} doesn\'t exist: {1:s}'.format(
                    mdl,basedir))
            else:
                self.models.append(mdl)
                self.basedirs[mdl] = basedir

    def set_model(self, model, savdir=None, load_method='pyathena', verbose=False):
        
        self.model = model
        self.sim = LoadSimFeedbackTest(self.basedirs[model], savdir=savdir,
                                       load_method=load_method, verbose=verbose)
        return self.sim


def load_all_feedback_test_sn(force_override=False):

    basedir = '/tigress/jk11/FEEDBACK-TEST/FEEDBACK-TEST'
    # basedir = '/scratch/gpfs/jk11/FEEDBACK-TEST/'
    models = dict(

        SN_n001_Z0001_N128=osp.join(basedir,'SN-n0.01-Z0.001-N128'),
        SN_n01_Z0001_N128=osp.join(basedir,'SN-n0.1-Z0.001-N128'),
        SN_n1_Z0001_N128=osp.join(basedir,'SN-n1-Z0.001-N128'),
        SN_n10_Z0001_N128=osp.join(basedir,'SN-n10-Z0.001-N128'),
        SN_n100_Z0001_N128=osp.join(basedir,'SN-n100-Z0.001-N128'),

        SN_n001_Z001_N128=osp.join(basedir,'SN-n0.01-Z0.01-N128'),
        SN_n01_Z001_N128=osp.join(basedir,'SN-n0.1-Z0.01-N128'),
        SN_n1_Z001_N128=osp.join(basedir,'SN-n1-Z0.01-N128'),
        SN_n10_Z001_N128=osp.join(basedir,'SN-n10-Z0.01-N128'),
        SN_n100_Z001_N128=osp.join(basedir,'SN-n100-Z0.01-N128'),

        SN_n001_Z01_N128=osp.join(basedir,'SN-n0.01-Z0.1-N128'),
        SN_n01_Z01_N128=osp.join(basedir,'SN-n0.1-Z0.1-N128'),
        SN_n1_Z01_N128=osp.join(basedir,'SN-n1-Z0.1-N128'),
        SN_n10_Z01_N128=osp.join(basedir,'SN-n10-Z0.1-N128'),
        SN_n100_Z01_N128=osp.join(basedir,'SN-n100-Z0.1-N128'),

        SN_n001_Z1_N128=osp.join(basedir,'SN-n0.01-Z1-N128'),
        SN_n01_Z1_N128=osp.join(basedir,'SN-n0.1-Z1-N128'),
        SN_n1_Z1_N128=osp.join(basedir,'SN-n1-Z1-N128'),
        SN_n10_Z1_N128=osp.join(basedir,'SN-n10-Z1-N128'),
        SN_n100_Z1_N128=osp.join(basedir,'SN-n100-Z1-N128'),

        # SN_n001_Z1_N256=osp.join(basedir,'SN-n0.01-Z1-N256'),
        # SN_n01_Z1_N256=osp.join(basedir,'SN-n0.1-Z1-N256'),
        # SN_n1_Z1_N256=osp.join(basedir,'SN-n1-Z1-N256'),
        # SN_n10_Z1_N256=osp.join(basedir,'SN-n10-Z1-N256'),
        # SN_n100_Z1_N256=osp.join(basedir,'SN-n100-Z1-N256'),

        SN_n001_Z3_N128=osp.join(basedir,'SN-n0.01-Z3-N128'),
        SN_n01_Z3_N128=osp.join(basedir,'SN-n0.1-Z3-N128'),
        SN_n1_Z3_N128=osp.join(basedir,'SN-n1-Z3-N128'),
        SN_n10_Z3_N128=osp.join(basedir,'SN-n10-Z3-N128'),
        SN_n100_Z3_N128=osp.join(basedir,'SN-n100-Z3-N128'),

    )
    
    sa = LoadSimFeedbackTestAll(models)

    # Check if pickle exists
    savdir = osp.join('/tigress', getpass.getuser(), 'FEEDBACK-TEST/pickles')
    if not osp.exists(savdir):
        os.makedirs(savdir)
        
    fpkl = osp.join(savdir, 'feedback-test-all.p')
    if not force_override and osp.isfile(fpkl):
        r = pd.read_pickle(fpkl)
        return sa, r

    df_list = []

    print(sa.models)
    # Save key results to a single dataframe
    for mdl in sa.models:
        print(mdl, end=' ')
        s = sa.set_model(mdl, verbose=False)
        df = s.get_summary_sn(as_dict=True)
        df_list.append(pd.DataFrame(pd.Series(df, name=mdl)).T)

    # print(df)
    df = pd.concat(df_list, sort=True).sort_index(ascending=False)
    df.to_pickle(fpkl)

    return sa, df



def plt_hst_sn_diff_Z(n0=1.0):
    """Plot history

    Parameters
    ----------
    n0 : float
         initial number density of hydrogen
    """

    sa, df = load_all_feedback_test_sn(force_override=False)
    fig, axes = plt.subplots(2,3,figsize=(16,8), sharex=True, constrained_layout=True)
    axes = axes.flatten()
    for mdl in sa.models:
        d = df.loc[mdl]
        if d['n0'] != n0:
            continue
        # print(mdl,d['n0'])

        h = d['hst']
        c = d['linecolorZ']
        x = h['time']
        axes[0].loglog(x, h['Rsh'], c=c, ls='-')
        axes[1].loglog(x, h['Mhi'], c=c, ls='-')
        axes[1].loglog(x, h['Msh'], c=c, ls='--')
        axes[2].loglog(x, h['pr'], c=c, ls='-', label=r'$Z=$' + '{0:g}'.format(d['Z_gas']))
        axes[3].loglog(x, h['Ethm']+h['Ekin'], c=c, ls='-')
        axes[3].loglog(x, h['Ethm'], c=c, ls='--')
        axes[3].loglog(x, h['Ekin'], c=c, ls=':')
        axes[4].loglog(x, h['pok_bub'], c=c, ls='-')
        #axes[5].loglog(x, h['dt'], c=c, ls='-')
        axes[5].semilogx(x,h['etash'], c=c, ls='-')
        
    plt.setp(axes[3:], xlabel=r'time [Myr]')#, xlim=(1e-3,1e0))
    plt.setp(axes[0], ylabel=r'$R_{\rm snr}\,[{\rm pc}]$')#, ylim=(10,50))
    plt.setp(axes[1], ylabel=r'${\rm mass}\;[M_{\odot}]$')#, ylim=(1e1,2e4))
    plt.setp(axes[2], ylabel=r'$p_{\rm snr}\;[M_{\odot}\,{\rm km}\,{\rm s}^{-1}]$')#, ylim=(1e4,5e5))
    plt.setp(axes[3], ylabel=r'$energy\;[{\rm erg}]$')
    plt.setp(axes[4], ylabel=r'$P_{\rm bub}/k_{\rm B}\;[{\rm K}\,{\rm cm}^{-3}]$') #, ylim=(1e3,1e10))
    plt.setp(axes[5], ylabel=r'$\eta = v_{\rm s,sh}t/R_{\rm sh}$', ylim=(0, 0.5))
    plt.suptitle('N={0:d}, '.format(d['Nx']) + r'$n_0=$' + '{0:g}'.format(n0) +\
                 r'$\;{\rm cm}^{-3}$')
    
    axes[2].legend(loc=4)
    plt.savefig('/tigress/jk11/figures/NEWCOOL/FEEDBACK-TEST-SN/SN-hst-n{0:g}.png'.format(n0))
    
    return fig

def plt_hst_sn_diff_n(Z=1.0):
    """Plot history

    Parameters
    ----------
    Z: float
        metallicty
    """

    sa, df = load_all_feedback_test_sn(force_override=False)

    fig, axes = plt.subplots(2,3,figsize=(16,8), sharex=True, constrained_layout=True)
    axes = axes.flatten()

    for mdl in sa.models:
        d = df.loc[mdl]

        if d['Z_gas'] != Z:
            continue
        # print(mdl,d['Z_gas'],d['r_sf'])

        h = d['hst']
        x = h['time']/d['t_sf']
        c = d['linecolorn']
        axes[0].loglog(x, h['Rsh']/d['r_sf'], c=c, ls='-')
        axes[1].loglog(x, h['Mhi']/d['Mhi_sf'], c=c, ls='-')
        axes[1].loglog(x, h['Msh']/d['Mhi_sf'], c=c, ls='--')
        axes[2].loglog(x, h['pr']/d['pr_sf'], c=c, ls='-',label=r'$n=$'+'{0:g}'.format(d['n0']))
        axes[3].loglog(x, (h['Ethm']+h['Ekin'])/(d['Ethm_sf'] + d['Ekin_sf']), c=c, ls='-')
        axes[3].loglog(x, h['Ethm']/(d['Ethm_sf'] + d['Ekin_sf']), c=c, ls=':')
        axes[3].loglog(x, h['Ekin']/(d['Ethm_sf'] + d['Ekin_sf']), c=c, ls='--')
        # axes[3].loglog(x, h['Ethm'], c=c, ls='--')
        # axes[3].loglog(x, h['Ekin'], c=c, ls=':')
        axes[4].loglog(x, h['pok_bub']/d['pok_bub_sf'], c=c, ls='-')
        #axes[5].loglog(x, h['dt'], c=c, ls='-')
        axes[5].semilogx(x,h['etash'], c=c, ls='-')
        
    plt.setp(axes[3:], xlabel=r'time [Myr]', xlim=(5e-2,2e1))
    plt.setp(axes[0], ylabel=r'$R_{\rm snr}/r_{\rm sf}$')#, ylim=(10,50))
    plt.setp(axes[1], ylabel=r'$mass/M_{\rm h,sf}$')#, ylim=(1e1,2e4))
    plt.setp(axes[2], ylabel=r'$p_{\rm r}/p_{\rm r,sf}$')#, ylim=(1e4,5e5))
    plt.setp(axes[3], ylabel=r'$E/(E_{\rm kin,sf}+E_{\rm thm,sf})$')#, ylim=(1e4,5e5))
    plt.setp(axes[4], ylabel=r'$P_{\rm bub}/k_{\rm B}\;[{\rm K}\,{\rm cm}^{-3}]$') #, ylim=(1e3,1e10))
    #plt.setp(axes[5], ylabel=r'$dt$')
    plt.setp(axes[5], ylabel=r'$\eta$', ylim=(0, 0.5))
    plt.suptitle('N={0:d}, '.format(d['Nx']) + r'$Z=$' + '{0:g}'.format(Z))

    for ax in axes:
        ax.axvline(1.0, linestyle='-', lw=0.5, color='grey')
    axes[2].legend(loc=4)
    plt.savefig('/tigress/jk11/figures/NEWCOOL/FEEDBACK-TEST-SN/SN-hst-Z{0:g}.png'.format(Z))
    
    return fig
