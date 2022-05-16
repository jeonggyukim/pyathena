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
                 muH=1.4271, verbose=False):
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
                                                 verbose=verbose)
        # Set unit and domain
        try:
            muH = self.par['problem']['muH']
        except KeyError:
            pass
        self.muH = muH
        self.u = Units(muH=muH)
        self.domain = self._get_domain_from_par(self.par)
        if self.test_newcool():
            self.test_newcool_params()

    def test_newcool(self):
        try:
            if self.par['configure']['new_cooling'] == 'ON':
                newcool = True
            else:
                newcool = False
        except KeyError:
            newcool = False
        return newcool

    def test_newcool_params(self):
        s = self
        try:
            s.iCoolH2colldiss = s.par['cooling']['iCoolH2colldiss']
        except KeyError:
            s.iCoolH2colldiss = 0

        try:
            s.iCoolH2rovib = s.par['cooling']['iCoolH2rovib']
        except KeyError:
            s.iCoolH2rovib = 0

        try:
            s.iCoolH2rovib = s.par['cooling']['iCoolH2rovib']
        except KeyError:
            s.iCoolH2rovib = 0

        try:
            s.ikgr_H2 = s.par['cooling']['ikgr_H2']
        except KeyError:
            s.ikgr_H2 = 0

        # s.config_time = pd.to_datetime(s.par['configure']['config_date'])
        # if 'PDT' in s.par['configure']['config_date']:
        #     s.config_time = s.config_time.tz_localize('US/Pacific')
        # if s.config_time < pd.to_datetime('2021-06-30 20:29:36 -04:00'):
        #     s.iCoolHIcollion = 0
        # else:
        #     s.iCoolHIcollion = 1


    def show_timeit(self):
        import matplotlib.pyplot as plt
        try:
            time = pd.read_csv(self.files['timeit'],delim_whitespace=True)

            tfields = [k.split('_')[0] for k in time.keys() if k.endswith('tot')]

            for tf in tfields:
                if tf == 'rayt': continue
                plt.plot(time['time'],time[tf].cumsum()/time['all'].cumsum(),label=tf)
            plt.legend()
        except KeyError:
            print("No timeit plot is available")



    def get_timeit_mean(self):
        try:
            time = pd.read_csv(self.files['timeit'],delim_whitespace=True)

            tfields = [k.split('_')[0] for k in time.keys() if k.endswith('tot')]

            return time[tfields].mean()
        except:
            print("No timeit file is available")

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
        df['t_sf_M'] = float(h.loc[(h['Mi']+h['Mh']).max() ==
                                (h['Mi']+h['Mh']), 'time'])
        df['t_sf_E'] = h.where(h['Ethm']+h['Ekin']-h['Ethm_u']-h['Ekin_u'] > 0.7e51).time.max()
        df['t_sf']=df['t_sf_E']
        # Radius at the time of shell formation
        df['r_sf'] = float(h.loc[h['time'] == df['t_sf'], 'Rsh'])
        # Mass of ionized and hot gas (T > 2e4K)
        df['Mhi_sf'] = float(h.loc[h['time'] == df['t_sf'], 'Mhi'])
        # Shell mass
        df['Msh_sf'] = float(h.loc[h['time'] == df['t_sf'], 'Msh'])
        # SNR mass
        df['Msnr_sf'] = df['Mhi_sf']+df['Msh_sf']
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


class LoadSimFeedbackTestAll(object):
    """Class to load multiple simulations"""
    def __init__(self, models=None, muH=None):

        # Default models
        if models is None:
            models = dict()
        if muH is None:
            muH = dict()
            for mdl in models:
                muH[mdl] = 1.4271
        self.models = []
        self.basedirs = dict()
        self.muH = dict()
        self.simdict = dict()

        for mdl, basedir in models.items():
            if not osp.exists(basedir):
                print('[LoadSimFeedbackTestAll]: Model {0:s} doesn\'t exist: {1:s}'.format(
                    mdl,basedir))
            else:
                self.models.append(mdl)
                self.basedirs[mdl] = basedir
                if mdl in muH:
                    self.muH[mdl] = muH[mdl]
                else:
                    print('[LoadSimFeedbackTestAll]: muH for {0:s} has to be set'.format(
                          mdl))

    def set_model(self, model, savdir=None, load_method='pyathena', verbose=False):
        self.model = model
        try:
            self.sim = self.simdict[model]
        except KeyError:
            self.sim = LoadSimFeedbackTest(self.basedirs[model], savdir=savdir,
                                           muH=self.muH[model],
                                           load_method=load_method, verbose=verbose)
            self.simdict[model] = self.sim

        return self.sim

    # adding two objects
    def __add__(self, o):
        for mdl in o.models:
            if not (mdl in self.models):
                self.models += [mdl]
                self.basedirs[mdl] = o.basedirs[mdl]
                self.muH[mdl] = o.muH[mdl]
                if mdl in o.simdict: self.simdict[mdl] = o.simdict[mdl]

        return self

    # get self class with only one key
    def __getitem__(self, key):
        return self.set_model(key)

    def __setitem__(self, key, value):
        if (type(value) == LoadSimFeedbackTest):
            self.models.append(key)
            self.simdict[key] = value
            self.basedirs[key] = value.basedir
            self.muH[key] = value.muH
        else:
            print("Assigment only accepts LoadSimFeedbackTest")
