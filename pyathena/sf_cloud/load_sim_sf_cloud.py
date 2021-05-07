import os
import time
import os.path as osp
import pandas as pd
import numpy as np
import matplotlib as mpl
from scipy import integrate
from scipy import interpolate
import astropy.units as au
import astropy.constants as ac

from ..util.cloud import Cloud
from ..util.split_container import split_container
from ..load_sim import LoadSim
from ..util.units import Units
from ..util.cloud import Cloud

from .hst import Hst
from .slc_prj import SliceProj
from .dust_pol import DustPol
from .fields import Fields

from .compare import Compare
from .pdf import PDF
from .virial import Virial
from .virial2 import Virial2 # simpler version
from .plt_snapshot_2panel import PltSnapshot2Panel
from .plt_snapshot_vtk2d import PltSnapshotVTK2D

from .outflow import Outflow
from .sfr import get_SFR_mean
from .starpar import StarPar
from .xray import Xray

class LoadSimSFCloud(LoadSim, Hst, StarPar, SliceProj, PDF,
                     DustPol, Virial, Virial2, Outflow, Fields, Xray,
                     PltSnapshot2Panel,PltSnapshotVTK2D):
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
        """
        Return key simulation results such as SFE, t_SF, t_dest,H2, etc.
        """
        

        par = self.par
        cl = Cloud(M=par['problem']['M_cloud'],
                   R=par['problem']['R_cloud'],
                   alpha_vir=par['problem']['alpha_vir'])

        df = dict()
        df['par'] = par

        # Read hst, virial, outflow analysies
        h = self.read_hst(force_override=True)
        df['hst'] = h
        
        # try:
        #     ho = self.read_outflow_all(force_override=False)
        #     df['hst_of'] = ho
        # except:
        #     self.logger.warning('read_outflow_all() failed!')
        #     df['hst_of'] = None

        # df['marker'] = markers[df['seed'] - 1]
        
        df['basedir'] = self.basedir
        df['domain'] = self.domain
        # Input parameters
        df['mhd'] = par['configure']['gas'] == 'mhd'

        df['iWind'] = par['feedback']['iWind']
        df['iSN'] = par['feedback']['iSN']
        df['iPhot'] = par['radps']['iPhot']
        df['iRadp'] = par['radps']['apply_force']
        
        df['Nx'] = int(par['domain1']['Nx1'])
        df['tlim'] = par['time']['tlim']
        
        df['M'] = float(par['problem']['M_cloud'])
        df['R'] = float(par['problem']['R_cloud'])
        df['Sigma'] = df['M']/(np.pi*df['R']**2)
        df['seed'] = int(np.abs(par['problem']['rseed']))
        df['alpha_vir'] = float(par['problem']['alpha_vir'])
        df['vesc'] = cl.vesc.to('km/s').value
        df['sigma1d'] = cl.sigma1d.to('km/s').value
        df['rho'] = cl.rho.cgs.value
        df['nH'] = cl.nH.cgs.value
        df['tff'] = cl.tff.to('Myr').value

        
        if df['mhd']:
            df['muB'] = float(par['problem']['muB'])
            df['B'] = (2.0*np.pi*(cl.Sigma*ac.G**0.5/df['muB']).cgs.value*au.microGauss*1e6).value
            df['vA'] = (df['B0']*1e-6)/np.sqrt(4.0*np.pi*df['rho'])/1e5
            # df['label'] = r'B{0:d}.A{1:d}.S{2:d}'.\
            #               format(int(df['muB']),int(df['alpha_vir']),int(df['seed']))
        else:
            df['muB'] = np.inf
            df['B'] = 0.0
            df['vA'] = 0.0
        #     df['label'] = r'Binf.A{0:d}.S{1:d}'.\
        #                   format(int(df['alpha_vir']),int(df['seed']))
        
        # # Simulation results
        # M_sp_final = h['M_sp'].iloc[-1]
        M_sp_final = max(h['M_sp'].values)
        df['M_sp_final'] = M_sp_final
        df['SFE'] = M_sp_final/df['M']
        df['t_final'] = h['time'].iloc[-1]
        df['tau_final'] = df['t_final']/df['tff']

        # # Outflow efficiency
        # df['eps_of'] = max(h['Mof'].values)/df['M']
        # df['eps_of_HI'] = max(h['Mof_HI'].values)/df['M']
        # df['eps_of_H2'] = max(h['Mof_H2'].values)/df['M']
        # df['eps_of_HII'] = max(h['Mof_HII'].values)/df['M']

        def get_markersize(M):
            log10M = [4., 5., 6.]
            ms = [40.0, 120.0, 360.0]
            return np.interp(np.log10(M), log10M, ms)

        def get_linewidth(M):
            logM_min = 4.0
            logM_max = 6.0
            lw_min = 1.0
            lw_max = 4.0
            return lw_min + (lw_max - lw_min)*(np.log10(M)-logM_min)/(logM_max-logM_min)

        df['norm'] = mpl.colors.LogNorm(vmin=1e1, vmax=2e3)
        df['cmap'] = mpl.cm.viridis
        df['linecolor'] = df['cmap'](df['norm'](df['Sigma']))
        df['linewidth'] = get_linewidth(df['M'])
        df['markersize'] = get_markersize(df['M'])

        if df['iRadp'] and not df['iPhot'] and not df['iWind'] and not df['iSN']:
            df['marker'] = 's'
            df['color'] = 'C0'
            df['edgecolor'] = 'none'
        elif df['iPhot'] and not df['iRadp'] and not df['iWind'] and not df['iSN']:
            df['marker'] = 'o'
            df['color'] = 'C1'
            df['edgecolor'] = 'none'
        elif df['iWind'] and not df['iPhot'] and not df['iRadp'] and not df['iSN']:
            df['marker'] = 'P'
            df['color'] = 'C2'
            df['edgecolor'] = 'none'
        elif df['iSN'] and not df['iPhot'] and not df['iRadp'] and not df['iWind']:
            df['marker'] = 'X'
            df['color'] = 'C3'
            df['edgecolor'] = 'none'

        df['kwargs_scatter'] = dict(m=df['marker'], s=df['markersize'], c=df['color'])

        idx_SF0, = h['M_sp'].to_numpy().nonzero()
        if len(idx_SF0):
            df['t_*'] = h['time'][idx_SF0[0]-1]
            df['tau_*'] = df['t_*']/df['tff']
            # Time at which XX% of star formation is complete
            df['t_95%'] = h['time'][h.M_sp > 0.95*M_sp_final].values[0]
            df['tau_95%'] = df['t_95%']/df['tff']
            df['t_90%'] = h['time'][h.M_sp > 0.90*M_sp_final].values[0]
            df['tau_90%'] = df['t_90%']/df['tff']
            df['t_80%'] = h['time'][h.M_sp > 0.80*M_sp_final].values[0]
            df['tau_80%'] = df['t_80%']/df['tff']
            df['t_50%'] = h['time'][h.M_sp > 0.50*M_sp_final].values[0]
            df['tau_50%'] = df['t_50%']/df['tff']
            df['t_SF'] = df['t_90%'] - df['t_*'] # SF duration
            df['tau_SF'] = df['t_SF']/df['tff']
            df['t_SF95'] = df['t_95%'] - df['t_*'] # SF duration
            df['tau_SF95'] = df['t_SF']/df['tff']
        #     df['t_SF2'] = M_sp_final**2 / \
        #                 integrate.trapz(h['SFR']**2, h.time)
        #     df['tau_SF2'] = df['t_SF2']/df['tff']
        #     df['SFR_mean'] = get_SFR_mean(h, 0.0, 90.0)['SFR_mean']
        #     df['SFE_3Myr'] = h.loc[h['time'] > df['t_*'] + 3.0, 'M_sp'].iloc[0]/df['M']
        #     df['t_dep'] = df['M']/df['SFR_mean'] # depletion time t_dep = M0/SFR_mean
        #     df['eps_ff'] = df['tff']/df['t_dep'] # SFE per free-fall time eps_ff = tff0/tdep
        #     # Time at which neutral gas mass < 5% of the initial cloud mass
        #     df['t_mol_5%'] = h.loc[h['MH2_cl'] < 0.05*df['M'], 'time'].iloc[0]
        #     df['t_dest_mol'] = df['t_mol_5%'] - df['t_*']
        #     try:
        #         df['t_neu_5%'] = h.loc[h['MH2_cl'] + h['MHI_cl'] < 0.05*df['M'], 'time'].iloc[0]
        #         df['t_dest_neu'] = df['t_neu_5%'] - df['t_*']
        #     except IndexError:
        #         df['t_neu_5%'] = np.nan
        #         df['t_dest_neu'] = np.nan
        #     # print('t_dep, eps_ff, t_dest_mol, t_dest_neu',
        #     #       df['t_dep'],df['eps_ff'],df['t_dest_mol'],df['t_dest_neu'])
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
            df['t_SF95'] = np.nan
            df['tau_SF95'] = np.nan
        #     df['t_SF2'] = np.nan
        #     df['tau_SF2'] = np.nan
        #     df['SFR_mean'] = np.nan
        #     df['SFE_3Myr'] = np.nan
        #     df['t_dep'] = np.nan
        #     df['eps_ff'] = np.nan
        #     df['t_mol_5%'] = np.nan
        #     df['t_dest_mol'] = np.nan
        #     df['t_neu_5%'] = np.nan
        #     df['t_dest_neu'] = np.nan
            
        # try:
        #     df['fesc_cum_PH'] = h['fesc_cum_PH'].iloc[-1] # Lyman Continuum
        #     df['fesc_cum_FUV'] = h['fesc_cum_FUV'].iloc[-1]
        #     df['fesc_cum_3Myr_PH'] = h.loc[h['time'] < df['t_*'] + 3.0,'fesc_cum_PH'].iloc[-1]
        #     df['fesc_cum_3Myr_FUV'] = h.loc[h['time'] < df['t_*'] + 3.0,'fesc_cum_FUV'].iloc[-1]
        # except KeyError:
        #     print('Error in calculating fesc_cum')
        #     df['fesc_cum_PH'] = np.nan
        #     df['fesc_cum_FUV'] = np.nan
        #     df['fesc_cum_3Myr_PH'] = np.nan
        #     df['fesc_cum_3Myr_FUV'] = np.nan

        # try:
        #     hv = df['hst_vir']
        #     f = interpolate.interp1d(hv['time'].values, hv['avir_cl_alt'])
        #     df['avir_t_*'] = f(df['t_*'])
        #     f2 = interpolate.interp1d(hv2['time'].values,
        #                               ((2.0*(hv2['T_thm_cl_all']+hv2['T_kin_cl_all']) + hv2['B_cl_all'])/\
        #                                                    hv2['W_cl_all']).values)
        #     df['avir_t_*2'] = f2(df['t_*'])
        # except (KeyError, TypeError):
        #     df['avir_t_*'] = np.nan
        #     df['avir_t_*2'] = np.nan
        #     pass

        # try:
        #     ho = df['hst_of']
        #     df['eps_ion_cl'] = (ho['totcl_HII_int'].iloc[-1] + h['MHII_cl'].iloc[-1])/df['M']
        #     df['eps_of_H2_cl'] = ho['totcl_H2_int'].iloc[-1]/df['M']
        #     df['eps_of_HI_cl'] = ho['totcl_HI_int'].iloc[-1]/df['M']
        #     df['eps_of_neu_cl'] = df['eps_of_H2_cl'] + df['eps_of_HI_cl']
        #     df['eps_of_HII_cl'] = ho['totcl_HII_int'].iloc[-1]/df['M']
        #     df['eps_of_H2_cl_z'] = ho['zcl_H2_int'].iloc[-1]/df['M']
        #     df['eps_of_HI_cl_z'] = ho['zcl_HI_int'].iloc[-1]/df['M']
        #     df['eps_of_HII_cl_z'] = ho['zcl_HII_int'].iloc[-1]/df['M']
        # except (KeyError, TypeError):
        #     df['eps_ion_cl'] = np.nan
        #     df['eps_of_H2_cl'] = np.nan
        #     df['eps_of_HI_cl'] = np.nan
        #     df['eps_of_neu_cl'] = np.nan
        #     df['eps_of_HII_cl'] = np.nan
        #     df['eps_of_H2_cl_z'] = np.nan
        #     df['eps_of_HI_cl_z'] = np.nan
        #     df['eps_of_HII_cl_z'] = np.nan
        #     pass
            
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
            idx_SF0, = h['M_sp'].to_numpy().nonzero()
            t0 = h['time_code'][idx_SF0[0] - 1]
            t_code = [t0 + dt_Myr_/u.Myr for dt_Myr_ in dt_Myr]
            # print('time of first SF [Myr]', t0*self.u.Myr)
            # print('time of first SF [code]', t0)
        elif sp_frac is not None:
            sp_frac = np.atleast_1d(sp_frac)
            h = self.read_hst()
            M_sp_final = h['M_sp'].iloc[-1]
            idx = [np.where(h['M_sp'] > sp_frac_*M_sp_final)[0][0] \
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

    @staticmethod
    def mscatter(x, y, ax=None, m=None, **kwargs):
        # https://stackoverflow.com/questions/52303660/iterating-markers-in-plots/52303895#52303895
        import matplotlib.markers as mmarkers
        if not ax:
            ax = plt.gca()
        sc = ax.scatter(x, y, **kwargs)
        if (m is not None) and (len(m)==len(x)):
            paths = []
            for marker in m:
                if isinstance(marker, mmarkers.MarkerStyle):
                    marker_obj = marker
                else:
                    marker_obj = mmarkers.MarkerStyle(marker)
                path = marker_obj.get_path().transformed(
                            marker_obj.get_transform())
                paths.append(path)
            sc.set_paths(paths)
        return sc

    @staticmethod
    def legend_feedback(ax, line_args=None, legend_args=None, alpha=1.0):

        line_args_ = dict(marker='o', markerfacecolor='r', linestyle='none',
                          markersize=np.sqrt(120.0), linewidth=0.5, alpha=alpha)
        legend_args_ = dict(loc='best', numpoints=1,
                            frameon=True, fontsize=18)

        if line_args is not None:
            line_args_.update(line_args)

        markers = ['o', 's', 'P', 'X']
        colors = ['C0', 'C1', 'C2', 'C3']
        labels = ['PH', 'RP', 'WN', 'SN']

        lines = []
        for m, c in zip(markers, colors):

            line_args_['marker'] = m
            line_args_['markerfacecolor'] = c
            line_args_['markeredgecolor'] = 'none'

            l = mpl.lines.Line2D([0], [0], **line_args_)
            lines.append(l)

        if legend_args is not None:
            legend_args_.update(legend_args)
            
        leg = ax.legend(lines, labels, **legend_args_)
        leg1 = ax.add_artist(leg)
        return leg1

    @staticmethod
    def legend_mass(ax, legend_args=None, marker='o', mew=1.0, color='k', alpha=1.0):

        legend_args_ = dict(loc='best', numpoints=1,
                            frameon=True, fontsize=18)
        if legend_args is not None:
            legend_args_.update(legend_args)

        # Create another legend
        s1 = mpl.lines.Line2D([0], [0], marker=marker, color=color, ls='none',
                              markersize=np.sqrt(120.0), markeredgecolor='k',
                              markerfacecolor='none',
                              markeredgewidth=mew, alpha=alpha)
        s2 = mpl.lines.Line2D([0], [0], marker=marker, color=color, ls='none',
                              markersize=np.sqrt(360.0), markeredgecolor='k',
                              markerfacecolor='none',
                              markeredgewidth=mew, alpha=alpha)
        # s3 = mpl.lines.Line2D([0], [0], marker=marker, color=color, ls='none',
        #                       markersize=np.sqrt(360.0), markeredgecolor='none',
        #                       markeredgewidth=mew, alpha=alpha)

        sym = [s1, s2] #, s3]

        leg = ax.legend(sym,
                        [#r'$M_{\rm 0}=10^4\,M_{\odot}$',
                         r'$M_{\rm 0}=10^5\,M_{\odot}$',
                         r'$M_{\rm 0}=10^6\,M_{\odot}$'],
                        **legend_args_)

        return leg

    
def load_all_sf_cloud(force_override=False):
    
    models = dict(
        # Hydro tests
        M1E6R60_PH_Binf_N128='/scratch/gpfs/jk11/SF-CLOUD/M1E6R60-PH-Binf-N128',
        M1E6R60_RP_Binf_N128='/scratch/gpfs/jk11/SF-CLOUD/M1E6R60-RP-Binf-N128',
        M1E6R60_SN_Binf_N128='/scratch/gpfs/jk11/SF-CLOUD/M1E6R60-SN-Binf-N128',

        M1E5R20_PH_Binf_N128='/scratch/gpfs/jk11/SF-CLOUD/M1E5R20-PH-Binf-N128',
        M1E5R20_RP_Binf_N128='/scratch/gpfs/jk11/SF-CLOUD/M1E5R20-RP-Binf-N128',
        M1E5R20_SN_Binf_N128='/scratch/gpfs/jk11/SF-CLOUD/M1E5R20-SN-Binf-N128',


        
        ## Early tests
        # ALL_N128='/perseus/scratch/gpfs/jk11/SF-CLOUD/M1E5R20.RWS.A4.B2.N128.test1',
        # ALL_N256='/perseus/scratch/gpfs/jk11/SF-CLOUD/M1E5R20.RWS.A4.B2.N256.test1',
        # ALL_N256_redV3='/tigress/jk11/SF-CLOUD/M1E5R20.RWS.A4.B2.N256.test2',
        
        ## Control model tests
        # ALL_N128_HYD='/tigress/jk11/SF-CLOUD/M1E5R20.ALL.N128.test.roe.hydro',
        # M1E5R05_PHRPSN='/tigress/jk11/SF-CLOUD/M1E5R05-PHRPSN',
        # M1E5R05_PHRPSN_B64='/perseus/scratch/gpfs/jk11/SF-CLOUD/M1E5R05-PHRPSN-B64',

        # M1E5R20_RP_Binf_N128='/scratch/gpfs/jk11/SF-CLOUD/M1E5R20-RP-Binf-N128',
        # M1E5R20_PH_Binf_N128='/scratch/gpfs/jk11/SF-CLOUD/M1E5R20-PH-Binf-N128',
        # M1E5R05_RP_Binf_N128='/scratch/gpfs/jk11/SF-CLOUD/M1E5R05-RP-Binf-N128',
        # M1E5R05_PH_Binf_N128='/scratch/gpfs/jk11/SF-CLOUD/M1E5R05-PH-Binf-N128',

        # M1E5R05_PHRPSN_Binf_N128='/scratch/gpfs/jk11/SF-CLOUD/M1E5R05-PHRPSN-Binf-N128',

        # M1E6R60_PHRPSN_Binf_N128='/scratch/gpfs/jk11/SF-CLOUD/M1E6R60-PHRPSN-Binf-N128',
        # M1E6R15_PHRPSN_Binf_N128='/scratch/gpfs/jk11/SF-CLOUD/M1E6R15-PHRPSN-Binf-N128',
        

        # PHRP_N128='/perseus/scratch/gpfs/jk11/SF-CLOUD/M1E5R20.PH.RP.A4.B2.N128.test',
        # RP_N128='/perseus/scratch/gpfs/jk11/SF-CLOUD/M1E5R20.RP.A4.B2.N128.test',
        # RPWNSN_N128_HLLD='/perseus/scratch/gpfs/jk11/SF-CLOUD/M1E5R20.PH.RP.WN.SN.A4.B2.N128.test.hlld'
    )
    
    sa = LoadSimSFCloudAll(models)

    # Check if pickle exists
    fpkl = osp.join('/tigress/jk11/SF-CLOUD/pickles/sf-cloud-all.p')
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

    df = pd.concat(df_list, sort=True).sort_index(ascending=False)
    df['markersize'] = df['markersize'].astype(float)
    df.to_pickle(fpkl)

    return sa, df
