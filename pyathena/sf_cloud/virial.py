# virial.py

import matplotlib.pyplot as plt
import numpy as np
import yt
import yt.units as yu
import pandas as pd
import astropy.units as au
import astropy.constants as ac

from ..load_sim import LoadSim

# Static variable decorator
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def plt_avir_compare(models=['B8S4', 'B2S4', 'B1S4', 'B05S4']):

    from .load_sim_sf_cloud import load_all_alphabeta
    sa, r = load_all_alphabeta(force_override=False)

    #set_plt.set_plt_fancy()
    labels = dict(avir_cl_alt=r'$[2(E_{\rm thm} + E_{\rm kin}) + E_{\rm mag}]/|E_{\rm grav}|$',
                  avir_cl_noB_alt=r'$2(E_{\rm thm} + E_{\rm kin})/|E_{\rm grav}|$',
                  avir_cl_noBthm_alt=r'$2E_{\rm kin}/|E_{\rm grav}|$',
                  avir_obs_xy=r'$5\sigma_{{\rm 1d},xy}^2 R_{\rm obs}/(G M_{\rm cl})$',
                  avir_obs_z=r'$5\sigma_{{\rm 1d},z}^2 R_{\rm obs}/(G M_{\rm cl})$')

    fig, axes = plt.subplots(2, 2, figsize=(12,10), gridspec_kw=dict(hspace=0.4))
    axes = axes.flatten()
    for i, mdl in enumerate(models):
        plt.sca(axes[i])
        rr = r.loc[mdl]
        hv = rr['hst_vir']
        plt.plot(hv['time'], hv['avir_cl_alt'], c='k', label=labels['avir_cl_alt'])
        plt.plot(hv['time'], hv['avir_cl_noB_alt'], label=labels['avir_cl_noB_alt'])
        plt.plot(hv['time'], hv['avir_cl_noBthm_alt'], label=labels['avir_cl_noBthm_alt'])
        plt.plot(hv['time'], hv['avir_obs_xy'], ls='--', c='grey', label=labels['avir_obs_xy'])
        plt.plot(hv['time'], hv['avir_obs_z'], ls=':', c='grey', label=labels['avir_obs_z'])
        plt.yscale('log')
        plt.title(mdl, fontsize='medium')
        plt.axvline(rr['t_*'], lw=1)

    for ax in axes:
        ax.grid()
        #ax.set_xlim(0,10)
        ax.set_ylim(0.1,10)
        ax.set_xlabel(r'${\rm time}\;[{\rm Myr}]$')

    axes[0].legend(bbox_to_anchor=(3.05,0.65))

    return fig

class Virial:

    def get_num_max_virial(self):
        """Maximum snapshot number for which virial analysis will be performed
        """
        h = self.read_hst()
        Mstar_final = max(h['Mstar'].values)

        # time at which 90% of star formation occurred
        t_90 = h['time'][h['Mstar'] > 0.9*Mstar_final].values[0]

        return int(t_90/self.par['output1']['dt']) + 1

    @LoadSim.Decorators.check_pickle
    def read_virial_all(self, nums=None, prefix='virial_all',
                        savdir=None, force_override=False):

        rr = dict()
        if nums is None:
            nummax = self.get_num_max_virial()
            nums = range(0,nummax)
            print('Max step: ', nummax, end=' ')

        #print(int(self.par['problem']['rseed'],self.par['problem']['muB'] == 2.0))

        # if (int(np.abs(self.par['problem']['rseed'])) == 5) and \
        #    (self.par['problem']['muB'] == 2.0):
        #     return None

        for i,num in enumerate(nums):
            print(num, end=' ')
            r = self.read_virial(num=num, force_override=False)
            if i == 0:
                for k in r.keys():
                    rr[k] = []

            for k in r.keys():
                try:
                    rr[k].append(r[k].value.item())
                except:
                    rr[k].append(r[k])

        fac = 1.26    # Rcl = R50*fac, fac = 2^1/3 for a uniform sphere
        rr = pd.DataFrame(rr)

        # Kinetic energy calculated with velocity relative to the mean velocity
        rr['T_kin_neu_cl_alt'] = (0.5*rr['Mgas_neu_cl'].values*au.M_sun*(
            rr['vrms_x_neu_cl'].values**2 + rr['vrms_y_neu_cl'].values**2 + rr['vrms_z_neu_cl'].values**2)*
                                  (au.km/au.s)**2).to('erg')
        # Use velocity dispersion within R50
        rr['T_kin_neu_cl_alt_50'] = (0.5*rr['Mgas_neu_cl'].values*au.M_sun*(
            rr['vrms_x_neu_cl_50'].values**2 + rr['vrms_y_neu_cl_50'].values**2 + rr['vrms_z_neu_cl_50'].values**2)*
                                  (au.km/au.s)**2).to('erg')

        rr['avir_sph'] = (2.0*(rr['T_thm_sph'] + rr['T_kin_sph']
                        - rr['T_surf_sph']) + rr['M_sph'])/rr['W_sph']
        rr['avir_cl'] = (2.0*(rr['T_thm_neu_cl'] + rr['T_kin_neu_cl'])
                         + rr['M_neu_cl'])/rr['W_neu_cl']
        rr['avir_cl_noB'] = (2.0*(rr['T_thm_neu_cl'] + rr['T_kin_neu_cl']))/rr['W_neu_cl']
        rr['avir_cl_noBthm'] = (2.0*rr['T_kin_neu_cl'])/rr['W_neu_cl']

        rr['avir_cl_alt'] = (2.0*(rr['T_thm_neu_cl'] + rr['T_kin_neu_cl_alt'])
                         + rr['M_neu_cl'])/rr['W_neu_cl']
        rr['avir_cl_noB_alt'] = (2.0*(rr['T_thm_neu_cl'] + rr['T_kin_neu_cl_alt']))/rr['W_neu_cl']
        rr['avir_cl_noBthm_alt'] = (2.0*rr['T_kin_neu_cl_alt'])/rr['W_neu_cl']

        # rr['avir_cl_alt_50'] = (2.0*(rr['T_thm_neu_cl'] + rr['T_kin_neu_cl_alt_50'])
        #                  + rr['M_neu_cl'])/rr['W_neu_cl']
        # rr['avir_cl_noB_alt_50'] = (2.0*(rr['T_thm_neu_cl'] + rr['T_kin_neu_cl_alt_50']))/rr['W_neu_cl'


        rr['avir_obs_x'] = ((5.0*(rr['vrms_x_neu_cl'].values*au.km/au.s)**2*\
                             (fac*rr['R50'].values*au.pc))/\
                            (ac.G*rr['Mgas_neu_cl'].values*au.Msun)).to('')
        rr['avir_obs_y'] = ((5.0*(rr['vrms_y_neu_cl'].values*au.km/au.s)**2*\
                             (fac*rr['R50'].values*au.pc))/\
                            (ac.G*rr['Mgas_neu_cl'].values*au.Msun)).to('')
        rr['avir_obs_z'] = ((5.0*(rr['vrms_z_neu_cl'].values*au.km/au.s)**2*\
                             (fac*rr['R50'].values*au.pc))/\
                            (ac.G*rr['Mgas_neu_cl'].values*au.Msun)).to('')
        rr['avir_obs_xy'] = ((5.0*(0.5*(rr['vrms_x_neu_cl'].values**2 +
                                        rr['vrms_y_neu_cl'].values**2)*(au.km/au.s)**2)*\
                              (fac*rr['R50'].values*au.pc))/\
                             (ac.G*rr['Mgas_neu_cl'].values*au.Msun)).to('')
        rr['avir_obs_xyz'] = ((5.0*(1.0/3.0*(rr['vrms_x_neu_cl'].values**2 +
                                             rr['vrms_y_neu_cl'].values**2 +
                                             rr['vrms_z_neu_cl'].values**2)*(au.km/au.s)**2)*\
                               (fac*rr['R50'].values*au.pc))/\
                              (ac.G*rr['Mgas_neu_cl'].values*au.Msun)).to('')

        # alpha_vir using velocity dispersion within half mass radius
        rr['avir_obs_x_50'] = ((5.0*(rr['vrms_x_neu_cl_50'].values*au.km/au.s)**2*\
                                (fac*rr['R50'].values*au.pc))/\
                               (ac.G*rr['Mgas_neu_cl'].values*au.Msun)).to('')
        rr['avir_obs_y_50'] = ((5.0*(rr['vrms_y_neu_cl_50'].values*au.km/au.s)**2*\
                                (fac*rr['R50'].values*au.pc))/\
                               (ac.G*rr['Mgas_neu_cl'].values*au.Msun)).to('')
        rr['avir_obs_z_50'] = ((5.0*(rr['vrms_z_neu_cl_50'].values*au.km/au.s)**2*\
                                (fac*rr['R50'].values*au.pc))/\
                               (ac.G*rr['Mgas_neu_cl'].values*au.Msun)).to('')
        rr['avir_obs_xy_50'] = ((5.0*(0.5*(rr['vrms_x_neu_cl_50'].values**2 +
                                           rr['vrms_y_neu_cl_50'].values**2)*(au.km/au.s)**2)*\
                                 (fac*rr['R50'].values*au.pc))/\
                                (ac.G*rr['Mgas_neu_cl'].values*au.Msun)).to('')
        rr['avir_obs_xyz_50'] = ((5.0*(1.0/3.0*(rr['vrms_x_neu_cl_50'].values**2 +
                                                rr['vrms_y_neu_cl_50'].values**2 +
                                                rr['vrms_z_neu_cl_50'].values**2)*(au.km/au.s)**2)*\
                                  (fac*rr['R50'].values*au.pc))/\
                                 (ac.G*rr['Mgas_neu_cl'].values*au.Msun)).to('')

        rr['W_neu_cl_obs'] = (3.0*ac.G*rr['Mgas_neu_cl'].values**2*au.M_sun**2/
                              (5.0*fac*rr['R50'].values*au.pc)).to('erg')

        rr['W_neu_cl_obs_alt'] = (3.0*ac.G*rr['Mgas_neu_cl'].values**2*au.M_sun**2/
                                  (5.0*rr['R_rms_neu_cl'].values*au.pc)).to('erg')

        return rr

    @staticmethod
    def add_fields_virial(ds, mhd, x0, xCM_neu_cl):

        four_PI_inv = 1/(4.0*np.pi)

        @static_vars(x0=x0)
        def _dist(field, data):
            return np.sqrt((data["x"] - _dist.x0[0])**2 +
                           (data["y"] - _dist.x0[0])**2 +
                           (data["z"] - _dist.x0[0])**2)

        @static_vars(xCM_neu_cl=xCM_neu_cl)
        def _dist_neu_cl(field, data):
            return np.sqrt((data["x"] - _dist_neu_cl.xCM_neu_cl[0])**2 +
                           (data["y"] - _dist_neu_cl.xCM_neu_cl[0])**2 +
                           (data["z"] - _dist_neu_cl.xCM_neu_cl[0])**2)

        def _rdotPi_thm_x(field, data):
            return data['x']*data['pressure']

        def _rdotPi_thm_y(field, data):
            return data['y']*data['pressure']

        def _rdotPi_thm_z(field, data):
            return data['z']*data['pressure']

        def _rdotPi_x(field, data):
            return data['x']*(data['pressure'] + data['density']*data['velocity_x']**2) +\
                   data['y']*(data['density']*data['velocity_y']*data['velocity_x']) +\
                   data['z']*(data['density']*data['velocity_z']*data['velocity_x'])

        def _rdotPi_y(field, data):
            return data['y']*(data['pressure'] + data['density']*data['velocity_y']**2) +\
                   data['x']*(data['density']*data['velocity_x']*data['velocity_y']) +\
                   data['z']*(data['density']*data['velocity_z']*data['velocity_y'])

        def _rdotPi_z(field, data):
            return data['z']*(data['pressure'] + data['density']*data['velocity_z']**2) +\
                   data['x']*(data['density']*data['velocity_x']*data['velocity_z']) +\
                   data['y']*(data['density']*data['velocity_y']*data['velocity_z'])

        if mhd:
            def _rdotTM_x(field, data):
                return four_PI_inv*(
                    data['x']*0.5*(data['magnetic_field_x']**2 -
                                   data['magnetic_field_y']**2 -
                                   data['magnetic_field_z']**2) +\
                    data['y']*(data['magnetic_field_x']*data['magnetic_field_y']) +\
                    data['z']*(data['magnetic_field_x']*data['magnetic_field_z']))

            def _rdotTM_y(field, data):
                return four_PI_inv*(
                    data['y']*0.5*(data['magnetic_field_y']**2 -
                                   data['magnetic_field_z']**2 -
                                   data['magnetic_field_x']**2) +\
                    data['x']*(data['magnetic_field_x']*data['magnetic_field_y']) +\
                    data['z']*(data['magnetic_field_y']*data['magnetic_field_z']))

            def _rdotTM_z(field, data):
                return four_PI_inv*(
                    data['z']*0.5*(data['magnetic_field_z']**2 -
                                   data['magnetic_field_x']**2 -
                                   data['magnetic_field_y']**2) +\
                    data['x']*(data['magnetic_field_x']*data['magnetic_field_z']) +\
                    data['y']*(data['magnetic_field_y']*data['magnetic_field_z']))

        def _rhovrsq_x(field, data):
            return data['density']*data['velocity_x']*data['dist']**2

        def _rhovrsq_y(field, data):
            return data['density']*data['velocity_y']*data['dist']**2

        def _rhovrsq_z(field, data):
            return data['density']*data['velocity_z']*data['dist']**2

        def _rho_rsq(field, data):
            return data['density']*data['dist']**2


        ds.add_field("dist", function=_dist, units="pc", sampling_type='cell',
                     force_override=True)
        ds.add_field("dist_neu_cl", function=_dist_neu_cl, units="pc",
                     sampling_type='cell', force_override=True)

        ds.add_field("rdotPi_thm_x", function=_rdotPi_thm_x, units="erg/cm**2",
                     sampling_type='cell', force_override=True)
        ds.add_field("rdotPi_thm_y", function=_rdotPi_thm_y, units="erg/cm**2",
                     sampling_type='cell', force_override=True)
        ds.add_field("rdotPi_thm_z", function=_rdotPi_thm_z, units="erg/cm**2",
                     sampling_type='cell', force_override=True)
        ds.add_field("rdotPi_x", function=_rdotPi_x, units="erg/cm**2",
                     sampling_type='cell', force_override=True)
        ds.add_field("rdotPi_y", function=_rdotPi_y, units="erg/cm**2",
                     sampling_type='cell', force_override=True)
        ds.add_field("rdotPi_z", function=_rdotPi_z, units="erg/cm**2",
                     sampling_type='cell', force_override=True)

        if mhd:
            ds.add_field("rdotTM_x", function=_rdotTM_x, units="erg/cm**2",
                         sampling_type='cell', force_override=True)
            ds.add_field("rdotTM_y", function=_rdotTM_y, units="erg/cm**2",
                         sampling_type='cell', force_override=True)
            ds.add_field("rdotTM_z", function=_rdotTM_z, units="erg/cm**2",
                         sampling_type='cell', force_override=True)

        ds.add_field("rhovrsq_x", function=_rhovrsq_x, units="g/s",
                     sampling_type='cell', force_override=True)
        ds.add_field("rhovrsq_y", function=_rhovrsq_y, units="g/s",
                     sampling_type='cell', force_override=True)
        ds.add_field("rhovrsq_z", function=_rhovrsq_z, units="g/s",
                     sampling_type='cell', force_override=True)
        ds.add_field("rho_rsq", function=_rho_rsq, units="g*cm**-1",
                     sampling_type='cell', force_override=True)
        ds.add_gradient_fields(("athena","gravitational_potential"))

        return ds

    @LoadSim.Decorators.check_pickle
    def read_virial(self, num, Rsph_over_Rcl=1.95,
                    prefix='virial', savdir=None, force_override=False):
        """
        Function to calculate volume integral of various thermal/magnetic/gravitational
        energy terms in virial theorem
        Also calculates center of mass, half mass radius of neutral gas etc.
        """

        # Print no log messages
        from yt.funcs import mylog
        mylog.setLevel(50)

        ds = self.load_vtk(num, load_method='yt')
        da = ds.all_data()

        # Set threshold
        xCL = 'specific_scalar_CL'
        xCL_min = 1e-2     # Cloud gas if xCL > xCL_min
        xneu_min = 0.5     # Neutral if xHI + 2xH2 > xneu_min
        xH2_min = 0.25
        x0 = ds.domain_center

        # Volume of a cell in cm**3
        dV = ((ds.domain_width/ds.domain_dimensions).prod()).to('pc**3')

        # Find indices for cloud and neutral portion of it
        idx_neu = (da['xHI'] + 2.0*da['xH2'] > xneu_min)
        idx_ion = (da['xHI'] + 2.0*da['xH2'] <= xneu_min)
        idx_cl = da[xCL] > xCL_min
        idx_neu_cl = (da['xHI'] + 2.0*da['xH2'] > xneu_min) & (da[xCL] > xCL_min)
        idx_ion_cl = (da['xHI'] + 2.0*da['xH2'] <= xneu_min) & (da[xCL] > xCL_min)
        idx_H2_cl = (2.0*da['xH2'] > xH2_min) & (da[xCL] > xCL_min)

        # Calculate initial magnetic field
        if self.par['configure']['gas'] == 'hydro':
            mhd = False
        else:
            mhd = True

        if mhd:
            muB = self.par['problem']['muB']
        else:
            muB = np.inf
        M0 = self.par['problem']['M_cloud']
        R0 = self.par['problem']['R_cloud']
        B0mag = 1.3488004135072468e-05*(M0*1e-5)*(20.0/R0)**2*(2.0/muB)*yu.gauss
        thetaB = self.par['problem']['theta_B0']*np.pi/180.0
        phiB = self.par['problem']['phi_B0']*np.pi/180.0
        B0 = B0mag*yt.YTArray([np.sin(thetaB)*np.cos(phiB),
                               np.sin(thetaB)*np.sin(phiB),
                               np.cos(thetaB)])
        print('B0',B0)

        # Save results to a dictionary
        r = dict()

        # r['model'] = self.basename
        r['Rsph_over_Rcl'] = Rsph_over_Rcl

        r['time_code'] = ds.current_time.item()
        r['time'] = ds.current_time.to('Myr')

        # Mass
        r['Mgas_tot'] = (da['density'].sum()*dV).to('Msun')
        r['Mgas_neu'] = (da['density'][idx_neu].sum()*dV).to('Msun')
        r['Mgas_ion'] = (da['density'][idx_ion].sum()*dV).to('Msun')
        r['Mgas_cl'] = (da['density'][idx_cl].sum()*dV).to('Msun')
        r['Mgas_neu_cl'] = (da['density'][idx_neu_cl].sum()*dV).to('Msun')
        r['Mgas_ion_cl'] = (da['density'][idx_ion_cl].sum()*dV).to('Msun')
        r['Mgas_H2_cl'] = (da['density'][idx_H2_cl].sum()*dV).to('Msun')

        # Volume
        r['V_tot'] = ds.domain_width.prod().to('pc**3')
        r['V_neu'] = dV*idx_neu.sum()
        r['V_ion'] = dV*idx_ion.sum()
        r['V_cl'] = dV*idx_cl.sum()
        r['V_neu_cl'] = dV*idx_neu_cl.sum()
        r['V_ion_cl'] = dV*idx_ion_cl.sum()
        r['V_H2_cl'] = dV*idx_H2_cl.sum()

        # Center of mass
        r['xCM'] = yt.YTArray([(da[ax]*da['density']).sum()/da['density'].sum() \
                               for ax in ('x','y','z')]).to('pc')
        r['xCM_cl'] = yt.YTArray([(da[ax][idx_cl]*da['density'][idx_cl]).sum()/\
                                  da['density'][idx_cl].sum() \
                                  for ax in ('x','y','z')]).to('pc')
        r['xCM_neu_cl'] = yt.YTArray([(da[ax][idx_neu_cl]*da['density'][idx_neu_cl]).sum()/\
                                      da['density'][idx_neu_cl].sum() \
                                      for ax in ('x','y','z')]).to('pc')
        r['xCM_ion_cl'] = yt.YTArray([(da[ax][idx_ion_cl]*da['density'][idx_ion_cl]).sum()/\
                                      da['density'][idx_ion_cl].sum() \
                                      for ax in ('x','y','z')]).to('pc')

        # Add fields
        ds = self.add_fields_virial(ds, mhd, x0=x0, xCM_neu_cl=r['xCM_neu_cl'])

        # Calculate radius that encloses xx % of neutral cloud mass
        # (distance is measured from xCM_neu_cl)
        # and corresponding free-fall time
        M_encl_tot = r['Mgas_neu_cl']

        dist_neu_cl = da['dist_neu_cl'][idx_neu_cl]
        idx_srt = np.argsort(dist_neu_cl)
        dist_neu_cl_srt = dist_neu_cl[idx_srt]
        M_encl = (da['density'][idx_neu_cl][idx_srt].cumsum()*dV).to('Msun')
        percentage = [50, 67, 90, 95]

        for p in percentage:
            idx = np.where(M_encl/M_encl_tot > p*1e-2)[0]
            r['R{0:d}'.format(p)] = dist_neu_cl_srt[idx[0]]
            r['rho{0:d}'.format(p)] = p*1e-2*M_encl_tot/\
                        (4.0*np.pi*r['R{0:d}'.format(p)]**3/3.0)
            r['tff{0:d}'.format(p)] = np.sqrt(3.0*np.pi/\
                        (32.0*yu.G*r['rho{0:d}'.format(p)])).to('Myr')

        r['Ekin_neu_cl'] = (0.5*da['density'][idx_neu_cl]* \
                            da['velocity_magnitude'][idx_neu_cl]**2*dV).sum().to('erg')
        r['Ekin_ion_cl'] = (0.5*da['density'][idx_ion_cl]* \
                            da['velocity_magnitude'][idx_ion_cl]**2*dV).sum().to('erg')
        r['vdisp_neu_cl'] = (np.sqrt(2.0*r['Ekin_neu_cl']/r['Mgas_neu_cl'])).to('km/s')
        r['vdisp_ion_cl'] = (np.sqrt(2.0*r['Ekin_ion_cl']/r['Mgas_ion_cl'])).to('km/s')

        # Velocity dispersion within spheres of radii R50, R67, R90, R95
        for p in percentage:
            sph_p = ds.sphere(r['xCM_neu_cl'],
                              (r['R{0:d}'.format(p)].value.item(), "pc"))
            idx_p = (sph_p['xHI'] + 2.0*sph_p['xH2'] > xneu_min) & (sph_p[xCL] > xCL_min)

            for ax in ('x','y','z'):
                r[f'vmean_{ax}_neu_cl_{p}'] = \
                    ((sph_p['density']*sph_p[f'velocity_{ax}'])[idx_p].sum()/
                     (sph_p['density'][idx_p]).sum()).to('km/s')
                r[f'vrms_{ax}_neu_cl_{p}'] = \
                    (sph_p['density'][idx_p]*((sph_p[f'velocity_{ax}'])[idx_p] -
                        r[f'vmean_{ax}_neu_cl_{p}'])**2).sum()/\
                        (sph_p['density'][idx_p].sum())
                r[f'vrms_{ax}_neu_cl_{p}'] = np.sqrt(r[f'vrms_{ax}_neu_cl_{p}']).to('km/s')

        for ax in ('x','y','z'):
            r[f'vmean_{ax}_neu_cl'] = ((da['density']*da[f'velocity_{ax}'])[idx_neu_cl].sum()/
                                       (da['density'][idx_neu_cl]).sum()).to('km/s')
            r[f'vrms_{ax}_neu_cl'] =\
                (da['density'][idx_neu_cl]*((da[f'velocity_{ax}'])[idx_neu_cl] -
                                            r[f'vmean_{ax}_neu_cl'])**2).sum()/\
                (da['density'][idx_neu_cl].sum())
            r[f'vrms_{ax}_neu_cl'] = np.sqrt(r[f'vrms_{ax}_neu_cl']).to('km/s')

        Rcl = self.par['problem']['R_cloud']
        sph_ = ds.sphere(x0, (1.99*Rcl, "pc"))
        surf = ds.surface(sph_, "dist", (Rsph_over_Rcl*Rcl, "pc"))
        sph = ds.sphere(x0, (Rsph_over_Rcl*Rcl, "pc"))

        # These are the coordinates of all triangle vertices in the surface
        triangles = surf.triangles

        # construct triange normal vectors
        w1 = surf.triangles[:,1] - surf.triangles[:,0]
        w2 = surf.triangles[:,1] - surf.triangles[:,2]

        # calculate triangle areas
        vector_area = np.cross(w1, w2)/2*(yu.pc**2).to('cm**2')
        scalar_area = np.linalg.norm(vector_area, axis=1)
        surf_area = scalar_area.sum()

        # idx for neutral cloud within sphere
        idx = (sph['xHI'] + 2.0*sph['xH2'] > xneu_min) & (sph[xCL] > xCL_min)

        # Surface integral
        if mhd:
            rdotTM = np.column_stack([surf['rdotTM_{0:s}'.format(x)] \
                                      for x in 'xyz'])*(yu.erg/yu.cm**2)
        rdotPi = np.column_stack([surf['rdotPi_{0:s}'.format(x)] \
                                  for x in 'xyz'])*(yu.erg/yu.cm**2)
        rdotPi_thm = np.column_stack([surf['rdotPi_thm_{0:s}'.format(x)] \
                                      for x in 'xyz'])*(yu.erg/yu.cm**2)
        rhovrsq = np.column_stack([surf['rhovrsq_{0:s}'.format(x)] \
                                   for x in 'xyz'])*(yu.g/yu.s)

        r['Mgas_sph'] = ((sph['cell_volume']*sph['density']).sum()).to('Msun')
        r['Mgas_neu_cl_sph'] = ((sph['cell_volume'][idx]*sph['density'][idx]).sum()).to('Msun')
        r['I_E'] = ((sph['cell_volume']*sph['rho_rsq']).sum()).to('g*cm**2')
        r['I_E_neu_cl'] = ((sph['cell_volume'][idx]*sph['rho_rsq'][idx]).sum()).to('g*cm**2')
        r['V_neu_cl_sph'] = ((sph['cell_volume'][idx]).sum()).to('pc**3')

        # Effective radius # Incorrect
        r['R_rms'] = np.sqrt(5.0/3.0*r['I_E']/r['Mgas_sph']).to('pc')
        r['R_rms_neu_cl'] = np.sqrt(5.0/3.0*r['I_E_neu_cl']/r['Mgas_neu_cl_sph']).to('pc')

        # Flux of momentum of inertia
        r['S_surf_sph'] = 0.5*np.sum(vector_area*rhovrsq)

        # Surface integral of Reynolds stress
        r['T_surf_sph'] = 0.5*np.sum(vector_area*rdotPi)
        # Surface integral of Reynolds stress (only themal pressure term)
        r['T_surf_thm_sph'] = 0.5*np.sum(vector_area*rdotPi_thm)

        # Thermal + kinetic energy (for sphere)
        r['T_thm_sph'] = 1.5*((sph['cell_volume']*sph['pressure']).sum()).to('erg')
        r['T_kin_sph'] = 0.5*((sph['cell_volume']*sph['density']*
                               sph['velocity_magnitude']**2).sum()).to('erg')
        r['T_thm_neu_cl_sph'] = 1.5*((sph['cell_volume'][idx]*sph['pressure'][idx]).sum()).to('erg')
        r['T_kin_neu_cl_sph'] = 0.5*((sph['cell_volume'][idx]*sph['density'][idx]*
                                      sph['velocity_magnitude'][idx]**2).sum()).to('erg')
        # Thermal + kinetic energy (for entire volume)
        r['T_thm'] = 1.5*((da['cell_volume']*da['pressure']).sum()).to('erg')
        r['T_kin'] = 0.5*((da['cell_volume']*da['density']*
                               da['velocity_magnitude']**2).sum()).to('erg')
        r['T_thm_neu_cl'] = 1.5*((da['cell_volume'][idx_neu_cl]*da['pressure'][idx_neu_cl]).sum()).to('erg')
        r['T_kin_neu_cl'] = 0.5*((da['cell_volume'][idx_neu_cl]*da['density'][idx_neu_cl]*
                                      da['velocity_magnitude'][idx_neu_cl]**2).sum()).to('erg')

        # Surface integral of Maxwell stress
        if mhd:
            r['M_surf_sph'] = np.sum(vector_area*rdotTM)
            # Alternative way
            # r['M_surf_'] = surf.calculate_flux('rdotTM_x', 'rdotTM_y', 'rdotTM_z', fluxing_field="ones")
            # Volume integral of magnetic energy density
            r['M_vol_sph'] = (sph['cell_volume']*sph['magnetic_energy']).sum().to('erg')
            r['M_sph'] = r['M_vol_sph'] + r['M_surf_sph']
            # Define average magnetic field on surface
            r['B_surf_avg_x'] = (surf['magnetic_field_x']*scalar_area).sum()/surf_area
            r['B_surf_avg_y'] = (surf['magnetic_field_y']*scalar_area).sum()/surf_area
            r['B_surf_avg_z'] = (surf['magnetic_field_z']*scalar_area).sum()/surf_area
            r['B_surf_avg_mag'] = np.sqrt(r['B_surf_avg_x']**2 +
                                           r['B_surf_avg_y']**2 +
                                           r['B_surf_avg_z']**2)
            # Alternative method to calculate surface-averaged B-field magnitude
            # Bmag^2 = -6*M_surf/R_sph^3
            r['B_surf_avg_mag_alt'] = (np.sqrt(-6.0*r['M_surf_sph']/
                                               ((r['Rsph_over_Rcl']*Rcl*yu.pc)**3))).to('gauss')

            # Magnetic energy as obtaind by volume integral of B^2 - B_avg^2
            # Using B_avg=B0. This is incorrect because B_avg changes with time
            r['M_neu_cl0_sph'] = (((sph['magnetic_field_magnitude'][idx]**2 - B0mag**2)*
                                   sph['cell_volume'][idx]).sum()).to('erg') / (8.0*np.pi)
            # This will be our fiducial choice
            r['M_neu_cl_sph'] = (((sph['magnetic_field_magnitude'][idx]**2 - r['B_surf_avg_mag']**2)*
                                   sph['cell_volume'][idx]).sum()).to('erg') / (8.0*np.pi)
            # Alternative
            r['M_neu_cl_sph_alt'] = (((sph['magnetic_field_magnitude'][idx]**2 - r['B_surf_avg_mag_alt']**2)*
                                   sph['cell_volume'][idx]).sum()).to('erg') / (8.0*np.pi)

            r['M_neu_cl0'] = (((da['magnetic_field_magnitude'][idx_neu_cl]**2 - B0mag**2)*
                                   da['cell_volume'][idx_neu_cl]).sum()).to('erg') / (8.0*np.pi)
            r['M_neu_cl'] = (((da['magnetic_field_magnitude'][idx_neu_cl]**2 - r['B_surf_avg_mag']**2)*
                                   da['cell_volume'][idx_neu_cl]).sum()).to('erg') / (8.0*np.pi)
            r['M_neu_cl_alt'] = (((da['magnetic_field_magnitude'][idx_neu_cl]**2 - r['B_surf_avg_mag_alt']**2)*
                                  da['cell_volume'][idx_neu_cl]).sum()).to('erg') / (8.0*np.pi)

        else:
            r['M_surf_sph'] = 0.0
            r['M_vol_sph'] = 0.0
            r['M_sph'] = 0.0
            r['B_surf_avg_x'] = 0.0
            r['B_surf_avg_y'] = 0.0
            r['B_surf_avg_z'] = 0.0
            r['B_surf_avg_mag'] = 0.0
            r['B_surf_avg_mag_alt'] = 0.0
            r['M_neu_cl0_sph'] = 0.0
            r['M_neu_cl_sph'] = 0.0
            r['M_neu_cl_sph_alt'] = 0.0
            r['M_neu_cl0'] = 0.0
            r['M_neu_cl'] = 0.0
            r['M_neu_cl_alt'] = 0.0

        # Gravitational term (gravitational energy in the absence of external potential)
        r['W_sph'] = (sph['density']*\
                      (sph['x']*sph['gravitational_potential_gradient_x'] +\
                       sph['y']*sph['gravitational_potential_gradient_y'] +\
                       sph['z']*sph['gravitational_potential_gradient_z'])*
                      sph['cell_volume']).sum().to('erg')

        r['W_neu_cl_sph'] = (sph['density'][idx]*\
                             (sph['x'][idx]*sph['gravitational_potential_gradient_x'][idx] +\
                              sph['y'][idx]*sph['gravitational_potential_gradient_y'][idx] +\
                              sph['z'][idx]*sph['gravitational_potential_gradient_z'][idx])*
                             sph['cell_volume'][idx]).sum().to('erg')

        r['W'] = (da['density']*\
                      (da['x']*da['gravitational_potential_gradient_x'] +\
                       da['y']*da['gravitational_potential_gradient_y'] +\
                       da['z']*da['gravitational_potential_gradient_z'])*
                      da['cell_volume']).sum().to('erg')

        r['W_neu_cl'] = (da['density'][idx_neu_cl]*\
                         (da['x'][idx_neu_cl]*da['gravitational_potential_gradient_x'][idx_neu_cl] +\
                          da['y'][idx_neu_cl]*da['gravitational_potential_gradient_y'][idx_neu_cl] +\
                          da['z'][idx_neu_cl]*da['gravitational_potential_gradient_z'][idx_neu_cl])*
                         da['cell_volume'][idx_neu_cl]).sum().to('erg')

        return r
