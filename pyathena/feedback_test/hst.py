# hst.py

import os
import numpy as np
import pandas as pd
import astropy.constants as ac
import astropy.units as au

from ..io.read_hst import read_hst
from ..load_sim import LoadSim

from scipy import integrate
from scipy.integrate import cumtrapz

class Hst:

    @LoadSim.Decorators.check_pickle_hst
    def read_hst(self, savdir=None, force_override=False):
        """Function to read hst and convert quantities to convenient units
        """

        hst = read_hst(self.files['hst'], force_override=force_override)

        par = self.par
        u = self.u
        domain = self.domain
        # volume of resolution element (code unit)
        dvol = domain['dx'].prod()
        # total volume of domain (code unit)
        vol = domain['Lx'].prod()
        vol_cgs = vol*u.cm**3
        Myr_cgs = (1.0*au.Myr).to('s').value

        iWind = par['feedback']['iWind']
        if par['configure']['new_cooling'] == 'ON':
            newcool = True
        else:
            newcool =False

        # Time in code unit
        hst['time_code'] = hst['time']
        # Time in Myr
        hst['time'] *= u.Myr
        # Time step
        hst['dt_code'] = hst['dt']
        hst['dt'] *= u.Myr
        if par['configure']['new_cooling'] == 'ON' and par['configure']['radps'] == 'ON':
            for f in ('dt_cool_min','dt_xH2_min','dt_xHII_min'):
                hst[f] *= u.Myr*vol

        # Shell formation time for a single SN in Myr
        # (Eq.7 in Kim & Ostriker 2015)
        tsf = 4.4e-2*par['problem']['n0']**-0.55
        hst['time_tsf'] = hst['time']/tsf
        hst['tsf'] = tsf

        # Shell formation time for wind goes here..

        # Total gas mass in Msun
        hst['mass'] *= vol*u.Msun

        try:
            # Mass weighted SNR position in pc
            hst['Rsh'] = hst['Rsh_den']/hst['Msh']*u.pc
            # shell mass in Msun
            hst['Msh'] *= u.Msun*vol
        except KeyError:
            pass

        if 'Vhps' in hst.columns and 'Vhf' in hst.columns:
            hst['Vh'] = hst['Vhps'] + hst['Vhf']

        if 'Ethm_hps' in hst.columns and 'Ethm_hf' in hst.columns:
            hst['Ethm_h'] = hst['Ethm_hps'] + hst['Ethm_hf']

        # try:
        #     # hot gas mass in Msun
        #     hst['Mh'] *= u.Msun*vol
        # except KeyError:
        #     try:
        #         hst['Mhot'] *= u.Msun*vol
        #     except KeyError:
        #         hst['Mhps'] *= u.Msun*vol
        #         hst['Mhf'] *= u.Msun*vol

        # try:
        #     # warm gas mass in Msun
        #     hst['Mw'] *= u.Msun*vol
        # except KeyError:
        #     hst['Mwarm'] *= u.Msun*vol

        # try:
        #     # intermediate temperature gas in Msun
        #     hst['Mi'] *= u.Msun*vol
        # except KeyError:
        #     hst['Minter'] *= u.Msun*vol

        # try:
        #     # cold gas mass in Msun
        #     hst['Mc'] *= u.Msun*vol
        # except KeyError:
        #     hst['Mcold'] *= u.Msun*vol


        cols = ['Mh', 'Mhot', 'Mhps', 'Mhf', 'Mw', 'Mwarm', 'Mi', 'Minter',
                'Mc', 'Mcold']
        for c in cols:
            try:
                hst[c] *= vol*u.Msun
            except KeyError:
                continue

        # try:
        #     # Hot and ionized
        #     hst['Mhi'] = hst['Mh'] + hst['Mi']
        # except KeyError:
        #     try:
        #         hst['Mhi'] = hst['Mhot'] + hst['Minter']
        #     except:
        #         try:
        #             hst['Mhi'] = hst['Mhf'] + hst['Mhps'] + hst['Minter']
        #         except Keyerror:
        #             pass

        # Mass of molecular, atomic, and ionized gas
        # H2, HI, HII: sum of passive scalars
        # mol, ato, ion: density of cells with H2/HI/HII fraction > 50%
        if newcool:
            cols = ['M_H2', 'M_HI', 'M_HII', 'M_mol', 'M_ato', 'M_ion']
            for c in cols:
                try:
                    hst[c] *= vol*u.Msun
                except KeyError:
                    continue

        # Total/hot gas/shell momentum in Msun*km/s
        pr_conv = vol*(u.mass*u.velocity).to('Msun km s-1').value
        hst['pr'] *= pr_conv
        try:
            hst['pr_h'] *= pr_conv
        except KeyError:
            try:
                hst['pr_hot'] *= pr_conv
            except KeyError:
                try:
                    hst['pr_hf'] *= pr_conv
                    hst['pr_hps'] *= pr_conv
                except KeyError:
                    pass

        hst['prsh'] *= pr_conv

        ########################
        # Outward radial force #
        ########################
        cols = ['Fthm']
        if par['radps']['apply_force'] == 1:
            cols += ['Frad']

        for c in cols:
            try:
                hst[c] *= vol*u.Msun*u.kms/u.Myr
                hst[c + '_int'] = integrate.cumtrapz(hst[c], hst['time'], initial=0.0)
            except KeyError:
                self.logger.warning('[read_hst]: Column {0:s} not found'.format(c))
                continue

        # energy in ergs
        E_conv = vol*(u.energy).cgs.value
        hst['Ethm'] *= E_conv
        hst['Ekin'] *= E_conv
        for ph in ('c','u','w','i','h'):
            try:
                hst['Ethm_'+ph] *= E_conv
            except:
                pass
            try:
                hst['Ekin_'+ph] *= E_conv
            except:
                pass

        # Mean cool/heat rates
        # cooling rate is in code units since Jan 27, 2022
        # commit ID: 531110941f771a247a9206690c534b977c677c4b
        if self.config_time > pd.to_datetime('2022-01-27 00:00:00 -04:00'):
            L_conv = vol*(u.energy/u.time).cgs.value
        else:
            L_conv = vol_cgs

        hst['cool_rate'] *= L_conv
        hst['heat_rate'] *= L_conv
        hst['net_cr'] *= L_conv
        hst['net_cr_cumul'] = integrate.cumtrapz(hst['net_cr'],
                                                 hst['time'], initial=0.0)*Myr_cgs
        hst['heat_rate_cumul'] = integrate.cumtrapz(hst['heat_rate'],
                                                    hst['time'], initial=0.0)*Myr_cgs
        hst['cool_rate_cumul'] = integrate.cumtrapz(hst['cool_rate'],
                                                    hst['time'], initial=0.0)*Myr_cgs

        # SNR velocity in km/s
        # hst['vsnr'] = hst['pr']/(hst['Msh'] + hst['Mi'] + hst['Mh'])
        # SNR radius
        # hst['Rsnr'] = hst['pr']/(hst['Msh'] + hst['Mi'] + hst['Mh'])

        # Dimensionless deceleration parameter
        # hst['eta'] = hst['vsnr']*hst['time_code']/hst['Rsnr']

        vol_cgs = vol*u.length.cgs.value**3
        try:
            hst['pok_bub'] = (hst['Ethm_i'] + hst['Ethm_h'])/\
                ((hst['Vi'] + hst['Vh'])*vol_cgs)/ac.k_B.cgs.value
        except:
            pass

        hst['Vbub'] *= vol
        hst['Rbub'] = (3.0*hst['Vbub']/(4.0*np.pi))**(1.0/3.0)
        hst['vrbub'] = np.gradient(hst['Rbub'], hst['time_code'])
        hst['etabub'] = hst['vrbub']*hst['time']/hst['Rbub']

        hst['vrsh'] = np.gradient(hst['Rsh'], hst['time_code'])
        hst['etash'] = hst['vrsh']*hst['time']/hst['Rsh']

        # Radiation feedback turned on
        if par['configure']['radps'] == 'ON':
            hst = self._calc_radiation(hst)
        #hst.index = hst['time_code']

        # Wind feedback turned on
        if par['feedback']['iWind'] > 0:
            hst = self._calc_wind(hst)


        self.hst = hst

        return hst

    def _calc_wind(self, hst):
        par = self.par
        u = self.u
        domain = self.domain
        # total volume of domain (code unit)
        vol = domain['Lx'].prod()
        pr_conv = vol*(u.mass*u.velocity).to('Msun km s-1').value

        hst['wind_Minj'] *= vol*u.Msun
        hst['wind_Einj'] *= vol*u.erg
        hst['wind_pinj'] *= vol*u.Msun*u.kms
        hst['wind_Mdot'] *= vol*u.Msun/u.Myr
        hst['wind_Edot'] *= vol*u.erg/u.s
        hst['wind_pdot'] *= vol*u.Msun*u.kms/u.Myr

        try:
            hst['wind_pr_c'] = hst['pr_c_swind_mixed4']*pr_conv
            hst['wind_pr_i'] = hst['pr_i_swind_mixed4']*pr_conv
            hst['wind_pr_w'] = hst['pr_w_swind_mixed4']*pr_conv
            try:
                hst['wind_pr_u'] = hst['pr_u_swind_mixed4']*pr_conv
            except: # typo
                hst['wind_pr_u'] = hst['pr_y_swind_mixed4']*pr_conv

            hst['wind_pr'] = hst['wind_pr_c'] + hst['wind_pr_i'] + \
                hst['wind_pr_w'] + hst['wind_pr_u'] + hst['wind_pr_hf'] + \
                hst['wind_pr_hps']
        except:
            pass

        hst['wind_pr_hf'] = hst['pr_hf']*pr_conv
        hst['wind_pr_hps'] = hst['pr_hps']*pr_conv

        return hst

    def _calc_radiation(self, hst):

        par = self.par
        u = self.u
        domain = self.domain
        # total volume of domain (code unit)
        vol = domain['Lx'].prod()
        from scipy.integrate import cumtrapz

        # Ionization/dissociation fronts
        hst['RIF'] = hst['RIF']/hst['RIF_vol']*u.pc
        hst['RDF'] = hst['RDF']/hst['RDF_vol']*u.pc

        # Effective radius of Stromgren sphere based on cells with Erad_PH > 0
        vol = domain['Lx'].prod()
        try:
            hst['V_Erad_PH'] *= vol*u.pc
            hst['Reff_Erad_PH'] = (3.0*hst['V_Erad_PH']/(4.0*np.pi))**(1.0/3.0)

            hst['nHII_Erad_PH'] *= vol/hst['V_Erad_PH']
            hst['nHIIsq_Erad_PH'] *= vol/hst['V_Erad_PH']
            hst['clumping_HII'] = hst['nHIIsq_Erad_PH']/hst['nHII_Erad_PH']**2
        except KeyError:
            pass

        # Total/escaping luminosity in Lsun
        ifreq = dict()
        for f in ('PH','LW','PE','PE_unatt'):
            try:
                ifreq[f] = par['radps']['ifreq_{0:s}'.format(f)]
            except KeyError:
                pass

        # total volume of domain (code unit)
        if self.config_time > pd.to_datetime('2022-05-01 00:00:00 -04:00'):
            vol = 1.0
        else:
            vol = domain['Lx'].prod()

        # Total luminosity
        hst['Ltot'] = 0.0
        for k,v in ifreq.items():
            if v >= 0:
                hst['Ltot'] += vol*hst[f'Ltot{v}'] # in code units first

        # Expected radial momentum injection
        if par['radps']['apply_force'] == 1 and par['configure']['ionrad']:
            # Radiation pressure only (in the optically-thick limit)
            hst['pr_inj_rad'] = cumtrapz(hst['Ltot']/ac.c.to(u.velocity).value,
                                        hst['time_code'], initial=0.0)
            hst['pr_inj_rad'] *= (u.mass*u.velocity).value

        # Total luminosity in Lsun
        hst['Ltot'] *= u.Lsun

        # Other luminosities
        for i in range(par['radps']['nfreq']):
            for k, v in ifreq.items():
                if i == v:
                    try:
                        hst[f'Ltot_{k}'] = hst[f'Ltot{i}']*u.Lsun*vol
                        hst[f'Lesc_{k}'] = hst[f'Lesc{i}']*u.Lsun*vol
                        hst[f'Ldust_{k}'] = hst[f'Ldust{i}']*u.Lsun*vol
                        # Photon rate
                        hnu = (par['radps'][f'hnu_{k}']*au.eV).cgs.value
                        hst[f'Qtot_{k}'] = hst[f'Ltot_{k}'].values * \
                                           (ac.L_sun.cgs.value)/hnu
                        hst[f'Qesc_{k}'] = hst[f'Lesc_{k}'].values * \
                                           (ac.L_sun.cgs.value)/hnu
                        hst[f'Qdust_{k}'] = hst[f'Ldust_{k}'].values * \
                                           (ac.L_sun.cgs.value)/hnu
                    except KeyError:
                        pass

        try:
            col = ['phot_rate_HI', 'phot_rate_H2',
                   'rec_rate_rad_HII', 'rec_rate_gr_HII']
            for c in col:
                hst[c] *= vol

            hst['phot_rate'] = hst['phot_rate_HI'] + hst['phot_rate_H2']
            hst['rec_rate_HII'] = hst['rec_rate_rad_HII'] + hst['rec_rate_gr_HII']
        except:
            pass

        return hst
