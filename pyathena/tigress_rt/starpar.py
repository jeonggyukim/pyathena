# starpar.py

import numpy as np
import pandas as pd

from ..load_sim import LoadSim
from ..util.mass_to_lum import mass_to_lum

class StarPar():

    @LoadSim.Decorators.check_pickle
    def read_starpar_all(self, prefix='starpar_all',
                         savdir=None, force_override=False):
        rr = dict()
        for i in self.nums_starpar:
            print(i, end=' ')
            r = self.read_starpar(num=i, force_override=False)
            if i == 0:
                for k in r.keys():
                    rr[k] = []

            for k in r.keys():
                try:
                    rr[k].append(r[k].value.item())
                except:
                    rr[k].append(r[k])

        rr = pd.DataFrame(rr)
        return rr
    
    @LoadSim.Decorators.check_pickle
    def read_starpar(self, num, savdir=None, force_override=False):

        sp = self.load_starpar_vtk(num)
        u = self.u
        domain = self.domain
        par = self.par
        LxLy = domain['Lx'][0]*domain['Lx'][1]
        
        try:
            agemax = par['radps']['agemax_rad']
        except KeyError:
            self.logger.warning('agemax_rad was not found and set to 40 Myr.')
            agemax = 20.0

        sp['age'] *= u.Myr
        sp['mage'] *= u.Myr
        sp['mass'] *= u.Msun
        
        # Select non-runaway starpar particles with mass-weighted age < agemax_rad
        isrc = np.logical_and(sp['mage'] < agemax,
                              sp['mass'] != 0.0)

        if np.sum(isrc) == 0:
            return None

        # Center of mass, luminosity, standard deviation of z-position
        # Summary
        r = dict()
        r['sp'] = sp
        r['time'] = sp.time
        r['nstars'] = sp.nstars
        r['isrc'] = isrc
        r['nsrc'] = np.sum(isrc)
        
        # Select only sources
        sp = sp[(sp['mage'] < agemax) & (sp['mass'] != 0.0)].copy()
        
        # Calculate luminosity of source particles
        LtoM = mass_to_lum(model='SB99')
        sp['Qi'] = LtoM.calc_Qi_SB99(sp['mass'], sp['mage'])
        sp['L_PE'] = LtoM.calc_LPE_SB99(sp['mass'], sp['mage'])
        sp['L_LW'] = LtoM.calc_LLW_SB99(sp['mass'], sp['mage'])
        sp['L_FUV'] = sp['L_PE'] + sp['L_LW']

        # Save source as separate DataFrame
        r['sp_src'] = sp
        
        r['z_max'] = np.max(sp['x3'])
        r['z_min'] = np.min(sp['x3'])
        r['z_mean_mass'] = np.average(sp['x3'], weights=sp['mass'])
        r['z_mean_Qi'] = np.average(sp['x3'], weights=sp['Qi'])
        r['z_mean_LFUV'] = np.average(sp['x3'], weights=sp['L_FUV'])
        r['stdz_mass'] = np.sqrt(np.average((sp['x3'] - r['z_mean_mass'])**2,
                                            weights=sp['mass']))
        r['stdz_Qi'] = np.sqrt(np.average((sp['x3'] - r['z_mean_Qi'])**2,
                                          weights=sp['Qi']))
        r['stdz_LFUV'] = np.sqrt(np.average((sp['x3'] - r['z_mean_LFUV'])**2,
                                            weights=sp['L_FUV']))

        r['Qi_tot'] = np.sum(sp['Qi'])
        r['L_LW_tot'] = np.sum(sp['L_LW'])
        r['L_PE_tot'] = np.sum(sp['L_PE'])
        r['L_FUV_tot'] = np.sum(sp['L_FUV'])
        
        # Ionizing photon per unit area
        r['Phi_i'] = r['Qi_tot']/LxLy
        # FUV luminosity per unit area
        r['Sigma_FUV'] = r['L_FUV_tot']/LxLy
        
        return r
