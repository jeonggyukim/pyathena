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
        for num in self.nums_starpar:
            print(num, end=' ')
            r = self.read_starpar(num=num, force_override=force_override)
            if r is None:
                continue
            
            if not rr: # Create keys
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

        mtl = mass_to_lum(model='SB99')
        
        sp = self.load_starpar_vtk(num)
        if sp.empty:
            return None
        
        u = self.u
        domain = self.domain
        par = self.par
        
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
        else:
            if np.sum(isrc) != sp.nstars:
                print('Check: nsrc {0:d} != nstars {1:d}'.format(np.sum(isrc),
                                                                 sp.nstars))
            
            sp['Qi'] = mtl.calc_Qi_SB99(sp['mass'], sp['mage'])
            sp['L_LW'] = mtl.calc_LLW_SB99(sp['mass'], sp['mage'])
            sp['L_PE'] = mtl.calc_LPE_SB99(sp['mass'], sp['mage'])
            sp['L_FUV'] = sp['L_PE'] + sp['L_LW']
            
        # Center of mass, luminosity, standard deviation of z-position
        # Summary
        r = dict()
        r['sp'] = sp
        r['time'] = sp.time*u.Myr
        r['nstars'] = sp.nstars
        r['mtot'] = sp['mass'].sum()
        r['p1tot'] = (sp['mass']*sp['v1']).sum()
        r['p2tot'] = (sp['mass']*sp['v2']).sum()
        r['p3tot'] = (sp['mass']*sp['v3']).sum()
        r['prtot'] = (sp['mass']*(sp['v1']*sp['x1'] +
                                  sp['v2']*sp['x2'] +
                                  sp['v1']*sp['x3']) /
                      np.sqrt(sp['x1']**2 + sp['x2']**2 + sp['x3']**2)).sum()
        r['isrc'] = isrc
        r['nsrc'] = np.sum(isrc)
        
        r['Qitot'] = np.sum(sp['Qi'])
        r['L_LW'] = np.sum(sp['L_LW'])
        r['L_PE'] = np.sum(sp['L_PE'])
        r['L_FUV'] = np.sum(sp['L_FUV'])

        r['idx_mm'] = np.where(sp['mass'] == sp['mass'].max())[0][0]
        for i in range(1,4):
            r[f'x{i}cm'] = np.sum(sp[f'x{i}']*sp['mass'])/np.sum(sp['mass'])
            r[f'x{i}cl_PH'] = np.sum(sp[f'x{i}']*sp['Qi'])/np.sum(sp['Qi'])
            r[f'x{i}cl_FUV'] = np.sum(sp[f'x{i}']*sp['L_FUV'])/np.sum(sp['L_FUV'])
            r[f'x{i}mm'] = sp.iloc[r['idx_mm']][f'x{i}']
            
                    
        return r
