import numpy as np
import astropy.units as au
import pandas as pd

from ..load_sim import LoadSim

from ..microphysics.h2 import calc_xH2eq

class H2:

    @LoadSim.Decorators.check_pickle
    def read_H2eq_all(self, nums=None, prefix='H2eq_all',
                      savdir=None, force_override=False):
        rr = dict()
        if nums is None:
            nums = self.nums

        self.logger.info('H2eq_all: {0:s} nums:'.format(self.basename), nums, end=' ')

        for i,num in enumerate(nums):
            print(num, end=' ')
            r = self.read_H2eq(num=num, savdir=savdir, force_override=False)
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
    def read_H2eq(self, num, prefix='H2eq',
                  savdir=None, force_override=False):

        par = self.par
        u = self.u
        h = self.read_hst()
        ds = self.load_vtk(num)
        dd = ds.get_field(['nH', 'xH2', 'xHII',
                           'rad_energy_density_LW_diss', 'xi_CR'])

        # Unattenuated CRIR
        xi_CR0 = h['xi_CR0'].loc[(h['time_code'] - ds.domain['time']).abs().idxmin()]
        try:
            # In case there are two rows with the same time_code
            xi_CR0 = h['xi_CR0'].loc[(h['time_code'] - ds.domain['time']).abs().idxmin()].iloc[0]
        except AttributeError:
            xi_CR0 = h['xi_CR0'].loc[(h['time_code'] - ds.domain['time']).abs().idxmin()]

        xi_CR = dd['xi_CR'].data.flatten()
        zeta_LW = (dd['rad_energy_density_LW_diss']*u.energy_density.value).data.flatten()
        nH = dd['nH'].data.flatten()
        nH2 = (dd['nH']*dd['xH2']).data.flatten()
        xH2eq_constCR = calc_xH2eq(nH, xi_CR0, k_gr=par['cooling']['kgr_H2'], zeta_LW=zeta_LW)
        xH2eq_noLW = calc_xH2eq(nH, xi_CR, k_gr=par['cooling']['kgr_H2'], zeta_LW=0.0)
        xH2eq_noLW_constCR = calc_xH2eq(nH, xi_CR0, k_gr=par['cooling']['kgr_H2'], zeta_LW=0)
        xH2eq = calc_xH2eq(nH, xi_CR, k_gr=par['cooling']['kgr_H2'], zeta_LW=zeta_LW)

        print('time:', ds.domain['time'], '  xi_CR0:', xi_CR0)
        print('xi_CR_minmax:', xi_CR.min(), xi_CR.max())
        print('(zeta_LW/zeta_LW0)_minmax:', zeta_LW.min()/5.7e-11, zeta_LW.max()/5.7e-11)

        nH_1d = np.logspace(np.log10(dd['nH'].min()),np.log10(dd['nH'].max()))
        xH2eq_1d = calc_xH2eq(nH_1d, xi_CR0, k_gr=par['cooling']['kgr_H2'], zeta_LW=0.0)

        Mconv = self.domain['dx'].prod()*(u.length.cgs.value**3*u.density.value) / \
                (1.0*au.M_sun).cgs.value

        res = dict()

        # Total gas mass
        res['Mgas'] = np.sum(nH)*Mconv
        res['MH2'] = np.sum(nH2)*Mconv
        res['MH2eq'] = np.sum(nH*xH2eq)*Mconv
        res['MH2eq_noLW'] = np.sum(nH*xH2eq_noLW)*Mconv
        res['MH2eq_constCR'] = np.sum(nH*xH2eq_constCR)*Mconv
        res['MH2eq_noLW_constCR'] = np.sum(nH*xH2eq_noLW_constCR)*Mconv
        res['xi_CR0'] = xi_CR0
        res['xi_CR_min'] = xi_CR.min()
        res['zeta_LW_max'] = zeta_LW.max()
        res['zeta_LW_min'] = zeta_LW.min()
        res['zeta_LW_avg'] = np.sum(zeta_LW*nH2)/np.sum(zeta_LW)

        res['nH_1d'] = nH_1d
        res['xH2eq_1d'] = nH_1d
        res['time_code'] = ds.domain['time']

        # Calculate 2d pdfs
        res['pdf2d'] = dict()

        k = 'xH2eq-xH2'
        hw, binex, biney = np.histogram2d(2.0*xH2eq, 2.0*(dd['xH2']).data.flatten(),
                                          bins=(np.linspace(0,1.0,51),np.linspace(0,1,101)),
                                          weights=Mconv*dd['nH'].data.flatten())
        res['pdf2d'][k] = dict()
        res['pdf2d'][k]['binex'] = binex
        res['pdf2d'][k]['biney'] = biney
        res['pdf2d'][k]['h'] = hw

        k = 'xH2eq-xe'
        hw, binex, biney = np.histogram2d(2.0*xH2eq, (dd['xHII']).data.flatten(),
                                          bins=(np.linspace(0,1,51),np.linspace(0.0,1.0,51)),
                                          weights=Mconv*dd['nH'].data.flatten())
        res['pdf2d'][k] = dict()
        res['pdf2d'][k]['binex'] = binex
        res['pdf2d'][k]['biney'] = biney
        res['pdf2d'][k]['h'] = hw

        k = 'nH-xH2'
        hw, binex, biney = np.histogram2d(nH, 2.0*(dd['xH2']).data.flatten(),
                                          bins=(np.logspace(-5,5,101),np.linspace(0,1,101)),
                                          weights=Mconv*dd['nH'].data.flatten())
        res['pdf2d'][k] = dict()
        res['pdf2d'][k]['binex'] = binex
        res['pdf2d'][k]['biney'] = biney
        res['pdf2d'][k]['h'] = hw

        k = 'nH-xH2eq'
        hw, binex, biney = np.histogram2d(nH, 2.0*xH2eq,
                                          bins=(np.logspace(-5,5,101),np.linspace(0,1,101)),
                                          weights=Mconv*dd['nH'].data.flatten())
        res['pdf2d'][k] = dict()
        res['pdf2d'][k]['binex'] = binex
        res['pdf2d'][k]['biney'] = biney
        res['pdf2d'][k]['h'] = hw


        # Calculate 1d pdf
        res['pdf1d'] = dict()
        k = 'nH-w-nH2'
        hw, bine = np.histogram(dd['nH'].data.flatten(),
                                bins=np.logspace(-5,5,101),
                                weights=(2.0*Mconv*dd['nH']*dd['xH2']).data.flatten())
        res['pdf1d'][k] = dict()
        res['pdf1d'][k]['bine'] = bine
        res['pdf1d'][k]['h'] = hw

        k = 'nH-w-nH'
        hw, bine = np.histogram(dd['nH'].data.flatten(),
                                bins=np.logspace(-5,5,101),
                                weights=(Mconv*dd['nH']).data.flatten())
        res['pdf1d'][k] = dict()
        res['pdf1d'][k]['bine'] = bine
        res['pdf1d'][k]['h'] = hw

        k = 'nH-w-nH2eq'
        hw, bine = np.histogram(dd['nH'].data.flatten(),
                                bins=np.logspace(-5,5,101),
                                weights=(2.0*Mconv*xH2eq)*(dd['nH'].data.flatten()))
        res['pdf1d'][k] = dict()
        res['pdf1d'][k]['bine'] = bine
        res['pdf1d'][k]['h'] = hw

        return res
