# extract_data.py

import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as ac
import astropy.units as au

import pyathena as pa
from pyathena.classic import cc_arr
from ..load_sim import LoadSim


class ExtractData:

    @LoadSim.Decorators.check_pickle
    def read_VFF_Peters17(self, num, savdir=None, force_override=False):

        r = dict()

        ds = self.load_vtk(num, load_method='pyathena_classic')
        x1d, y1d, z1d = cc_arr(ds.domain)
        z, _, _ = np.meshgrid(z1d, y1d, x1d, indexing='ij')
        idx_z = np.abs(z) < 100.0
        tot = idx_z.sum()

        T = ds.read_all_data('temperature')
        xn = ds.read_all_data('xn')
        idx_c = (T[idx_z] <= 300.0)
        idx_wi = ((T[idx_z] > 300.0) & (T[idx_z] <= 8.0e3) & (xn[idx_z] < 0.1))
        idx_wn = ((T[idx_z] > 300.0) & (T[idx_z] <= 8.0e3) & (xn[idx_z] > 0.1))
        idx_whn = ((T[idx_z] > 8000.0) & (T[idx_z] < 5.0e5) & (xn[idx_z] > 0.1))
        idx_whi = ((T[idx_z] > 8000.0) & (T[idx_z] < 5.0e5) & (xn[idx_z] < 0.1))
        idx_h = (T[idx_z] > 5e5)

        r['time'] = ds.domain['time']
        r['f_c'] = idx_c.sum()/tot
        r['f_wi'] = idx_wi.sum()/tot
        r['f_wn'] = idx_wn.sum()/tot
        r['f_whi'] = idx_whi.sum()/tot
        r['f_whn'] = idx_whn.sum()/tot
        r['f_h'] = idx_h.sum()/tot

        return r

    @LoadSim.Decorators.check_pickle
    def read_EM_pdf(self, num, savdir=None, force_override=False):

        ds = self.load_vtk(num)
        nH = ds.get_field(field='density')
        xn = ds.get_field(field='specific_scalar[0]')
        nesq = ((1.0 - xn)*nH)**2

        z2 = 200.0

        bins = np.linspace(-8, 5, 100)
        dz = ds.domain['dx'][0]
        id0 = 0
        id1 = ds.domain['Nx'][2] // 2

        # Calculate EM integrated from z = 200pc
        id2 = id1 + int(z2/dz)

        EM0 = nesq[id0:,:,:].sum(axis=0)*dz
        EM1 = nesq[id1:,:,:].sum(axis=0)*dz
        EM2 = nesq[id2:,:,:].sum(axis=0)*dz

        h0, b0, _ = plt.hist(np.log10(EM0.flatten()), bins=bins, histtype='step', color='C0');
        h1, b1, _ = plt.hist(np.log10(EM1.flatten()), bins=bins, histtype='step', color='C1');
        h2, b2, _ = plt.hist(np.log10(EM2.flatten()), bins=bins, histtype='step', color='C2');

        return dict(EM0=EM0, EM1=EM1, EM2=EM2, bins=bins, h0=h0, h1=h1, h2=h2)

    @LoadSim.Decorators.check_pickle
    def read_phot_dust_U_pdf(self, num, z0=200.0,
                             ifreq_ion=0, savdir=None, force_override=False):

        s = self
        sigmapi = s.par['radps']['sigma_ph[0]']
        sigmad = s.par['radps']['kappa_dust[0]']*s.u.density.value
        c = ac.c.cgs.value
        # mean energy of ionizing photons
        hnu = s.par['radps']['hnu[{0:1d}]'.format(ifreq_ion)]*((1.0*au.eV).cgs.value)

        ds = s.load_vtk(num=num, load_method='pyathena_classic')

        #print(ds.domain)
        bins_nH = np.linspace(-5, 4, 61)
        bins_U = np.linspace(-6, 1, 61)
        bins_z = np.linspace(ds.domain['left_edge'][2], ds.domain['right_edge'][2], ds.domain['Nx'][2]//16 + 1)
        #print(bins_nH, bins_U, bins_z)

        nH = ds.read_all_data('density').flatten()
        Erad0 = ds.read_all_data('Erad0').flatten() # cgs unit
        xn = ds.read_all_data('xn').flatten()
        T = ds.read_all_data('temperature').flatten()
        ne = nH*(1.0 - xn)
        nHI = nH*xn

        Erad0ph = Erad0/hnu  # photon number density
        U = Erad0ph/nH       # ionization parameter
        x1d, y1d, z1d = pa.classic.cc_arr(ds.domain)
        z, _, _ = np.meshgrid(z1d, y1d, x1d, indexing='ij')
        # Warm phase indices
        w = ((T > 5050.0) & (T < 2.0e4) & (xn < 0.1))
        dvol = ds.domain['dx'].prod()*ac.pc.cgs.value**3
        zw = z.flatten()[w]
        Uw = U[w]
        nesqw = (ne**2)[w]
        nHw = nH[w]
        # Tw = T[w]

        # Local photoionization/dust absorption rate in a cell
        ph_rate = nHI[w]*c*sigmapi*Erad0ph[w]*dvol
        di_rate = nH[w]*c*sigmad*Erad0ph[w]*dvol

        # print('phrate, dirate',ph_rate.sum(),di_rate.sum())

        q = di_rate/(ph_rate + di_rate)
        qma = np.ma.masked_invalid(q)
        ma = qma.mask
        wlz = np.abs(zw) < z0
        bins = np.linspace(-5, 3, 80)

        # nH_warm PDF weighted by ph_rate or di_rate (all, at |z| < 200pc, above 200 pc)
        hdi, bedi, _ = plt.hist(np.log10(nHw[~ma]), bins=bins_nH, weights=di_rate[~ma], alpha=0.3, color='C0');
        hph, beph, _ = plt.hist(np.log10(nHw[~ma]), bins=bins_nH, weights=ph_rate[~ma], color='green', histtype='step');
        hdi_lz, bedi_lz, _ = plt.hist(np.log10(nHw[wlz]), bins=bins_nH,
                                      weights=di_rate[wlz], alpha=0.3, color='C0');
        hph_lz, beph_lz, _ = plt.hist(np.log10(nHw[wlz]), bins=bins_nH,
                                      weights=ph_rate[wlz], color='C0', histtype='step');
        hdi_hz, bedi_hz, _ = plt.hist(np.log10(nHw[~wlz]), bins=bins_nH,
                                          weights=di_rate[~wlz], alpha=0.3, color='C1');
        hph_hz, beph_hz, _ = plt.hist(np.log10(nHw[~wlz]), bins=bins_nH,
                                          weights=ph_rate[~wlz], color='C1', histtype='step');

        # Ionization parameter PDF weighted by ph_rate or di_rate (all, at |z| < 200pc, above 200 pc)
        hU, beU, _ = plt.hist(np.log10(Uw), bins=bins_U,
                              weights=nesqw, alpha=0.3, color='C0');
        hU_lz, beU_lz, _ = plt.hist(np.log10(Uw[wlz]), bins=bins_U,
                                    weights=nesqw[wlz], alpha=0.3, color='C1');
        hU_hz, beU_hz, _ = plt.hist(np.log10(Uw[~wlz]), bins=bins_U,
                                    weights=nesqw[~wlz], alpha=0.3, color='C2');

        # 2d histogram of z vs q = di/(di + ph) weighted by nesq
        hzq, bezqx, bezqy, _ = plt.hist2d(zw[~ma], np.log10(q[~ma]), weights=nesqw[~ma], bins=bins_z);
        bczqx = 0.5*(bezqx[1:] + bezqx[:-1])
        bczqy = 0.5*(bezqy[1:] + bezqy[:-1])
        qavg = np.empty_like(bczqx)

        for i in range(hzq.shape[0]):
            try:
                qavg[i] = np.average(10.0**bczqy, weights=hzq[i,:])
            except ZeroDivisionError:
                qavg[i] = np.nan

        ph_rate_zprof_all = (nHI*c*sigmapi*Erad0ph*dvol).reshape(np.flip(ds.domain['Nx']))
        di_rate_zprof_all = (nH*c*sigmad*Erad0ph*dvol).reshape(np.flip(ds.domain['Nx']))
        ph_rate_zprof = np.ma.sum(np.ma.masked_array(ph_rate_zprof_all, ((T < 5e3) | (T > 2e4))), axis=(1,2))
        di_rate_zprof = np.ma.sum(np.ma.masked_array(di_rate_zprof_all, ((T < 5e3) | (T > 2e4))), axis=(1,2))

        res = dict(bins_nH=bins_nH, bins_U=bins_U, bins_z=bins_z,
                   hdi=hdi, bedi=bedi,
                   hph=hph, beph=beph,
                   hdi_lz=hdi_lz, bedi_lz=bedi_lz,
                   hdi_hz=hdi_hz, bedi_hz=bedi_hz,
                   hph_lz=hph_lz, beph_lz=beph_lz,
                   hph_hz=hph_hz, beph_hz=beph_hz,
                   hU=hU, beU=beU,
                   hU_lz=hU_lz, beU_lz=beU_lz,
                   hU_hz=hU_hz, beU_hz=beU_hz,
                   hzq=hzq, bezqx=bezqx, bezqy=bezqy, bczqx=bczqx, bczqy=bczqy,
                   qavg=qavg,
                   z1d=z1d, ph_rate_zprof=ph_rate_zprof, di_rate_zprof=di_rate_zprof)

        return res
