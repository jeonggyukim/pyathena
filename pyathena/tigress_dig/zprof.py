# zprof.py

import os
import os.path as osp

import xarray as xr
import numpy as np

from ..load_sim import LoadSim
from ..io.read_zprof import read_zprof_all, ReadZprofBase

class Zprof(ReadZprofBase):

    @LoadSim.Decorators.check_netcdf_zprof
    def _read_zprof(self, phase='whole', savdir=None, force_override=False):
        """Function to read zprof and convert quantities to convenient units.
        """

        ds = read_zprof_all(osp.dirname(self.files['zprof'][0]),
                            self.problem_id,
                            phase=phase, force_override=False)

        # Delete zeroth (t=0.0) snapshot.
        # Since coordinates cannot be changed,
        # we have to create a new dataset from raw data.
        if ds.time[0] == 0.0:
            da_ = ds.to_array()
            dat = np.delete(da_.data, 0, axis=-1)
            data_vars = dict()
            i = 0
            for v in ds.variables:
                if v == 'z' or v == 'time':
                    continue
                else:
                    #print(dat[i,...].shape)
                    data_vars[v] = (('z', 'time'), dat[i, ...])
                    i += 1

            ds = xr.Dataset(data_vars, coords=dict(z=ds.z, time=ds.time[1:]))

        # Divide all variables by total area Lx*Ly
        domain = self.domain
        dxdy = domain['dx'][0]*domain['dx'][1]
        Atot = domain['Lx'][0]*domain['Lx'][1]

        ds = ds/Atot

        if self.par['configure']['species_HI'] == 'ON':
            # For the moment s1 is assumed to be nH0
            ds['nH0'] = ds.s1
            # Volume filling factor of ionized gas
            if 'xn' in ds:
                ds['xi'] = ds.A - ds.xn
            else:
                ds['xi'] = ds.A - ds.xHI
            # Electron number density averaged over Atot
            ds['ne'] = ds.d - ds.s1
            # Electron number density averaged over Atot
            ds['nebar'] = ds.ne/ds.xi

        # Rename time to time_code and use physical time in Myr as dimension
        ds = ds.rename(dict(time='time_code'))
        ds = ds.assign_coords(time=ds.time_code*self.u.Myr)
        ds = ds.assign_coords(z_kpc=ds.z*self.u.kpc)
        ds = ds.swap_dims(dict(time_code='time'))

        # self._set_attrs(ds.domain)

        return ds


    def get_fields_for_zprof(self, num):
        """
        Function to read Kado-Fong+2020's ray-tracing
        post-processed vtk file as an xarray dataset

        Reads nH, T, xHII, and Erad_LyC
        """

        ds = self.load_vtk(num)
        d = dict()
        d['nH'] = ds.read_all_data('density')
        d['T'] = ds.read_all_data('temperature')
        d['xHII'] = 1.0 - ds.read_all_data('specific_scalar0')
        d['Erad_LyC'] = ds.read_all_data('rad_energy_density0')* \
                        self.u.energy_density.cgs.value

        from pyathena.classic.utils import cc_arr
        x,y,z = cc_arr(ds.domain)

        dd = xr.Dataset(
                data_vars=dict(
                    nH=(['z','y','x'], d['nH']),
                    T=(['z','y','x'], d['T']),
                    xHII=(['z','y','x'], d['xHII']),
                    Erad_LyC=(['z','y','x'], d['Erad_LyC'])),
                coords=dict(
                    x=(['x'], x),
                    y=(['y'], y),
                    z=(['z'], z),
                    time=ds.domain['time']),
                attrs=ds.domain,
            )

        return ds, dd

    @LoadSim.Decorators.check_pickle
    def read_zprof_partially_ionized(self, num, prefix='pionized',
                                     savdir=None, force_override=False):
        """
        Compute z-profile of gas binned by xHII
        """

        ds, dd = self.get_fields_for_zprof(num)
        dd['ne'] = dd['nH']*dd['xHII']
        dd['nesq'] = dd['ne']**2

        NxNy = ds.domain['Nx'][0]*ds.domain['Nx'][1]
        nbins = 11
        bins = np.linspace(0, 1, num=nbins)

        idx = []
        ne_ma = []
        nesq_ma = []
        area = []
        ne = []
        nesq = []

        idx_w1 = []
        ne_ma_w1 = []
        nesq_ma_w1 = []
        area_w1 = []
        ne_w1 = []
        nesq_w1 = []

        idx_w2 = []
        ne_ma_w2 = []
        nesq_ma_w2 = []
        area_w2 = []
        ne_w2 = []
        nesq_w2 = []

        for i in range(nbins-1):
            print(bins[i+1], end=' ')
            if i == 0:
                idx.append((bins[i] <= dd['xHII']) & (dd['xHII'] <= bins[i+1]))
                idx_w1.append((bins[i] <= dd['xHII']) & (dd['xHII'] <= bins[i+1]) &
                              (dd['T'] >= 6.0e3) & (dd['T'] < 1.5e4))
                idx_w2.append((bins[i] <= dd['xHII']) & (dd['xHII'] <= bins[i+1]) &
                              (dd['T'] >= 1.5e4) & (dd['T'] < 3.5e4))
            else:
                idx.append((bins[i] < dd['xHII']) & (dd['xHII'] <= bins[i+1]))
                idx_w1.append((bins[i] < dd['xHII']) & (dd['xHII'] <= bins[i+1]) &
                              (dd['T'] >= 6.0e3) & (dd['T'] < 1.5e4))
                idx_w2.append((bins[i] < dd['xHII']) & (dd['xHII'] <= bins[i+1]) &
                              (dd['T'] >= 1.5e4) & (dd['T'] < 3.5e4))

            ne_ma.append(np.ma.masked_invalid(dd.where(idx[i])['ne'].data))
            nesq_ma.append(np.ma.masked_invalid(dd.where(idx[i])['nesq'].data))

            ne_ma_w1.append(np.ma.masked_invalid(dd.where(idx_w1[i])['ne'].data))
            nesq_ma_w1.append(np.ma.masked_invalid(dd.where(idx_w1[i])['nesq'].data))
            ne_ma_w2.append(np.ma.masked_invalid(dd.where(idx_w2[i])['ne'].data))
            nesq_ma_w2.append(np.ma.masked_invalid(dd.where(idx_w2[i])['nesq'].data))

            area_tot = ds.domain['Lx'][0]*ds.domain['Lx'][1]
            dx = ds.domain['dx'][0]
            area.append(ne_ma[i].count(axis=(1,2))*dx**2)
            ne.append(ne_ma[i].sum(axis=(1,2)))
            nesq.append(nesq_ma[i].sum(axis=(1,2)))

            area_w1.append(ne_ma_w1[i].count(axis=(1,2))*dx**2)
            ne_w1.append(ne_ma_w1[i].sum(axis=(1,2)))
            nesq_w1.append(nesq_ma_w1[i].sum(axis=(1,2)))
            area_w2.append(ne_ma_w2[i].count(axis=(1,2))*dx**2)
            ne_w2.append(ne_ma_w2[i].sum(axis=(1,2)))
            nesq_w2.append(nesq_ma_w2[i].sum(axis=(1,2)))

        area = np.array(area)
        ne = np.array(ne) / NxNy
        nesq = np.array(nesq) / NxNy
        f_area = area / area.sum(axis=0)

        area_w1 = np.array(area_w1)
        ne_w1 = np.array(ne_w1) / NxNy
        nesq_w1 = np.array(nesq_w1) / NxNy
        f_area_w1 = area_w1 / area.sum(axis=0)

        area_w2 = np.array(area_w2)
        ne_w2 = np.array(ne_w1) / NxNy
        nesq_w2 = np.array(nesq_w1) / NxNy
        f_area_w2 = area_w2 / area.sum(axis=0)

        r = dict(area=area, area_w1=area_w1, area_w2=area_w2,
                 f_area=f_area, f_area_w1=f_area_w1, f_area_w2=f_area_w2,
                 ne=ne, ne_w1=ne_w1, ne_w2=ne_w2,
                 nesq=nesq, nesq_w1=nesq_w1, nesq_w2=nesq_w2,
                 bins=bins, nbins=nbins, NxNy=NxNy,
                 time=ds.domain['time'],
                 domain=ds.domain)

        return r

    def get_zprof_partially_ionized_all(self, nums,
                                        savdir, force_override=False):

        rr = dict()
        for i,num in enumerate(nums):
            print(num, end=' ')
            r = self.read_zprof_partially_ionized(
                num, savdir=savdir, force_override=False)
            if i == 0:
                for k in r.keys():
                    if k == 'time':
                        rr[k] = []
                    else:
                        rr[k] = []

            for k in r.keys():
                if k == 'time':
                    rr[k].append(r['time'])
                elif k == 'nbins' or k=='NxNy' or k == 'bins' or k == 'domain':
                    rr[k] = r[k]
                else:
                    rr[k].append(r[k])

        for k in rr.keys():
            if k == 'time' or k == 'nbins' or k=='NxNy' or k == 'bins' or k == 'domain':
                continue
            else:
                rr[k] = np.stack(rr[k],axis=0)

        return rr

    # def _set_attrs(self, zp):

    #     d1 = dict()
    #     d2 = dict()

    #     # Should be defined somewhere else, e.g., tigress..? # JGKIM
    #     d1['A'] = 'Area filling fraction'
    #     d2['A'] = '\int \Theta_{\rm ph} dxdy'
    #     d1['xn'] = 'Neutral fraction'
    #     d2['xn'] = '\int n_{\mathrm H^0}/n_{\mathrm H} dxdy'

    #     for v in zp.variables:
    #         if v in d1.keys() and v in d2.keys():
    #         #zp.attrs[v] = r'\texttt{{{0:s}}}:'.format(v)
    #             zp.attrs[v] = r'\text{{{0:s}}}:'.format(d1[v]) + d2[v]

    # def print_zprof_attrs(self, zp):

    #     # This function should also be defined in somewhere else # JGKIM
    #     from IPython.display import display, Math #, Latex

    #     display(Math(zp.attrs['xn']))
    #     display(Math(zp.attrs['A']))
