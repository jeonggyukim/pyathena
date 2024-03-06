import copy
import pickle
import os.path as osp
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from functools import reduce
import astropy.units as au
import astropy.constants as ac

from ..classic.utils import cc_arr
from ..classic.vtk_reader import AthenaDataSet

from ..load_sim import LoadSim
from ..decorators import check_netcdf_zprof_vtk

class ZprofFromVTK:

    @check_netcdf_zprof_vtk
    def read_zprof_from_vtk(self, num,
                            fields=['nH','ne','nesq','xHII','xe',
                                    'T','Uion','Erad_LyC','Erad_PE','Erad_LW'],
                            weights=['nH','ne','nesq'], phase_set_name='ncrrad',
                            prefix='zprof_vtk', savdir=None, force_override=False):

        if 'LyC_ma' in phase_set_name:
            if not 'Erad_LyC_mask' in fields:
                fields.append('Erad_LyC_mask')
            if not 'Uion' in fields:
                fields.append('Uion')
            if not 'Uion_pi' in fields:
                fields.append('Uion_pi')

        if 'eq' in phase_set_name:
            if not 'xHII_eq' in fields:
                fields.append('xHII_eq')

        ds = self.load_vtk(num)
        dd = ds.get_field(fields)
        v = fields[0]
        # Set phase id and fraction
        dd = dd.assign(ph_mask=(('z','y','x'), np.zeros_like(dd[v], dtype=int)))
        dd = dd.assign(frac=(('z','y','x'), np.ones_like(dd[v])))

        phs = self.phase_set[phase_set_name]
        for ph in phs.phases:
            all_cond = []
            for cond_ in ph.cond:
                all_cond.append((cond_[0])(dd, *(cond_[1:])))

            dd['ph_mask'] += ph.mask*reduce(np.logical_and, all_cond, 1)

        zp = dict()
        NxNy_inv = 1.0/dd['frac'].sum(dim=('x','y'))
        # z-profile of all gas
        zp['whole'] = dd.sum(dim=('x','y'), keep_attrs=True) * NxNy_inv
        for ph in phs.phases:
            idx = (dd['ph_mask'] == ph.mask)
            zp[ph.name] = dd.where(idx).sum(dim=('x','y'), keep_attrs=True)
            zp[ph.name] *= NxNy_inv

        if weights is not None:
            for w in np.atleast_1d(weights):
                zpw = dict()
                dw = (dd*dd[w])
                denom_inv = 1.0/dw['frac'].sum(dim=('x','y'))
                rename_dict = {v: v + '_w_' + w for v in list(dd.keys())}
                dw = dw.rename(rename_dict)
                zpw['whole'] = dw.sum(dim=('x','y'), keep_attrs=True) * denom_inv
                for ph in phs.phases:
                    idx = (dd['ph_mask'] == ph.mask)
                    zpw[ph.name] = dw.where(idx).sum(dim=('x','y'), keep_attrs=True)
                    zpw[ph.name] *= denom_inv

                for name in phs.phase_names:
                    zp[name] = xr.merge((zp[name], zpw[name]))

                del dw

        # Merge and create phase as new dimension
        zplist = []
        all_phase_names = ['whole'] + list(phs.phase_names)
        for name in all_phase_names:
            zplist.append(zp[name].expand_dims('phase').assign_coords(phase=[name]))

        zpa = xr.concat(zplist, dim='phase')
        zpa = zpa.to_array().to_dataset('phase')
        zpa = zpa.to_array('phase').to_dataset('variable')
        zpa = zpa.drop_vars('ph_mask')

        def _check_frac(zp):
            f = 0.0
            for ph in zp.phase.data:
                if ph != 'whole':
                    zp_ = zpa.sel(phase=ph)
                    f += zp_['frac']

            return np.allclose(f.data, 1.0, rtol=NxNy_inv, atol=NxNy_inv)

        if not _check_frac(zpa):
            self.logger.warning(
                    "Volume fractions of all phases don't add up to 1")

        zpa = zpa.assign_attrs(time=dd.attrs['time'])

        return zpa

    @check_netcdf_zprof_vtk
    def read_zprof_from_vtk_all(self, nums=None, phase_set_name='ncrrad',
                                prefix='zprof_vtk_all',
                                force_override=False,
                                read_zprof_from_vtk_kwargs=None):
        if read_zprof_from_vtk_kwargs is None:
            read_zprof_from_vtk_kwargs = dict(phase_set_name=phase_set_name,
                                              force_override=False)
        else:
            if 'phase_set_name' in read_zprof_from_vtk_kwargs:
                if phase_set_name != read_zprof_from_vtk_kwargs['phase_set_name']:
                    raise ValueError(
                        "phase_set_name in keyword argements different!")
            else:
                read_zprof_from_vtk_kwargs['phase_set_name'] = phase_set_name

        zplist = []
        for num in nums:
            print(num, end=' ')
            zpa = self.read_zprof_from_vtk(num, **read_zprof_from_vtk_kwargs)
            zplist.append(zpa.expand_dims(time=np.atleast_1d(zpa.attrs['time'])))

        self.logger.info(
            '[read_zprof_vtk] Concatenating {0:d} xarray datasets.'.format(len(zplist)))

        # Use concat.
        # Merge is much slower and uses more memory leading to crash.
        return xr.concat(zplist, dim='time')

    def merge_zprof_with_hst(self, zpa):
        h = self.read_hst_rad()
        columns = ['fesc_LyC', 'fesc_FUV', 'Ltot0', 'Ltot1', 'sfr10']
        for c in columns:
            f = interp1d(h['time'], h[c], fill_value='extrapolate', bounds_error=False)
            assign_kwargs = {c:(['time'], f(zpa['time']))}
            zpa = zpa.assign(**assign_kwargs) #fesc_LyC=(['time'], f(zpa['time'])))

        return zpa, h



def get_zprof_classic(basedir='/projects/EOSTRIKE/TIGRESS-classic/R8_8pc_newacc',
                      savdir='/tigress/jk11/NCR-RAD/TIGRESS-classic',
                      force_override=False):

    s = LoadSim(basedir)
    fname = osp.join(savdir, '{0:s}-zprof-Twarm.p'.format(s.basename))
    if not force_override and osp.exists(fname):
        print('Read from pickle {0:s}'.format(fname))
        r = pickle.load(open(fname, 'rb'))
        return r

    time = []
    Tmean = []

    for fname in s.files['vtk_id0'][0:3]:
        print(fname)
        ds = AthenaDataSet(fname)
        T = ds.read_all_data('temperature')
        ma_w = (T < 1.5e4) & (T > 6e3)
        Tw = np.ma.array(T, mask=~ma_w)
        Tmean.append(np.ma.mean(Tw, axis=(1,2)))
        time.append(ds.domain['time'])

    x, y, z = cc_arr(ds.domain)
    r = dict(time=np.array(time),
             Tw_mean=Tmean, Tbdry=[6e3, 1.5e4],
             basedir=s.basedir, z=z,
             domain=s.domain)

    pickle.dump(r, open(fname, 'wb'))
    print('Result dumped to {0:s}'.format(fname))

    return r
