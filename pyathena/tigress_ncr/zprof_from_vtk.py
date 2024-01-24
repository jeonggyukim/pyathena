import copy
import numpy as np
import xarray as xr
from functools import reduce

from .decorator import check_netcdf_zprof_vtk

class ZprofFromVTK:
    def _get_phase_info(self, phase_set=0):
        info0 = dict()
        info0['whole'] = dict(ph_id=0, cond=[])
        info0['CpU'] = dict(ph_id=1, cond=[['T', np.less, 6e3]])
        info0['WIM'] = dict(ph_id=2,
                             cond=[['xHII', np.greater_equal, 0.5],
                                   ['T', np.greater_equal, 6e3], ['T', np.less, 3.5e4]])
        info0['WNM'] = dict(ph_id=3,
                             cond=[['xHII', np.less, 0.5],
                                   ['T', np.greater_equal, 6e3], ['T', np.less, 3.5e4]])
        info0['hot'] = dict(ph_id=4, cond=[['T', np.greater_equal, 3.5e4]])

        info1 = dict()
        info1['whole'] = dict(ph_id=0, cond=[])
        info1['cold'] = dict(ph_id=1, cond=[['T', np.less, 3e3]])
        info1['wpion'] = dict(ph_id=2,
                              cond=[['xHII', np.greater_equal, 0.1], ['xHII', np.less, 0.9],
                                    ['T', np.greater_equal, 3e3], ['T', np.less, 3.5e4]])
        info1['wion'] = dict(ph_id=3,
                             cond=[['xHII', np.greater_equal, 0.9],
                                   ['T', np.greater_equal, 3e3], ['T', np.less, 3.5e4]])
        info1['wneu'] = dict(ph_id=4,
                             cond=[['xHII', np.less, 0.1],
                                   ['T', np.greater_equal, 3e3], ['T', np.less, 3.5e4]])
        info1['hot'] = dict(ph_id=5, cond=[['T', np.greater_equal, 3.5e4]])

        def _make_info_with_LyC_mask(info_in):
            info = copy.deepcopy(info_in)
            phases = list(info.keys())
            phases.remove('whole') # exclude whole
            nphase = len(phases)
            for ph in phases:
                info[ph + '_LyC'] = copy.deepcopy(info[ph])
                info[ph + '_LyC']['cond'].append(['Erad_LyC_mask', np.equal, 1.0])
                info[ph + '_noLyC'] = info.pop(ph)
                info[ph + '_noLyC']['cond'].append(['Erad_LyC_mask', np.equal, 0.0])
                info[ph + '_noLyC']['ph_id'] += nphase

            return info

        if phase_set == 0:
            return info0
        if phase_set == 1:
            return info1
        elif phase_set == 2:
            return _make_info_with_LyC_mask(info1)

    @check_netcdf_zprof_vtk
    def read_zprof_from_vtk(self, num,
                            fields=['nH','ne','nesq','xHII','xe',
                                    'T','Uion','Erad_LyC'],
                            phase_set=0,
                            prefix='zprof_vtk', weights=['nH','ne','nesq'],
                            savdir=None, force_override=False):

        if phase_set == 2 and not 'Erad_LyC_mask' in fields:
            fields.append('Erad_LyC_mask')

        ds = self.load_vtk(num)
        dd = ds.get_field(fields)
        v = list(dd.variables)[0]
        dd = dd.assign(ph_id=(('z','y','x'), np.zeros_like(dd[v], dtype=int)))
        dd = dd.assign(fV=(('z','y','x'), np.ones_like(dd[v])))

        ph_info = self._get_phase_info(phase_set)
        f_cond = lambda dd, f, op, v: op(dd[f], v)
        for ph in ph_info.keys():
            dd['ph_id'] += ph_info[ph]['ph_id']*\
                reduce(np.logical_and,
                       [f_cond(dd, *cond_) for cond_ in ph_info[ph]['cond']], 1)

        zp = dict()
        NxNy = dd['fV'].sum(dim=('x','y'))
        # z-profile of all gas
        zp['whole'] = dd.sum(dim=('x','y'), keep_attrs=True) / NxNy
        for ph in ph_info.keys():
            idx = (dd['ph_id'] == ph_info[ph]['ph_id'])
            zp[ph] = dd.where(idx).sum(dim=('x','y'), keep_attrs=True)
            zp[ph] /= NxNy

        if weights is not None:
            for w in np.atleast_1d(weights):
                zpw = dict()
                dw = (dd*dd[w])
                denom = dw['fV'].sum(dim=('x','y'))
                rename_dict = {v: v + '_w_' + w for v in list(dd.keys())}
                dw = dw.rename(rename_dict)
                zpw['whole'] = dw.sum(dim=('x','y'), keep_attrs=True) / denom
                for ph in ph_info.keys():
                    idx = (dd['ph_id'] == ph_info[ph]['ph_id'])
                    zpw[ph] = dw.where(idx).sum(dim=('x','y'), keep_attrs=True)
                    zpw[ph] /= denom

                for ph in ph_info.keys():
                    zp[ph] = xr.merge((zp[ph], zpw[ph]))

                del dw

        # Merge and create phase as new dimension
        zplist = []
        for ph in ph_info.keys():
            zplist.append(zp[ph].expand_dims('phase').assign_coords(phase=[ph]))

        zpa = xr.concat(zplist, dim='phase')
        zpa = zpa.to_array().to_dataset('phase')
        zpa = zpa.to_array('phase').to_dataset('variable')
        zpa = zpa.drop_vars('ph_id')

        def _check_fV(zp):
            fV = 0.0
            for ph in zp.phase.data:
                if ph != 'whole':
                    zp = zpa.sel(phase=ph)
                    fV += zp['fV']

            return np.allclose(fV.data, 1.0, rtol=1/NxNy, atol=1/NxNy)

        if not _check_fV(zpa):
            self.logger.warning(
                    "Volume fractions of all phases don't add up to 1")

        zpa = zpa.expand_dims(time=np.atleast_1d(dd.attrs['time']))

        return zpa

    @check_netcdf_zprof_vtk
    def read_zprof_from_vtk_all(self, nums, prefix='zprof_vtk_all',
                                force_override=False,
                                read_zprof_from_vtk_kwargs=None):
        if read_zprof_from_vtk_kwargs is None:
            read_zprof_from_vtk_kwargs = dict(force_override=False)

        zplist = []
        for num in nums:
            print(num, end=' ')
            zplist.append(self.read_zprof_from_vtk(num, **read_zprof_from_vtk_kwargs))

        return xr.merge(zplist)
