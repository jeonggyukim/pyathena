import numpy as np
import xarray as xr

from ..decorators.decorators import check_netcdf

class RadiationSlice:

    @check_netcdf
    def read_slice(self, num, prefix='slice', slc=None,
                   fields=['nH','T','xHI','xHII','xe', 'xi_CR',
                           'Erad_LW','Erad_PE','Erad_LyC','Uion','Uion_pi'],
                   savdir=None, force_override=False):
        ds = self.load_vtk(num)
        if slc is None:
            dx = ds.domain['dx'][2]
            slc = {'z': slice(-dx, dx)}

        dd = ds.get_field(fields)
        del dd.attrs['dfi']
        del dd.attrs['all_grid_equal']
        return dd.sel(**slc)

    @check_netcdf
    def read_slice_all(self, nums=None, merge_with_hst=True, prefix='slice_all',
                       force_override=False, read_slice_kwargs=None):
        if read_slice_kwargs is None:
            read_slice_kwargs = dict(prefix='slice', force_override=False)
        if nums is None:
            nums = self.nums

        dlist = []
        for num in nums:
            print(num, end=' ')
            d = self.read_slice(num, **read_slice_kwargs)
            dlist.append(d.expand_dims(time=np.atleast_1d(d.attrs['time'])))

        self.logger.info(
            '[read_slice] Concatenating {0:d} xarray datasets.'.format(len(dlist)))

        dd = xr.concat(dlist, dim='time')
        if merge_with_hst:
            dd = self._zprof_post_process(dd)
            dd = self.merge_zprof_with_hst(dd, force_override=force_override)

        return dd
