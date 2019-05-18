# read_zprof.py

import os
import os.path as osp

import xarray as xr
import numpy as np

from ..io.read_zprof import read_zprof_all

class ReadZprof:

    def read_zprof(self, phase='all', savdir=None, force_override=False):
        """Function to read all zprof output

        Parameters
        ----------
        phase : str or list of str
            List of thermal phases to read. Possible phases are
            'c': (phase1, cold, T < 180),
            'u': (phase2, unstable, 180 <= T < 5050),
            'w': (phase3, warm, 5050 <= T < 2e4),
            'h1' (phase4, warm-hot, 2e4 < T < 5e5),
            'h2' (phase5, hot-hot, T >= 5e5)
            'whole' (entire temperature range)
            'h' = h1 + h2
            '2p' = c + u + w
            If 'all', read all phases.
        savdir: str
            Name of the directory where pickled data will be saved.
        force_override: bool
            If True, read all (pickled) zprof profiles and dump in netCDF format.

        Returns
        -------
        zp : dict
            dictionary containing xarray datasets for each thermal phase.
        """
        
        dct = dict(c='phase1',
                   u='phase2',
                   w='phase3',
                   h1='phase4',
                   h2='phase5',
                   whole='whole')
        
        if phase == 'all':
            phase = list(dct.keys()) + ['h', '2p']
        else:
            phase = np.atleast_1d(phase)

        zp = dict()
        for ph in phase:
            if ph == 'h':
                zp[ph] = \
                self._read_zprof(phase=dct['h1'], savdir=savdir,
                                 force_override=force_override) + \
                self._read_zprof(phase=dct['h2'], savdir=savdir,
                                 force_override=force_override)
            elif ph == '2p':
                zp[ph] = \
                self._read_zprof(phase=dct['c'], savdir=savdir,
                                 force_override=force_override) + \
                self._read_zprof(phase=dct['u'], savdir=savdir,
                                 force_override=force_override) + \
                self._read_zprof(phase=dct['w'], savdir=savdir,
                                 force_override=force_override)
            else:
                zp[ph] = self._read_zprof(phase=dct[ph], savdir=savdir,
                                          force_override=force_override)

        if len(phase) == 1:
            self.zp = zp[ph]
        else:
            self.zp = zp

        return self.zp
    
    def _read_zprof(self, phase='whole', savdir=None, force_override=False):
        """Function to read zprof and convert quantities to convenient units.
        """

        # Create savdir if it doesn't exist
        if savdir is None:
            savdir = osp.join(self.savdir, 'zprof')
           
        if not osp.exists(savdir):
            os.makedirs(savdir)

        fnetcdf = '{0:s}.{1:s}.zprof.mod.nc'.format(self.problem_id, phase)
        fnetcdf = osp.join(savdir, fnetcdf)

        # Check if the original zprof files are updated
        mtime = max([osp.getmtime(f) for f in self.files['zprof']])

        if not force_override and osp.exists(fnetcdf) and \
            osp.getmtime(fnetcdf) > mtime:
            self.logger.info('[read_zprof]: Read {0:s}'.format(phase) + \
                             ' zprof from existing NetCDF dump.')
            ds = xr.open_dataset(fnetcdf)
            return ds
        else:
            self.logger.info('[read_zprof]: Read from original {0:s}'.\
                format(phase) + ' zprof dump and renormalize.'.format(phase))

        # If we are here, force_override is True or zprof files are updated.
        # Read original zprof dumps.
        ds = read_zprof_all(osp.dirname(self.files['zprof'][0]), self.problem_id,
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

        # For the moment s1 is assumed to be nH0 # JGKIM
        ds['nH0'] = ds.s1
        # Volume filling factor of ionized gas
        ds['xi'] = ds.A - ds.xn
        # Electron number density averaged over Atot
        ds['ne'] = ds.d - ds.s1
        # Electron number density averaged over Atot
        ds['nebar'] = ds.ne/ds.xi
        
        # Rename time to time_code and use physical time in Myr as dimension
        ds = ds.rename(dict(time='time_code'))
        ds = ds.assign_coords(time=ds.time_code*self.u.Myr)
        ds = ds.assign_coords(z_kpc=ds.z*self.u.kpc)
        ds = ds.swap_dims(dict(time_code='time'))
        
        # self._set_attrs(ds)
        
        # Somehow overwriting using mode='w' doesn't work..
        if osp.exists(fnetcdf):
            os.remove(fnetcdf)

        try:
            ds.to_netcdf(fnetcdf, mode='w')
        except IOError:
            pass

        return ds

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
