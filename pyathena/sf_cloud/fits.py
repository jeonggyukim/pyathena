from astropy.io import fits
import os
import numpy as np

class Fits:
    def save_to_fits(self, num, fields = ['nH','T'], savname=None):

        def _create_fits_header(ds, par):
            units = self.u
            hdr = fits.Header()
            hdr['time_code'] = ds.domain['time']
            hdr['time_Myr'] = (ds.domain['time']*u.Myr, 'Myr')
            hdr['xmin'] = (ds.domain['le'][0]*u.pc, 'pc')
            hdr['xmax'] = (ds.domain['re'][0]*u.pc, 'pc')
            hdr['ymin'] = (ds.domain['le'][1]*u.pc, 'pc')
            hdr['ymax'] = (ds.domain['re'][1]*u.pc, 'pc')
            hdr['zmin'] = (ds.domain['le'][2]*u.pc, 'pc')
            hdr['zmax'] = (ds.domain['re'][2]*u.pc, 'pc')
            hdr['dx'] = (ds.domain['dx'][0]*u.pc, 'pc')
            hdr['dy'] = (ds.domain['dx'][1]*u.pc, 'pc')
            hdr['dz'] = (ds.domain['dx'][2]*u.pc, 'pc')
            #hdr['unit']=(units[field].value, units[field].unit)

            if 'qshear' in par['problem']:
                hdr['qshear'] = par['problem']['qshear']
            if 'Omega' in par['problem']:
                hdr['Omega'] = (par['problem']['Omega'], 'km/s/pc')

            hdu = fits.PrimaryHDU(header=hdr)

            return hdu

        ds = self.load_vtk(num)
        hdul = fits.HDUList()
        hdu = _create_fits_header(ds, self.par)
        hdul.append(hdu)

        if savname is None:
            savdir = osp.join(self.savdir, 'fits')
            if osp.exists(savdir):
                os.makedirs(savdir)
            savname = osp.join(savdir, '{0:s}.{1:04d}.fits'.format(self.problem_id, num))

        names=dict(nH='den', T='temp', vx='vx', vy='vy', vz='vz', nHI='den_HI')
        dd = ds.get_field(fields)
        for field in fields:
            hdul.append(fits.ImageHDU(name=names[field], data=dd[field].values))

        hdul.writeto(savname, overwrite=True)

        return None
