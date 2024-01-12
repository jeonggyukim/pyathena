# dust_pol.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import os.path as osp
import astropy.units as au
from astropy.io import fits

from ..load_sim import LoadSim

class DustPol:
    """
    Methods to calculate dust polarization map projected onto x-y, y-z, x-z planes
    """

    @LoadSim.Decorators.check_pickle
    def read_IQU(self, num, los=('x','y','z'),
                 prefix='dust_pol', savdir=None, force_override=False):

        ds, dd = self.get_field_dust_pol(num)

        return self.calc_IQU(ds, dd, los=los)

    def get_field_dust_pol(self, num):
        self.logger.info('[DustPol.get_field]: reading vtk data..')
        ds = self.load_vtk(num)
        dd = self.ds.get_field(field=['nH', 'Bx', 'By', 'Bz'])

        return ds, dd

    def calc_IQU(self, ds, dd, los,
                 Td=18.0*au.K, p0=0.2,
                 sigmad=1e-26*au.cm**2, nu=353*au.GHz):
        """
        Td : Dust temperature
        p0 : intrisic polarization fraction
        sigmad: Dust opacity tau_353/NH ~ 1.2e-26 cm^2
            (from Planck XX, DOI: 10.1051/0004-6361/201424086)
        """

        axis_idx = dict(x=0, y=1, z=2)
        # Follows even cyclic permutation of xyz
        los = dict(z=dict(B1='Bx', B2='By', B3='Bz'),
                   x=dict(B1='By', B2='Bz', B3='Bx'),
                   y=dict(B1='Bz', B2='Bx', B3='By'))

        from astropy.modeling import models
        bb = models.BlackBody(temperature=Td)
        Bnu = bb(nu)

        I = dict()
        Q = dict()
        U = dict()
        NH = dict()
        B1proj = dict()
        B2proj = dict()

        for dim in los:
            dx_cgs = ds.domain['dx'][axis_idx[dim]]*self.u.length.cgs.value
            B1 = dd[los[dim]['B1']]
            B2 = dd[los[dim]['B2']]
            B3 = dd[los[dim]['B3']]

            # Bperp_sq = B1*B1 + B2*B2
            # cos(2phi) = (B2*B2 - B1*B1)/Bperp_sq
            # sin(2phi) = -2.0*B1*B2/Bperp_sq
            Bperp_sq = B1**2 + B2**2
            cos_gamma_sq = Bperp_sq/(Bperp_sq + B3**2)
            I[dim] = (((Bnu*sigmad*dx_cgs)*(1.0 - p0)*(cos_gamma_sq - 2.0/3.0)*dd['nH']).sum(dim=dim)).data
            Q[dim] = (((Bnu*sigmad*dx_cgs)*p0*((B2*B2-B1*B1)/Bperp_sq)*cos_gamma_sq*dd['nH']).sum(dim=dim)).data
            U[dim] = (((Bnu*sigmad*dx_cgs)*p0*(-2.0*B1*B2/Bperp_sq)*cos_gamma_sq*dd['nH']).sum(dim=dim)).data

            # Density weighted projection
            nHsum = dd['nH'].sum(dim=dim)
            NH[dim] = (nHsum*dx_cgs).data
            B1proj[dim] = ((B1*dd['nH']).sum(dim=dim)/nHsum).data
            B2proj[dim] = ((B2*dd['nH']).sum(dim=dim)/nHsum).data

        return dict(I=I, Q=Q, U=U,
                    NH=NH, B1proj=B1proj, B2proj=B2proj,
                    time=ds.domain['time'],
                    p0=p0, Td=Td.value, sigmad=sigmad.value, nu=nu.value)

    def plt_Bproj(self, num, density=1.5,
                  Bproj=True, polvec=False, savefig=True):

        r = self.read_IQU(num, force_override=False)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # ds = s.load_vtk(0)
        # cc = ds.get_cc_pos()
        # x = cc['x']
        # y = cc['y']
        sp = self.load_starpar_vtk(num)
        domain = self.domain
        # Assume that all directions have the same dx, Nx
        Nx = domain['Nx'][0]
        Ny = domain['Nx'][0]
        x = np.linspace(domain['le'][0] + domain['dx'][0]/2,
                        domain['re'][0] - domain['dx'][0]/2, Nx)
        y = np.linspace(domain['le'][1] + domain['dx'][1]/2,
                        domain['re'][1] - domain['dx'][1]/2, Ny)

        Nskip = 32
        xlabel = dict(x='y', y='z', z='x')
        ylabel = dict(x='z', y='x', z='y')

        for ax, dim in zip(axes, ('x','y','z')):
            I = r['I'][dim]
            Q = r['Q'][dim]
            U = r['U'][dim]
            NH = r['NH'][dim]
            B1proj = r['B1proj'][dim]
            B2proj = r['B2proj'][dim]
            ang = np.arctan2(U, Q)/2

            plt.sca(ax)
            plt.imshow(NH, norm=LogNorm(1e19,1e23), extent=(-40,40,-40,40), origin='lower')
            #plt.imshow(NH, norm=LogNorm(1e19,1e23), extent=(-40,40,-40,40))
            if Bproj:
                plt.streamplot(x.data, y.data, B1proj, B2proj, color='w', density=density,
                               linewidth=0.75, arrowsize=0.5)
            if polvec:
                plt.quiver(x[::Nskip], y[::Nskip],
                           np.cos(ang)[::Nskip,::Nskip], np.sin(ang)[::Nskip,::Nskip],
                           headwidth=0, headlength=0, headaxislength=0,
                           pivot='mid', color='r', scale=20, width=0.0075)

            plt.xlim(domain['le'][0], domain['re'][0])
            plt.ylim(domain['le'][0], domain['re'][0])
            plt.xlabel(xlabel[dim])
            plt.ylabel(ylabel[dim])

        plt.tight_layout()
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.suptitle(self.basename + '  t={0:4.1f}'.format(sp.time))

        if savefig:
            savdir = osp.join(self.savdir, 'dust_pol')
            if not osp.exists(savdir):
                os.makedirs(savdir)

            savname = osp.join(savdir, '{0:s}_{1:04d}.png'.format(self.basename, num))
            plt.savefig(savname, dpi=300, bbox_inches='tight')

        return fig

    def plt_dust_pol(self, num, dim='y', Nskip=8, ax=None):
        """Plot dust polarization vector (rotated) with quivers
        """
        if ax is None:
            ax = plt.gca()

        sigmad = self.par['opacity']['sigma_dust_PE0']*au.cm**2/1.87
        Sigma_thres_Avone = (self.u.muH*(au.u*1.008)/(sigmad)).to('Msun pc-2').value
        # print(sigmad, Sigma_thres_Avone)

        domain = self.domain
        dat = self.read_IQU(num)
        prj = self.read_prj(num)
        # sp = self.load_starpar_vtk(num_sp)
        extent = prj['extent'][dim]

        # Draw dust polarization
        Nx = domain['Nx'][0]
        Ny = domain['Nx'][0]
        x = np.linspace(domain['le'][0] + domain['dx'][0]/2,
                        domain['re'][0] - domain['dx'][0]/2, Nx)
        y = np.linspace(domain['le'][1] + domain['dx'][1]/2,
                        domain['re'][1] - domain['dx'][1]/2, Ny)
        I = dat['I'][dim]
        Q = dat['Q'][dim]
        U = dat['U'][dim]
        ang = np.arctan2(U, Q)/2

        B1proj = dat['B1proj'][dim]
        B2proj = dat['B2proj'][dim]
        Bmagproj = np.sqrt(B1proj**2 + B2proj**2)
        Bmax = 15.0 # Maximum magnetic field strength in microG

        ax.quiver(x[::Nskip], y[::Nskip],
                  np.cos(ang)[::Nskip,::Nskip]*Bmagproj[::Nskip,::Nskip]/Bmax*2.0,
                  np.sin(ang)[::Nskip,::Nskip]*Bmagproj[::Nskip,::Nskip]/Bmax*2.0,
                  headwidth=0, headlength=0, headaxislength=0,
                  pivot='mid', color='#02ab2e', scale=30, width=0.004)

        # Draw contour
        X, Y = np.meshgrid(x,y)
        Z = prj[dim]['Sigma_HI'] + prj[dim]['Sigma_H2']
        ax.contour(X,Y,Z, extent=extent, levels=[Sigma_thres_Avone], colors='k',
                   linewidths=0.75, linestyles='solid')

        return ax


    def write_IQU_to_fits(self, num):

        def _create_fits_header(r, domain):
            pc = self.u.pc
            hdr = fits.Header()

            hdr['xmin'] = (domain['le'][0]*pc, 'pc')
            hdr['xmax'] = (domain['re'][0]*pc, 'pc')
            hdr['ymin'] = (domain['le'][1]*pc, 'pc')
            hdr['ymax'] = (domain['re'][1]*pc, 'pc')
            hdr['zmin'] = (domain['le'][2]*pc, 'pc')
            hdr['zmax'] = (domain['re'][2]*pc, 'pc')
            hdr['dx'] = (domain['dx'][0]*pc, 'pc')
            hdr['dy'] = (domain['dx'][1]*pc, 'pc')
            hdr['dz'] = (domain['dx'][2]*pc, 'pc')

            hdr['time'] = (r['time']*self.u.Myr, 'Myr')
            hdr['nu'] = (r['nu'], 'GHz')
            hdr['Td'] = (r['Td'], 'K')
            hdr['sigmad'] = (r['sigmad'], 'cm^2/H')

            hdu = fits.PrimaryHDU(header=hdr)

            return hdu

        r = self.read_IQU(num)

        fitsname = osp.join(self.savdir, 'dust_pol',
                            self.problem_id + '.{0:04}.fits'.format(num))
        self.logger.info('Save fits file to {0:s}'.format(fitsname))
        hdul = fits.HDUList()
        hdu = _create_fits_header(r, self.domain)
        hdul.append(hdu)

        for axis in ('x','y','z'):
            for label in ('I', 'Q', 'U', 'NH', 'B1proj', 'B2proj'):
                name = label + '_' + axis
                data = r[label][axis]
                hdul.append(fits.ImageHDU(name=name, data=data))

        hdul.writeto(fitsname, overwrite=True)
        self.logger.info('[write_IQU_to_fits]: Wrote to {0:s}'.format(fitsname))
