import os
import os.path as osp
from shutil import copyfile
import getpass

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,LogNorm
from mpl_toolkits.axes_grid1 import ImageGrid
import cmocean
import astropy.constants as ac

from ..io.read_vtk import read_vtk

from ..plt_tools.plt_starpar import scatter_sp, colorbar_sp, legend_sp
from ..plt_tools.cmap_shift import cmap_shift
from ..plt_tools.make_movie import make_movie

class PltSnapshotVTK2D:

    def make_movie_snapshot_vtk2d(self, fps=15, basedir=None, savdir=None):
        if basedir is None:
            basedir = self.basedir

        if savdir is None:
            savdir = osp.join(basedir, 'movies')
            if not osp.exists(savdir):
                os.makedirs(savdir)

        fin = osp.join(basedir, 'snapshot_vtk2d/*.png')
        fout = osp.join(savdir, '{0:s}_snapshot_vtk2d.mp4'.format(self.basename))
        if make_movie(fin, fout, fps, fps):
            savdir2='/tigress/{0:s}/public_html/movies/SF-CLOUD/'.\
                format(getpass.getuser())
            fout2 = osp.join(savdir2, osp.basename(fout))
            copyfile(fout, fout2)
            print('Copied movie file to {0:s}'.format(fout2))

    def plt_snapshot_vtk2d(self, num, dim='y',
                           fields = ['Sigma','Sigma_H2','Sigma_HI',
                                     'EM','d','T',
                                     'P','vmag','Bmag'],
                           figsize=(20,16), nrows_ncols=(3,3), axes_pad=(0.5,0.8),
                           suptitle=None, savefig=False, savdir=None, make_movie=False):

        if self.par['configure']['gas'] == 'hydro':
            fields = ['Sigma','Sigma_H2','Sigma_HI',
                      'EM','d','T',
                      'P','vmag','Erad_LyC']

        fig = plt.figure(figsize=figsize)
        axes = ImageGrid(fig, 111, nrows_ncols=nrows_ncols,
                         axes_pad=axes_pad, label_mode='1',
                         share_all=True, cbar_location='top', cbar_mode='each',
                         cbar_size='5%', cbar_pad='1%')

        for ax,field in zip(axes,fields):
            dd = self.plt_snapshot_vtk2d_one_axis(self, num, field, dim, ax)


        sp = self.load_starpar_vtk(num)
        agemax_sp = 10.0
        if not sp.empty:
            for ax in (axes[0],):
                scatter_sp(sp, ax, dim=dim, norm_factor=1.0,
                           kind='prj', cmap=mpl.cm.cool_r, kpc=False, runaway=True,
                           agemax=agemax_sp, plt_old=True)
                extent = (self.domain['le'][0], self.domain['re'][0])
                ax.set_xlim(*extent)
                ax.set_ylim(*extent)

        # Add starpar age colorbar
        #cax = fig.add_axes([0.125, 0.9, 0.1, 0.015])
        # cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
        #                                orientation='horizontal', ticks=[0, 4, 8])
#         bbox0 = axes[0].get_position()
#         print(bbox0)
#         cb = colorbar_sp(fig, agemax_sp, cmap=mpl.cm.cool_r,
#                          bbox=[0.2, bbox0.y1+0.03, 0.05,0.01])
# #                               0.5*(bbox0.x1-bbox0.x0), 0.025])
#         cb.set_label(r'${\rm age}\;[{\rm Myr}]$', fontsize=14)
#         cb.ax.xaxis.set_ticks_position('top')
#         cb.ax.xaxis.set_label_position('top')


        bbox1 = axes[0].get_position()

        if suptitle is None:
            suptitle = self.basename
        plt.suptitle(suptitle + '  time={0:5.2f}'.format(sp.time),
                     x=0.5, y=bbox1.y1+0.05,
                     ha='center', va='bottom')

        if savefig:
            if savdir is None:
                savdir = osp.join(self.basedir, 'snapshot_vtk2d')
            if not osp.exists(savdir):
                os.makedirs(savdir)

            savname = osp.join(savdir, '{0:s}_{1:04d}.png'.\
                               format(self.basename, num))
            fig.savefig(savname, dpi=200, bbox_inches='tight')
            plt.close(fig)


    @staticmethod
    def plt_snapshot_vtk2d_one_axis(s, num, field, dim='y',
                                    ax=None, norm=None, cmap=None):

        dtoi = dict(x=1,y=2,z=3)

        cmap_def = dict(
            Sigma=plt.cm.pink_r,
            Sigma_H2=plt.cm.pink_r,
            Sigma_HI=plt.cm.pink_r,
            EM=plt.cm.plasma,
            d=plt.cm.Spectral_r,
            T=cmap_shift(mpl.cm.RdYlBu_r, midpoint=3./7.),
            P=plt.cm.magma,
            vmag=cmocean.cm.tempo,
            Bmag=cmocean.cm.amp,
            Erad_LyC=plt.cm.viridis,
        )

        norm_def = dict(
            Sigma=LogNorm(1e0,1e3),
            Sigma_H2=LogNorm(1e0,1e3),
            Sigma_HI=LogNorm(1e0,1e3),
            EM=LogNorm(1e1,1e5),
            d=LogNorm(1e-3,1e3),
            P=LogNorm(1e2,1e6),
            T=LogNorm(1e1,1e7),
            vmag=LogNorm(1,1000),
            Bmag=LogNorm(1e-1,1e2),
            Erad_LyC=LogNorm(1e-16,1e-10),
        )

        label = dict(
            Sigma=r'$\Sigma\,[M_{\odot}\,{\rm pc}^{-2}]$',
            Sigma_H2=r'$\Sigma_{\rm H_2}\,[M_{\odot}\,{\rm pc}^{-2}]$',
            Sigma_HI=r'$\Sigma_{\rm HI}\,[M_{\odot}\,{\rm pc}^{-2}]$',
            EM=r'${\rm EM}\,[{\rm cm}^{-6}\,{\rm pc}]$',
            d=r'$n_{\rm H}\,[{\rm cm}^{-3}]$',
            T=r'$T\,[{\rm K}]$',
            P=r'$P/k_{\rm B}\,[{\rm cm}^{-3}\,{\rm K}]$',
            vmag=r'$|\mathbf{v}|\,[{\rm km}\,{\rm s}^{-1}]$',
            Bmag=r'$|\mathbf{B}|\,[\mu {\rm G}]$',
            Erad_LyC=r'$\mathcal{E}_{\rm LyC}\,[{\rm erg}\,{\rm cm}^{-3}]$',
        )

        unit_conv = dict(
            Sigma=(s.u.density*s.domain['dx'][dtoi[dim]]*s.u.length).to('Msun/pc2').value,
            Sigma_H2=(s.u.density*s.domain['dx'][dtoi[dim]]*s.u.length).to('Msun/pc2').value,
            Sigma_HI=(s.u.density*s.domain['dx'][dtoi[dim]]*s.u.length).to('Msun/pc2').value,
            EM=s.u.pc,
            d=1.0,
            P=s.u.energy_density.cgs.value/ac.k_B.cgs.value,
            T=1.0,
            vmag=1.0,
            Bmag=np.sqrt(s.u.energy_density.cgs.value)*np.sqrt(4.0*np.pi)*1e6,
            Erad_LyC=s.u.energy_density.cgs.value
        )

        if ax is None:
            ax = plt.gca()
        if norm is None:
            norm = norm_def[field]
        if cmap is None:
            cmap = cmap_def[field]

        if dim == 'z':
            lx = s.domain['Lx'][0]; ly = s.domain['Lx'][1]
            xlabel = 'x [pc]'; ylabel = 'y [pc]'
        elif dim == 'y':
            lx = s.domain['Lx'][2]; ly = s.domain['Lx'][0]
            xlabel = 'x [pc]'; ylabel = 'z [pc]'
        elif dim == 'x':
            lx = s.domain['Lx'][1]; ly = s.domain['Lx'][2]
            xlabel = 'x [pc]'; ylabel = 'z [pc]'

        xticks = [-0.5*lx,-0.25*lx,0.0,0.25*lx,0.5*lx]
        yticks = [-0.5*ly,-0.25*ly,0.0,0.25*ly,0.5*ly]

        #if field.startswith('Sigma_'):
            #f = 'Sigma' + str(dtoi[dim]) + field[5:]
        #else:
            #f = field + str(dtoi[dim])
        f = field + '_' + dim

        # read vtk 2d output
        ds = read_vtk(s.files[f][num])
        d = ds.get_field(f)
        dd = d.sel(**{dim:0.0}, method='nearest')*unit_conv[field]

        # Set arguments
        imshow_args = dict(
            ax=ax, norm=norm, cmap=cmap, xticks=xticks, yticks=yticks,
            extend='neither', add_labels=False, cbar_ax=ax.cax,
            cbar_kwargs=dict(orientation='horizontal',
                             spacing='proportional', shrink=0.8))

        im = dd[f].plot.imshow(**imshow_args)
        ax.set(xlabel=xlabel, ylabel=ylabel, aspect='equal')
        cb = im.colorbar
        cb.set_label(label[field], fontsize=15)
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.tick_params(labelsize=12)

        return dd
