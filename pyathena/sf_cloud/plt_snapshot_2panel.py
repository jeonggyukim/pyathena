import os
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm, Normalize

from ..io.read_vtk import read_vtk
from ..plt_tools.plt_starpar import scatter_sp, colorbar_sp, legend_sp

class PltSnapshot2Panel:

    def plt_snapshot_2panel(self, num, name=None, dim='y', agemax_sp=8.0,
                            savdir=None, savfig=True):

        s = self
        if name is None:
            name = s.basename

        dim_to_idx = dict(z=2,y=1,x=0)
        idx = dim_to_idx[dim]
        domain = s.domain

        ds1 = read_vtk(s.files[f'Sigma_{dim}'][num])
        dd1 = ds1.get_field(f'Sigma_{dim}')
        try:
            ds2 = read_vtk(s.files[f'EM_{dim}'][num])
            dd2 = ds2.get_field(f'EM_{dim}')
        except IndexError:
            pass

        dfi = s.dfi # dictionary containing derived field info
        sp = s.load_starpar_vtk(num) # read starpar vtk as pandas DataFrame

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

        fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                                 constrained_layout=False)

        # Get neutral gas surface density in Msun/pc^2
        to_Sigma = (s.u.density*domain['dx'][idx]*s.u.length).to('Msun/pc2').value
        dd1['Sigma'] = to_Sigma*dd1[f'Sigma_{dim}'].sel(**{dim:0.0,'method':'nearest'})
        # Plot using xarray imshow
        im1 = dd1['Sigma'].plot.imshow(ax=axes[0], cmap='pink_r',
                                       norm=LogNorm(1e-1,1e3), origin='lower',
                                       extend='neither', add_labels=True, xticks=xticks,
                                       yticks=yticks,
                                       cbar_kwargs=dict(label=r'$\Sigma\;[M_{\odot}\,{\rm pc}^{-2}]$'))
        try:
            dd2['EM'] = dd2[f'EM_{dim}'].sel(**{dim:0.0,'method':'nearest'})
            im2 = dd2['EM'].plot.imshow(ax=axes[1], cmap='plasma', norm=LogNorm(3e1,3e5),origin='lower',
                                        extend='neither', add_labels=False, xticks=xticks, yticks=yticks,
                                        cbar_kwargs=dict(label=r'${\rm EM}\;[{\rm cm^{-6}\,{\rm pc}}]$'))
        except UnboundLocalError:
            pass

        for ax in axes:
            ax.set_aspect('equal')

        norm = mpl.colors.Normalize(vmin=0., vmax=agemax_sp)
        cmap = mpl.cm.cool_r
        # Scatter plot star particles (color: age, mass: size)
        if not sp.empty:
            for ax in (axes[0], axes[1]):
                scatter_sp(sp, ax, dim=dim, norm_factor=1.0,
                           kind='prj', cmap=cmap, kpc=False, runaway=True,
                           agemax=agemax_sp, plt_old=True)
                extent = (domain['le'][0], domain['re'][0])
                ax.set_xlim(*extent)
                ax.set_ylim(*extent)

        # Add starpar age colorbar
        #cax = fig.add_axes([0.125, 0.9, 0.1, 0.015])
        # cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
        #                                orientation='horizontal', ticks=[0, 4, 8])
        bbox0 = axes[0].get_position()
        cb = colorbar_sp(fig, agemax_sp, bbox=[bbox0.x0, bbox0.y1+0.03,
                                               0.5*(bbox0.x1-bbox0.x0), 0.025])

        # cbar_sp.ax.tick_params(labelsize=14)
        cb.set_label(r'${\rm age}\;[{\rm Myr}]$', fontsize=14)
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')

        # Add legends for starpar mass
        legend_sp(axes[0], norm_factor=1.0, mass=[1e2, 1e3], location='top', fontsize='medium',
                  #bbox_to_anchor=dict(top=(0.22, 0.97), right=(0.48, 0.91)))
                  bbox_to_anchor=dict(top=(bbox0.x0 + 0.45*(bbox0.x1-bbox0.x0), bbox0.y1+0.11),
                                      right=(bbox0.x1, bbox0.y1+0.15)))

        # Set simulation time [code] as suptitle
        # plt.suptitle(r'$t$={0:.2f}'.format(ds1.domain['time']), x=0.5, y=0.97)
        #plt.tight_layout()
        #plt.subplots_adjust(top=0.94)
        bbox1 = axes[1].get_position()
        plt.suptitle(name + '  time={0:5.2f} '.format(sp.time),
                     x=0.7, y=bbox1.y1+0.03, verticalalignment='bottom')
        #plt.subplots_adjust(wspace=0.4, hspace=None)

        if savfig:
            if savdir is None:
                savdir = osp.join(s.savdir,'snapshot_2panel')

            if not osp.exists(savdir):
                os.makedirs(savdir)

            plt.savefig(osp.join(savdir, '{0:s}_{1:04d}_{2:s}.png'.format(s.basename,num,dim)),
                        dpi=200)

        return fig
