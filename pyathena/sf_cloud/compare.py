
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

import os
import os.path as osp
import numpy as np
import astropy.constants as ac
import astropy.units as au
from matplotlib.colors import LogNorm

from ..io.read_vtk import read_vtk
from .slc_prj import SliceProj
from ..plt_tools import plt_starpar

class Compare(object):

    def comp_snapshot(self, models, num, norm_factor=3.0, savefig=True, prefix=None):
        s = self.set_model(models[0])
        Sigma_conv = ((s.u.muH*1.008*ac.u.cgs/au.cm**3).to('Msun/pc**3')).value
        norm = LogNorm(1e-1,1e3)
        cmap = 'Spectral_r'
        agemax = 10.0
        nr = 3
        nc = len(models)
        fig, axes = plt.subplots(nr, nc, figsize=(4.0*nc, 12.5),
                                 gridspec_kw=dict(hspace=0.0, wspace=0.0))
        
        for ic, mdl in enumerate(models):
            s = self.set_model(mdl)
            sp = s.load_starpar_vtk(num)
            for ir, axis in enumerate(('x','y','z')):
                extent = SliceProj.get_extent(s.domain)[axis]
                ii = ir + 1
                f = f'Sigma{ii}'
                dd = read_vtk(s.files[f][num])
                d = dd.get_field(f).squeeze()
                im1 = []
                im1.append(axes[ir,ic].imshow(d[f]*Sigma_conv, norm=norm,
                                              extent=extent, origin='lower', cmap=cmap))

                # Overplot starpar
                if not sp.empty:
                    plt_starpar.scatter(sp, axes[ir,ic], axis=axis, kind='proj', kpc=False,
                                        norm_factor=norm_factor, agemax=agemax)
                    plt_starpar.scatter(sp, axes[ir,ic], axis=axis, kind='proj', kpc=False,
                                        norm_factor=norm_factor, agemax=agemax)
                    axes[ir,ic].set_xlim(extent[0],extent[1])
                    axes[ir,ic].set_ylim(extent[2],extent[3])
                    
        bbox_ax_top = axes[0,0].get_position()
        cax = fig.add_axes([bbox_ax_top.x0+0.00, bbox_ax_top.y1+0.01,
                            bbox_ax_top.x1-bbox_ax_top.x0-0.0, 0.02])
        cbar = plt.colorbar(im1[0], cax=cax, orientation='horizontal')
        cbar.set_label(label=r'$\Sigma_{\rm gas}\;[M_{\odot}\,{\rm pc}^{-2}]$', fontsize='medium')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar_yticks = plt.getp(cbar.ax.axes, 'xticklabels')
        plt.setp(cbar_yticks, color='k', fontsize='small')

        for ax in axes.flatten():
            plt.axis('on')
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

        bbox_sp = [bbox_ax_top.x0+0.2, bbox_ax_top.y1+0.01,
                   0.1, 0.015]
        plt_starpar.colorbar(plt.gcf(), agemax,
                             #bbox=[0.35, 0.89, 0.1, 0.015])
                             bbox=bbox_sp)
        plt_starpar.legend(axes[0,2], norm_factor=4.0, mass=[1e2, 1e3], location='top',
                           fontsize='medium',
                           bbox_to_anchor=dict(top=(0.45, 0.93), right=(0.48, 0.91)))
        
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.suptitle('{0:s}    time={1:5.2f}'.format(prefix, sp.time), x=0.7, y=0.91)
        
        if savefig:
            if prefix is None:
                prefix = '-'.join(models)

            savdir = osp.join('/tigress/jk11/figures/GMC/comp_snapshot', prefix)
            if not osp.exists(savdir):
                os.makedirs(savdir)

            savname = osp.join(savdir, '{0:s}_{1:04d}.png'.format(prefix, num))
            plt.savefig(savname, dpi=200, bbox_inches='tight')
            
        return fig
    
