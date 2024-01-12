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
from ..plt_tools.utils import texteffect

class Compare(object):

    def comp_snapshot(self, models, num, labels, prefix, norm_factor=3.0, savefig=True):
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
            try:
                sp = s.load_starpar_vtk(num)
                for ir, dim in enumerate(('x','y','z')):
                    extent = SliceProj.get_extent(s.domain)[dim]
                    ii = ir + 1
                    f = f'Sigma{ii}'
                    dd = read_vtk(s.files[f][num])
                    d = dd.get_field(f).squeeze()
                    im1 = []
                    im1.append(axes[ir,ic].imshow(d[f]*Sigma_conv, norm=norm,
                                                  extent=extent, origin='lower', cmap=cmap))

                    # Overplot starpar
                    if not sp.empty:
                        plt_starpar.scatter_sp(sp, axes[ir,ic], dim=dim, kind='proj', kpc=False,
                                               norm_factor=norm_factor, agemax=agemax)
                        plt_starpar.scatter_sp(sp, axes[ir,ic], dim=dim, kind='proj', kpc=False,
                                               norm_factor=norm_factor, agemax=agemax)
                        axes[ir,ic].set_xlim(extent[0],extent[1])
                        axes[ir,ic].set_ylim(extent[2],extent[3])
            except OSError:
                print('File not found for model {0:s} and num {1:04d}. Skipping'.\
                      format(mdl, num))

        bbox0 = axes[0,0].get_position()
        cax = fig.add_axes([bbox0.x0+0.01, bbox0.y1+0.01,
                            bbox0.x1-bbox0.x0-0.02, 0.02])
        cbar = plt.colorbar(im1[0], cax=cax, orientation='horizontal')
        cbar.set_label(label=r'$\Sigma_{\rm gas}\;[M_{\odot}\,{\rm pc}^{-2}]$',
                       fontsize='medium')
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

        # Add labels
        for ax, label in zip(axes[0,:], labels):
            ax.text(0.05, 0.9, label, transform=ax.transAxes,
                    **texteffect(fontsize=20))

        bbox1 = axes[0,1].get_position()
        bbox2 = axes[0,2].get_position()
        plt_starpar.colorbar_sp(fig, agemax, bbox=[bbox1.x0+0.02, bbox1.y1+0.01,
                                                   bbox1.x1-bbox1.x0-0.04, 0.02])
        try:
            plt_starpar.legend_sp(axes[0,2], norm_factor=4.0, mass=[1e2, 1e3], location='top',
                                  fontsize='medium',
                                  bbox_to_anchor=dict(top=(bbox2.x0+0.02, bbox2.y1+0.05),
                                                      right=(bbox2.x1-0.04, bbox2.y1+0.06)))
        except IndexError:
            pass

        plt.subplots_adjust(wspace=None, hspace=None)
        plt.suptitle('time={0:5.2f} '.format(sp.time) + prefix,
                     x=0.7, y=bbox1.y1+0.05,
                     verticalalignment='bottom')

        if savefig:
            if prefix is None:
                prefix = '-'.join(models)

            savdir = osp.join('/tigress/jk11/figures/GMC/comp_snapshot', prefix)
            if not osp.exists(savdir):
                os.makedirs(savdir)

            savname = osp.join(savdir, '{0:s}_{1:04d}.png'.format(prefix, num))
            plt.savefig(savname, dpi=200, bbox_inches='tight')

        return fig
