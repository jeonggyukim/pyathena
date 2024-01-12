import os
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pyathena.plt_tools.plt_starpar import scatter_sp
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Snapshot_HIH2EM():
    def plt_snapshot_HIH2EM(self, num):

        nr = 2
        nc = 4
        fig, axes = plt.subplots(nr, nc, figsize=(16,21), #constrained_layout=True,
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1,
                                                  height_ratios=[0.75,0.2],
                                                  width_ratios=[0.25,0.25,0.25,0.25]))

        d = self.read_prj(num=num)
        sp = self.load_starpar_vtk(num=num)
        norm = LogNorm(1e-1,1e3)
        norm_EM = LogNorm(1e0,1e4)
        im1 = []
        im2 = []
        im3 = []
        im4 = []
        for i, axis in enumerate(('y','z')):
            extent = d['extent'][axis]
            im1.append(axes[i,0].imshow(d[axis]['Sigma_gas'],
                                         norm=norm, extent=extent, origin='lower', cmap=plt.cm.pink_r))
            im2.append(axes[i,1].imshow(d[axis]['Sigma_H2']+1e-20,
                                         norm=norm, extent=extent, origin='lower', cmap=plt.cm.pink_r))
            im3.append(axes[i,2].imshow(d[axis]['Sigma_HI'],
                                         norm=norm, extent=extent, origin='lower', cmap=plt.cm.pink_r))
            im4.append(axes[i,3].imshow(d[axis]['EM']+1e-20, norm=norm_EM,
                                         cmap='plasma', extent=extent, origin='lower'))

        # Overplot starpar
        for i, axis in enumerate(('y','z')):
            for j in range(nc):
                scatter_sp(sp, axes[i,j], dim=axis, kind='proj', kpc=False, norm_factor=8.0,
                           agemax=40.0, alpha=0.5)

        labels = [r'$\Sigma\;[M_{\odot}\,{\rm pc}^{-2}]$',
                  r'$\Sigma_{\rm H_2}\;[M_{\odot}\,{\rm pc}^{-2}]$',
                  r'$\Sigma_{\rm HI}\;[M_{\odot}\,{\rm pc}^{-2}]$',
                  r'${\rm EM}\;[{\rm cm}^{-6}\,{\rm pc}]$']

        for i,label in enumerate(labels):
            divider = make_axes_locatable(axes[0,i])
            cax = divider.append_axes("top", size="3%", pad=0.2)
            if i<3:
                cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.pink_r),
                                  cax=cax, orientation='horizontal', label=label)
            else:
                cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm_EM, cmap=plt.cm.plasma),
                                  cax=cax, orientation='horizontal', label=label)

            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.xaxis.set_label_position('top')


        plt.suptitle('{0:s}  t={1:.2f}'.format(self.basename, sp.time))
        for ax in axes[:, 1:].flatten():
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

        plt.subplots_adjust(top=0.93)

        savdir = osp.join('/tigress/jk11/figures/TIGRESS-NCR/', self.basename)
        if not osp.exists(savdir):
            os.makedirs(savdir)

        savname = osp.join(savdir, '{0:s}_HIH2EM_{1:04d}.png'.format(self.basename, num))
        plt.savefig(savname, dpi=200, bbox_inches='tight')


        print('saved to ', savname)
        return fig
