import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import ImageGrid
import astropy.units as au
import astropy.constants as ac

from ..plt_tools.cmap import cmap_apply_alpha, get_cmap_parula
from ..io.read_vtk import read_vtk
from ..plt_tools.set_plt import toggle_xticks,toggle_yticks
from ..plt_tools.plt_starpar import scatter_sp, colorbar_sp, legend_sp
from ..plt_tools.utils import texteffect
from ..util.scp_to_pc import scp_to_pc

from .load_sim_sf_cloud import load_all_alphabeta
from .slc_prj import SliceProj

cm_parula = get_cmap_parula()
cmap = dict(Sigma=cm_parula, Sigma_HI=cm_parula, Sigma_H2=cm_parula,
            EM='plasma',
            nHn=cmap_apply_alpha('Blues'),
            nH2=cmap_apply_alpha('Blues'),
            nHI=cmap_apply_alpha('Greens'),
            nHII=cmap_apply_alpha('Oranges'),
            T=plt.cm.jet,
            chi_PE=plt.cm.viridis,
            Erad_LyC=plt.cm.viridis,
            Erad_FUV=plt.cm.viridis
            )

label = dict(Sigma=r'$\Sigma\;[M_{\odot}\,{\rm pc}^{-2}]$',
             Sigma_H2=r'$\Sigma_{\rm H_2}\;[M_{\odot}\,{\rm pc}^{-2}]$',
             Sigma_HI=r'$\Sigma_{\rm HI}\;[M_{\odot}\,{\rm pc}^{-2}]$',
             EM=r'${\rm EM}\;[{\rm cm}^{-6}\,{\rm pc}]$',
             nHn=r'$n_{\rm H^0}+2n_{\rm H_2}\;[{\rm cm}^{-3}]$',
             nHI=r'$n_{\rm H^0}\;[{\rm cm}^{-3}]$',
             nHII=r'$n_{\rm H^+}\;[{\rm cm}^{-3}]$',
             T=r'$T\;[{\rm K}]$',
             chi_PE=r'$\chi_{\rm PE}$',
             Erad_LyC=r'$\mathcal{E}_{\rm LyC}\;[{\rm erg}\,{\rm cm}^{-3}]$',
             Erad_FUV=r'$\mathcal{E}_{\rm FUV}\;[{\rm erg}\,{\rm cm}^{-3}]$'
             )

norm = dict(Sigma=LogNorm(1e0,2e3),Sigma_HI=LogNorm(1e0,2e3),
            Sigma_H2=LogNorm(1e0,2e3), EM=LogNorm(3e1,3e5),
            nHn=LogNorm(1e-2,3e3),
            nHI=LogNorm(1e-2,3e3),
            nHII=LogNorm(1e-2,3e3),
            T=LogNorm(1e1,2e4),
            chi_PE=LogNorm(1e-2,1e4),
            Erad_FUV=LogNorm(5e-16,5e-11),
            Erad_LyC=LogNorm(5e-16,5e-11)
           )

class PltSnapshot(object):
    def __init__(self, norm_factor=1.0, agemax_sp=10.0, cmap_sp=plt.cm.OrRd_r):

        self.sa, self.r = load_all_alphabeta(force_override=False)
        self.norm_factor = norm_factor
        self.agemax_sp = agemax_sp
        self.cmap_sp = cmap_sp

    def plt_models(self, models, labels=None,
                   dt_Myr=[1.0,2.0,4.0,8.0], nums=None, dim='y',
                   field='Sigma', vtk_type='vtk_2d', title=None,
                   force_override=False):
        """Compare snapshots of different models at different times

        Parameters
        ----------
        dt_Myr : array-like
            (time - time of first SF) of snapshots in Myr
        vtk_type : str
            'vtk' for 3d vtk output and 'vtk2d' for 2d vtk output
        """

        # Obtain snapshot numbers for vtk[_2d] and starpar_vtk files
        if nums is not None:
            import copy
            tmp = copy.deepcopy(nums)
            nums = dict()
            nums_sp = dict()
            for mdl in models:
                nums[mdl] = tmp
                nums_sp[mdl] = tmp

            suffix = 'nums'
        elif dt_Myr is not None:
            nums = dict()
            nums_sp = dict()
            for mdl in models:
                s = self.sa.set_model(mdl)
                dt_output = s.get_dt_output()
                num_ratio = int(dt_output[vtk_type]/dt_output['vtk_sp'])
                nums[mdl] = s.get_nums(dt_Myr=dt_Myr, output=vtk_type)
                nums_sp[mdl] = nums[mdl]*num_ratio

            suffix = 'dt'

        # print(nums)
        # print(nums_sp)

        plt.rcParams['xtick.bottom'] = True
        plt.rcParams['ytick.left'] = True
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True

        # Create axes
        nr = len(models)
        nc = len(list(nums.values())[0])
        im = np.empty((nr,nc), dtype=mpl.image.AxesImage)
        self.fig = plt.figure(figsize=(nr*4+2, nc*4+2))
        g1 = ImageGrid(self.fig, [0.02, 0.05, 0.92, 0.94], (nr, nc), direction='row',
                       axes_pad=0.04, aspect=True, share_all=False, label_mode='all',
                       cbar_mode=None, cbar_size='2%', cbar_pad='1%')

        s = self.sa.set_model(models[0])
        extent = SliceProj.get_extent(s.domain)
        f = field

        if vtk_type == 'vtk_2d':
            dimtoi = dict(x=0,y=1,z=2)
            ii = dimtoi[dim] + 1
            if f == 'Sigma':
                ff = f'Sigma{ii}'
                conv = ((s.u.muH*1.008*ac.u.cgs/au.cm**3).to('Msun/pc**3')).value

        for ir,mdl in enumerate(models): # varying row
            print(mdl, end=' ')
            s = self.sa.set_model(mdl)
            for ic,(num,num_sp) in enumerate(zip(nums[mdl], nums_sp[mdl])):
                dd = read_vtk(s.files[ff][num])
                d = dd.get_field(ff).squeeze()
                ax = g1[ir*nc + ic]
                im[ir,ic] = ax.imshow(d[ff].data*conv, extent=extent[dim], norm=norm[f], cmap=cmap[f], origin='lower')
                sp = s.load_starpar_vtk(num_sp)
                if not sp.empty:
                    scatter_sp(sp, ax, dim=dim, kind='prj', cmap=self.cmap_sp,
                               norm_factor=self.norm_factor, agemax=self.agemax_sp)

        #self.set_ticks_and_labels(g1[nc*(nr-1)], dim, extent[dim])
        # toggle_xticks(g1[nc*(nr-1)], visible=False)
        # toggle_yticks(g1[nc*(nr-1)], visible=False)
        # self.set_ticks_and_labels(g1[-1], dim, extent[dim], right=True)

        for ax in g1:
            ax.xaxis.tick_bottom()
            ax.yaxis.tick_right()
            toggle_xticks(ax)
            toggle_yticks(ax)
        self.set_ticks_and_labels(g1[-1], dim, extent[dim], bottom=True, right=True)

        if labels is not None:
            for ir in range(nr):
                g1[ir*nc].annotate(labels[ir], (-0.09,0.5),
                                   ha='center', va='center',
                                   xycoords='axes fraction', rotation=90, fontsize=20)

        # Need to save figure before making colorbar axes
        self.savefig(name='snapshot-{0:s}-{1:s}-{2:s}.png'.\
                     format('-'.join(models), dim, suffix, dpi=50))
        bb = g1[nc-1].get_position(original=False)
        cax = self.fig.add_axes([bb.x0+0.16,bb.y0,0.01,bb.y1-bb.y0])
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap['Sigma'],
                                       norm=norm['Sigma'], orientation='vertical')

        cb.set_label(r'$\Sigma\;[M_{\odot}\,{\rm pc^{-2}}]$', fontsize=14)

        # Add starpar colorbar and legend
        norm_sp = Normalize(vmin=0., vmax=self.agemax_sp)
        bb = g1[2*nc-1].get_position(original=False)
        cax_sp = self.fig.add_axes([bb.x0+0.16,bb.y0+0.5*(bb.y1-bb.y0)-0.02,0.01,0.5*(bb.y1-bb.y0)])
        cb_sp = mpl.colorbar.ColorbarBase(cax_sp, cmap=self.cmap_sp, norm=norm_sp,
                                          orientation='vertical', extend='neither',
                                          ticks=[0, self.agemax_sp/2.0, self.agemax_sp])
        cb_sp.set_label(r'${\rm age}\;[{\rm Myr}]$', fontsize=14)
        cb_sp.ax.set_yticklabels(['0', '4', '8'])
        legend_sp(g1[3*nc-1], norm_factor=self.norm_factor, mass=[1e2, 1e3],
                  location='right', fontsize='medium',
                  bbox_to_anchor=dict(top=(0.08, 1.02), right=(bb.x0+0.15, bb.y0+0.06)))

        # Annotate time
        if dt_Myr is not None:
            for ic, dt_ in zip(range(nc), dt_Myr):
                if dt_ < 0.0:
                    dt_ = -dt_
                    sign = r'$t_{*,0}-$'
                else:
                    sign = r'$t_{*,0}+$'
                g1[ic].annotate(sign + r'{0:g}'.format(dt_) + r'Myr',
                                   (0.5, 1.025), ha='center', xycoords='axes fraction',
                                   **texteffect(fontsize=20))

        if title is not None:
            plt.suptitle(title, 0.5, 1.05, ha='center', va='bottom')

        self.savefig(name='snapshot-{0:s}-{1:s}-{2:s}.png'.format('-'.join(models), dim, suffix))


    def plt_model(self, model='B2S4', dt_Myr=[1.0,2.0,4.0,8.0], nums=None, dim='y',
                  fields=['Sigma','Sigma_HI','Sigma_H2','EM'],
                  prj=[True, True, True, True],
                  plt_sp=[True, True, True, True], norm_factor=1.0,
                  agemax_sp=10.0, cmap_sp=plt.cm.OrRd_r,
                  force_override=False):
        """Compare snapshots of different models at different times

        Parameters
        ----------
        dt_Myr : array-like
            (time - time of first SF) of snapshots in Myr
        """

        s = self.sa.set_model(model)
        self.rr = self.r.loc[model]

        dt_output = s.get_dt_output()
        num_ratio = int(dt_output['vtk']/dt_output['vtk_sp'])

        print('Model:', model)
        if nums is None:
            nums = s.get_nums(dt_Myr=dt_Myr)

        nr = len(fields)
        nc = len(nums)
        im = np.empty((nr,nc), dtype=mpl.image.AxesImage)
        self.fig = plt.figure(figsize=(nc*3.8+2, nr*3.5+2))
        g1 = ImageGrid(self.fig, [0.02, 0.05, 0.92, 0.94], (nr, nc), direction='row',
                       axes_pad=0.04, aspect=True, share_all=False, label_mode='1',
                       cbar_mode='edge', cbar_size='4%', cbar_pad='1%')
        self.cbar = []

        for ic,num in enumerate(nums): # varying column
            print(num, end=' ')
            # Read data
            dat = dict()
            if np.array(prj).sum() > 0:
                dat['prj'] = s.read_prj(num, force_override=force_override)
            if (~np.array(prj)).sum() > 0:
                dat['slc'] = s.read_slc(num, force_override=force_override)

            sp = s.load_starpar_vtk(num*num_ratio)
            for ir, f in enumerate(fields): # varying row
                ax = g1[ir*nc + ic]
                if prj[ir]:
                    kind = 'prj'
                else:
                    kind = 'slc'

                im[ir,ic] = ax.imshow(dat[kind][dim][f], extent=dat[kind]['extent'][dim],
                                      norm=norm[f], cmap=cmap[f], origin='lower')
                if f == 'nHI' or f == 'nHn':
                    imHII = ax.imshow(dat[kind][dim]['nHII'], extent=dat[kind]['extent'][dim],
                                      norm=norm['nHII'], cmap=cmap['nHII'], origin='lower')

                # Scatter plot starpar
                if plt_sp[ir]:
                    if not sp.empty:
                        scatter_sp(sp, ax, dim=dim, kind=kind, cmap=cmap_sp,
                                   norm_factor=norm_factor, agemax=agemax_sp)
                else:
                    pass

                if ic == nc - 1:
                    # Add colorbars
                    bb = g1[nc*(ir+1)-1].get_position(original=False)
                    cb = plt.colorbar(im[ir,nc-1], g1[nc*(ir+1)-1].cax,
                                      cmap=cmap[f], norm=norm[f], label=label[f])
                    # if f == 'nHI' or f == 'nHn':
                    # cb.set_label(label[f], labelpad=-3)
                    self.cbar.append(cb)

                    if f == 'nHI' or f == 'nHn':
                        # Add another colorbar for nHn and nHII
                        # Don't understand why (probably a bug) why bbox doesn't have correct valuess
                        # But it is okay when savefig is called.
                        self.savefig(name='snapshot-{0:s}-{1:s}.png'.format(model, dim))

                        # print(self.cbar[-1].ax.get_position())
                        bb = self.cbar[-1].ax.get_position()
                        cax = self.fig.add_axes([bb.x0+0.06,bb.y0,bb.x1-bb.x0,bb.y1-bb.y0])
                        plt.colorbar(imHII, cax,
                                     cmap=cmap['nHII'], norm=norm['nHII'], label=label['nHII'])
                        # cb.set_label(label['nHII'], labelpad=-3)

        # Add starpar legend
        colorbar_sp(self.fig, agemax_sp, bbox=[0.28, 1.0, 0.1, 0.008], cmap=cmap_sp)
        legend_sp(g1[0], norm_factor=norm_factor, mass=[1e2, 1e3],
                  location='top', fontsize='medium',
                  bbox_to_anchor=dict(top=(0.08, 1.02), right=(0.48, 0.90)))

        # Annotate time
        for ic, dt_ in zip(range(nc),dt_Myr):
            g1[ic].annotate(r'$t=t_{*,0}+$' + r'{0:.1f}'.format(dt_) + r'Myr',
                            (0.05, 0.9), ha='left', xycoords='axes fraction',
                            **texteffect(fontsize=20))

        self.set_ticks_and_labels(g1[nc*(nr-1)], dim, dat[kind]['extent'][dim])
        self.savefig(name='snapshot-{0:s}-{1:s}.png'.format(model, dim))

    @staticmethod
    def set_ticks_and_labels(ax, dim, extent, bottom=True, right=False):
        xmin, xmax = extent[0], extent[1]
        ymin, ymax = extent[2], extent[3]
        ax.set_xticks([xmin,xmin/2.0,0,xmax/2.0,xmax])
        ax.set_yticks([ymin,ymin/2.0,0,ymax/2.0,ymax])
        ax.set_xticklabels([int(x) for x in [xmin,xmin/2.0,0,xmax/2.0,xmax]])
        ax.set_yticklabels([int(y) for y in [ymin,ymin/2.0,0,ymax/2.0,ymax]])
        xlabel = dict(x=r'$y\;[{\rm pc}]$',y=r'$x\;[{\rm pc}]$',z=r'$x\;[{\rm pc}]$',)
        ylabel = dict(x=r'$z\;[{\rm pc}]$',y=r'$z\;[{\rm pc}]$',z=r'$y\;[{\rm pc}]$',)
        ax.set_xlabel(xlabel[dim])
        ax.set_ylabel(ylabel[dim])
        if bottom:
            ax.xaxis.tick_bottom()
            ax.xaxis.set_label_position("bottom")
        if right:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

    def savefig(self, name, basedir='/tigress/jk11/figures/GMC/paper/snapshot/', dpi=200):
        # Save figure
        savname = osp.join(basedir, name)
        self.fig.savefig(savname, dpi=dpi, bbox_inches='tight')
        scp_to_pc(savname, target='GMC-AB')
        print('saved to', savname)
        return fig
