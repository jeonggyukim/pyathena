import os
import os.path as osp
from shutil import copyfile
import getpass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, LogNorm
from ..plt_tools.cmap import Colormaps
from ..plt_tools.plt_starpar import scatter_sp
from ..plt_tools.make_movie import make_movie

mycm = Colormaps().cm

class PltSnapshotCombined:

    def make_movie_snapshot_combined(self, fps=15, basedir=None, savdir=None):
        if basedir is None:
            basedir = self.basedir

        if savdir is None:
            savdir = osp.join(basedir, 'movies')
            if not osp.exists(savdir):
                os.makedirs(savdir)

        fin = osp.join(basedir, 'snapshot-combined/*.png')
        fout = osp.join(savdir, '{0:s}-combined.mp4'.format(self.basename))
        if make_movie(fin, fout, fps, fps):
            savdir2='/tigress/{0:s}/public_html/movies/SF-CLOUD/'.\
                format(getpass.getuser())
            fout2 = osp.join(savdir2, osp.basename(fout))
            copyfile(fout, fout2)
            print('Copied movie file to {0:s}'.format(fout2))

    def plt_snapshot_combined(self, num, dim='y', fields=['nH', 'T', 'pok', 'Bx', 'Bz'],
                              pos=0.0, zoom=1.0, savfig=True):
        """Plot slices, projections, pdf, and mass history

        Parameters
        ----------
        pos : float or str
            If float, plot slices through the position.
            If 'star', plot slices through the most massive starpar.
        """

        if self.par['radps']['irayt'] == 0:
            noUV = True
        else:
            noUV = False

        if self.par['radps']['iPhotIon'] == 0:
            noPhotIon = True
        else:
            noPhotIon = False

        if self.par['configure']['gas'] == 'hydro':
            fields=['nH', 'T', 'pok']
            #fields.remove('Bx')
            #fields.remove('Bz')

        ds = self.load_vtk(num)
        sp = self.load_starpar_vtk(num)
        if sp.empty or isinstance(pos, float) or isinstance(pos, int):
            x0,y0 = (0.0,0.0)
        else:
            spmm = sp[sp['mass'] == sp['mass'].max()]
            pos = float(spmm['x3'])
            if dim == 'z':
                x0, y0 = float(spmm['x1']), float(spmm['x2'])
            elif dim == 'x':
                x0, y0 = float(spmm['x2']), float(spmm['x3'])
            elif dim == 'y':
                x0, y0 = float(spmm['x1']), float(spmm['x3'])

            self.logger.info('Most massive star x,y,z:({0:.2f},{0:.2f},{0:.2f})'.\
                format(float(spmm['x1']),float(spmm['x2']),float(spmm['x3'])))

        fig, axes = plt.subplots(3, 3, figsize=(20,15), constrained_layout=True)
        axes = axes.flatten()

        # Half the box size
        hL = 0.5*self.domain['Lx'][0] / zoom

        # Plot slices
        dd = ds.get_slice(dim, fields, pos)
        if zoom > 1.0:
            if dim == 'z':
                dd = dd.where((dd.x - x0 > -hL) & (dd.y - y0 > -hL) & \
                              (dd.x - x0 <  hL) & (dd.y - y0 <  hL), drop=True)
            elif dim == 'y':
                dd = dd.where((dd.z - x0 > -hL) & (dd.x - y0 > -hL) & \
                              (dd.z - x0 <  hL) & (dd.x - y0 <  hL), drop=True)
            elif dim == 'x':
                dd = dd.where((dd.y - x0 > -hL) & (dd.z - y0 > -hL) & \
                              (dd.y - x0 <  hL) & (dd.z - y0 <  hL), drop=True)

        for f in fields:
            if f == 'nH':
                dd['nH'].plot.imshow(ax=axes[0], norm=LogNorm(),
                                     cmap='Spectral_r', add_labels=False, extend='neither',
                                     cbar_kwargs=dict(label=r'$n_{\rm H}\;[{\rm cm}^{-3}]$'))
            elif f == 'T':
                dd['T'].plot.imshow(ax=axes[1], norm=LogNorm(),
                                    #norm=Normalize(5e3,3e4),
                                    cmap=mycm['T'], add_labels=False, extend='neither',
                                    cbar_kwargs=dict(label=r'$T\;[{\rm K}]$'))
            elif f == 'pok':
                dd['pok'].plot.imshow(ax=axes[2], norm=LogNorm(),
                                      cmap='inferno', add_labels=False, extend='neither',
                                      cbar_kwargs=dict(label=r'$P/k_{\rm B}\;[{\rm cm}^{-3}\,{\rm K}]$'))
            elif f == 'Bx':
                dd['Bx'].plot.imshow(ax=axes[3], norm=Normalize(-20,20),
                                     cmap='bwr', add_labels=False, extend='neither',
                                     cbar_kwargs=dict(label=r'$B_x$'))

            elif f == 'Bz':
                dd[f].plot.imshow(ax=axes[4], norm=Normalize(-20,20),
                                  cmap='bwr', add_labels=False, extend='neither',
                                  cbar_kwargs=dict(label=r'$B_z$'))

            # f = 'xHII'
            # dd[f].plot.imshow(ax=axes[3], norm=Normalize(0,1),
            #                   cmap='viridis', add_labels=False, extend='neither',
            #                   cbar_kwargs=dict(label=r'$x_{\rm HII}$'))
            # f = 'xH2'
            # (2.0*dd[f]).plot.imshow(ax=axes[4], norm=Normalize(0,1),
            #                         cmap='viridis', add_labels=False, extend='neither',
            #                         cbar_kwargs=dict(label=r'$2x_{\rm H_2}$'))
        # elif noPhotIon:
        #     f = 'xH2'
        #     (2.0*dd[f]).plot.imshow(ax=axes[3], norm=Normalize(0,1),
        #                             cmap='viridis', add_labels=False, extend='neither',
        #                             cbar_kwargs=dict(label=r'$2x_{\rm H_2}$'))
        #     f = 'Erad_FUV'
        #     if dd[f].max() != 0.0:
        #         dd[f].plot.imshow(ax=axes[4], norm=LogNorm(1e-12,1e-7),
        #                           cmap='viridis', add_labels=False, extend='neither',
        #       cbar_kwargs=dict(label=r'$\mathcal{E}_{\rm FUV}\,[{\rm erg}\,{\rm cm}^{-3}]$'))

        # else:
        #     f = 'Erad_LyC'
        #     if dd[f].max() != 0.0:
        #         dd[f].plot.imshow(ax=axes[3], norm=LogNorm(1e-12,1e-7),
        #                           cmap='viridis', add_labels=False, extend='neither',
        #       cbar_kwargs=dict(label=r'$\mathcal{E}_{\rm LyC}\,[{\rm erg}\,{\rm cm}^{-3}]$'))
        #     f = 'Erad_FUV'
        #     if dd[f].max() != 0.0:
        #         dd[f].plot.imshow(ax=axes[4], norm=LogNorm(1e-12,1e-7),
        #                           cmap='viridis', add_labels=False, extend='neither',
        #       cbar_kwargs=dict(label=r'$\mathcal{E}_{\rm FUV}\,[{\rm erg}\,{\rm cm}^{-3}]$'))

        # MASS HISTORY
        h = self.read_hst()
        axes[5].plot(h['tau'],h['M_sp'], label=r'$M_{\ast}$')
        axes[5].plot(h['tau'],h['M_H2_cl'], label=r'$M_{\rm H_2}$')
        axes[5].plot(h['tau'],h['M_cl_neu'], label=r'$M_{\rm cl,neu}$')
        axes[5].plot(h['tau'],h['M_cl_of'], ls='--', label=r'$M_{\rm cl,of}$')
        axes[5].axvline(ds.domain['time']*self.u.Myr/self.cl.tff.value, c='grey', lw=0.5)
        axes[5].legend()
        plt.setp(axes[5], xlabel=r'$t/t_{\rm ff,0}$',
                 ylabel=r'${\rm mass}\;[M_{\odot}]$')

        # PROJECTIONS
        d = ds.get_field(['nH','nesq','T'])
        conv_Sigma = (self.u.density*self.u.length).to('Msun pc-2')*ds.domain['dx'][0]
        conv_EM = ds.domain['dx'][0]
        d['Sigma'] = d['nH'].sum(dim=dim)*conv_Sigma
        d['EM'] = d['nesq'].sum(dim=dim)*conv_EM
        d['Sigma'].plot.imshow(ax=axes[-3], norm=LogNorm(),
                               cmap='pink_r', add_labels=False, extend='neither',
                               cbar_kwargs=dict(label=r'$\Sigma\,[M_{\odot}\,{\rm pc}^{-2}]$'))
        d['EM'].plot.imshow(ax=axes[-2], norm=LogNorm(),
                            cmap='plasma', add_labels=False, extend='neither',
                            cbar_kwargs=dict(label=r'${\rm EM}\,[{\rm cm}^{-6}\,{\rm pc}]$'))

        # Overplot star particles
        dt_output = self.get_dt_output()
        num_ratio = int(dt_output['vtk']/dt_output['vtk_sp'])
        num_sp = num*num_ratio
        sp = self.load_starpar_vtk(num_sp)
        if not sp.empty:
            scatter_sp(sp, axes[-2], dim=dim, kind='prj',
                       norm_factor=2.5, agemax=3.0*self.cl.tff.value)

        ax = axes[-1]
        # hb = ax.hexbin(d['nH'].data.flatten(), d['T'].data.flatten(),
        #                norm=LogNorm(), gridsize=30, # bins='log',
        #                xscale='log', yscale='log', mincnt=1,
        #                reduce_C_function=np.sum,
        #                C=d['nH'].data.flatten())
        if self.par['feedback']['iSN'] != 0 or self.par['feedback']['iWind'] != 0:
            nmin = 1e-3
            nmax = 1e3*self.cl.nH.value
            nbins = np.logspace(np.log10(nmin),np.log10(nmax),151)
            Tbins = np.logspace(0, 7, 141)
        else:
            nbins = np.logspace(-3,3,151)*self.cl.nH.value
            Tbins = np.logspace(0, 5, 101)

        hb = ax.hist2d(d['nH'].data.flatten(), d['T'].data.flatten(),
                       norm=LogNorm(),
                       bins=(nbins, Tbins),
                       weights=d['nH'].data.flatten())
        # ax.set_xlabel(r'$n_{\rm H}\,[{\rm cm}^{-3}]$')
        # ax.set_ylabel(r'$T$ [K]')
        # ax.set_xscale('log')
        # ax.set_yscale('log')

        plt.setp(ax, xscale='log', yscale='log',
                 xlabel=r'$n_{\rm H}\,[{\rm cm}^{-3}]$',
                 ylabel=r'$T$ [K]')
        #cb = fig.colorbar(hb, ax=axes[-1])

        for i,ax in enumerate(axes[0:-1]):
            if i == 5:
                continue
            ax.set_aspect('equal')
            plt.sca(ax)
            #plt.tick_params(axis='both', labelsize=0, length = 0)
            plt.tick_params(top='off', bottom='off', left='off', right='off',
                            labelleft='off', labelbottom='off')

            # xticks = [-0.5*lx,-0.25*lx,0.0,0.25*lx,0.5*lx]

        plt.suptitle(self.basename + r' $t$={0:4.2f}'.format(self.domain['time']) +\
                     r' $t/t_{\rm ff,0}$'+'={0:4.2f}'.format(ds.domain['time']*
                                                             self.u.Myr/self.cl.tff.value))
        # plt.tight_layout()

        if savfig:
            savdir = osp.join(self.basedir, 'snapshot-combined')
            if not osp.exists(savdir):
                os.makedirs(savdir)

            fout = osp.join(savdir, '{0:s}-combined-{1:04d}.png'.\
                            format(self.basename, num))
            plt.savefig(fout, dpi=200)

            savdir2='/tigress/{0:s}/public_html/movies/SF-CLOUD/{1:s}'.\
                format(getpass.getuser(), self.basename)
            if not osp.exists(savdir2):
                os.makedirs(savdir2)

            fout2 = osp.join(savdir2, '{0:s}-combined-{1:04d}.png'.\
                             format(self.basename, num))
            copyfile(fout, fout2)

        return fig,d,dd
