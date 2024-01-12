# pdf.py

import os.path as osp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as au
import astropy.constants as ac
import pandas as pd

from mpl_toolkits.axes_grid1 import ImageGrid

from ..plt_tools.cmap import cmap_apply_alpha
from ..util.scp_to_pc import scp_to_pc
from ..load_sim import LoadSim

class PDF:

    bins=dict(nH=np.logspace(-2,5,71),
              nHI=np.logspace(-2,5,71),
              nH2=np.logspace(-2,5,71),
              nHII=np.logspace(-2,5,71),
              T=np.logspace(0,5,51),
              pok=np.logspace(0,7,71),
              chi_PE_tot=np.logspace(-4,5,91),
              chi_FUV_tot=np.logspace(-4,5,91),
              Bmag=np.logspace(-7,-4,91),
              Erad_LyC=np.logspace(-17,-8,91),
    )

    @LoadSim.Decorators.check_pickle
    def read_pdf2d(self, num,
                   bin_fields=None, bins=None, prefix='pdf2d',
                   savdir=None, force_override=False):

        if bins is not None:
            self.bins = bins

        bin_fields_def = [['nH', 'pok'], ['nH', 'T']]
        if bin_fields is None:
            bin_fields = bin_fields_def

        ds = self.load_vtk(num=num)
        res = dict()

        for bf in bin_fields:
            k = '-'.join(bf)
            res[k] = dict()
            dd = ds.get_field(bf)
            xdat = dd[bf[0]].data.flatten()
            ydat = dd[bf[1]].data.flatten()
            # Volume weighted hist
            weights = None
            H, xe, ye = np.histogram2d(xdat, ydat, (self.bins[bf[0]], self.bins[bf[1]]),
                                       weights=weights)
            res[k]['H'] = H
            res[k]['xe'] = xe
            res[k]['ye'] = ye

            # Density weighted hist
            weights = (ds.get_field('nH'))['nH'].data.flatten()
            Hw, xe, ye = np.histogram2d(xdat, ydat, (self.bins[bf[0]], self.bins[bf[1]]),
                                        weights=weights)
            res[k]['Hw'] = Hw

        res['domain'] = ds.domain

        return res

    @LoadSim.Decorators.check_pickle
    def read_pdf2d_phase(self, num, prefix='pdf2d_phase',
                         savdir=None, force_override=False):
        """
        Read 2d pdf of density, chi_FUV, pok
        """

        r = dict()
        ds = self.load_vtk(num)
        fields = ['nH','xH2','xHII','xHI','pok','T','Bmag','Erad_LyC']

        self.logger.info('Reading fields {0:s}'.format(', '.join(fields)))
        dd = self.get_chi(ds, fields=fields, freq=['LW','PE']) # see ./fields.py

        #bins = (np.logspace(-2,5,71), np.logspace(-4,5,91))
        # Masked array
        idx_HII = dd['xHII'].data.flatten() > 0.5
        idx_HI = (dd['xHI'].data.flatten() > 0.5)
        idx_H2 = (dd['xH2'].data.flatten() > 0.25)
        #idx_HI = ~idx_HII & ~idx_H2

        dat_all = {
            'nH-chi_PE_tot': (dd['nH'].data.flatten(),
                              (dd['chi_PE_ext'] + dd['chi_PE']).data.flatten(),
                              dd['nH'].data.flatten()),
            'nH2-chi_PE_tot': (dd['nH'].data.flatten()[idx_H2],
                               (dd['chi_PE_ext'] + dd['chi_PE']).data.flatten()[idx_H2],
                               dd['nH'].data.flatten()[idx_H2]),
            'nHI-chi_PE_tot': (dd['nH'].data.flatten()[idx_HI],
                               (dd['chi_PE_ext'] + dd['chi_PE']).data.flatten()[idx_HI],
                               dd['nH'].data.flatten()[idx_HI]),
            'nHII-chi_PE_tot': (dd['nH'].data.flatten()[idx_HII],
                                (dd['chi_PE_ext'] + dd['chi_PE']).data.flatten()[idx_HII],
                                dd['nH'].data.flatten()[idx_HII]),

            'nH-chi_FUV_tot': (dd['nH'].data.flatten(),
                               (dd['chi_FUV_ext'] + dd['chi_FUV']).data.flatten(),
                               dd['nH'].data.flatten()),
            'nH2-chi_FUV_tot': (dd['nH'].data.flatten()[idx_H2],
                                (dd['chi_FUV_ext'] + dd['chi_FUV']).data.flatten()[idx_H2],
                                dd['nH'].data.flatten()[idx_H2]),
            'nHI-chi_FUV_tot': (dd['nH'].data.flatten()[idx_HI],
                                (dd['chi_FUV_ext'] + dd['chi_FUV']).data.flatten()[idx_HI],
                                dd['nH'].data.flatten()[idx_HI]),
            'nHII-chi_FUV_tot': (dd['nH'].data.flatten()[idx_HII],
                                 (dd['chi_FUV_ext'] + dd['chi_FUV']).data.flatten()[idx_HII],
                                 dd['nH'].data.flatten()[idx_HII]),

            'nH-pok': (dd['nH'].data.flatten(),
                       dd['pok'].data.flatten(),
                       dd['nH'].data.flatten()),
            'nH2-pok': (dd['nH'].data.flatten()[idx_H2],
                       dd['pok'].data.flatten()[idx_H2],
                        dd['nH'].data.flatten()[idx_H2]),
            'nHI-pok': (dd['nH'].data.flatten()[idx_HI],
                        dd['pok'].data.flatten()[idx_HI],
                        dd['nH'].data.flatten()[idx_HI]),
            'nHII-pok': (dd['nH'].data.flatten()[idx_HII],
                         dd['pok'].data.flatten()[idx_HII],
                         dd['nH'].data.flatten()[idx_HII]),

            'nH-Bmag': (dd['nH'].data.flatten(),
                       dd['Bmag'].data.flatten(),
                       dd['nH'].data.flatten()),
            'nH2-Bmag': (dd['nH'].data.flatten()[idx_H2],
                       dd['Bmag'].data.flatten()[idx_H2],
                        dd['nH'].data.flatten()[idx_H2]),
            'nHI-Bmag': (dd['nH'].data.flatten()[idx_HI],
                        dd['Bmag'].data.flatten()[idx_HI],
                        dd['nH'].data.flatten()[idx_HI]),
            'nHII-Bmag': (dd['nH'].data.flatten()[idx_HII],
                         dd['Bmag'].data.flatten()[idx_HII],
                         dd['nH'].data.flatten()[idx_HII]),

            'nH-T': (dd['nH'].data.flatten(),
                       dd['T'].data.flatten(),
                       dd['nH'].data.flatten()),
            'nH2-T': (dd['nH'].data.flatten()[idx_H2],
                       dd['T'].data.flatten()[idx_H2],
                        dd['nH'].data.flatten()[idx_H2]),
            'nHI-T': (dd['nH'].data.flatten()[idx_HI],
                        dd['T'].data.flatten()[idx_HI],
                        dd['nH'].data.flatten()[idx_HI]),
            'nHII-T': (dd['nH'].data.flatten()[idx_HII],
                         dd['T'].data.flatten()[idx_HII],
                         dd['nH'].data.flatten()[idx_HII]),

            'nH-Erad_LyC': (dd['nH'].data.flatten(),
                            dd['Erad_LyC'].data.flatten(),
                            dd['nH'].data.flatten()),
            'nHII-Erad_LyC': (dd['nH'].data.flatten()[idx_HII],
                              dd['Erad_LyC'].data.flatten()[idx_HII],
                              dd['nH'].data.flatten()[idx_HII]),

        }

        for k, (xdat,ydat,wdat) in dat_all.items():
            r[k] = dict()
            kx, ky = k.split('-')
            bins = (self.bins[kx], self.bins[ky])
            H, xe, ye = np.histogram2d(xdat, ydat, bins=bins, weights=None)
            Hw, _, _ = np.histogram2d(xdat, ydat, bins=bins, weights=wdat)
            r[k]['H'] = H
            r[k]['Hw'] = Hw
            r[k]['xe'] = xe
            r[k]['ye'] = ye

        return r

    @LoadSim.Decorators.check_pickle
    def read_density_pdf_all(self, prefix='density_pdf_all',
                             savdir=None, force_override=False):
        rr = dict()
        # nums = self.nums
        #nums = [0,10,20]
        nums = range(0, self.get_num_max_virial())


        print('density_pdf_all: {0:s} nums:'.format(self.basename), nums, end=' ')

        for i in nums:
            print(i, end=' ')
            r = self.read_density_pdf(num=i, force_override=False)
            if i == 0:
                for k in r.keys():
                    rr[k] = []

            for k in r.keys():
                try:
                    rr[k].append(r[k].value.item())
                except:
                    rr[k].append(r[k])

        rr = pd.DataFrame(rr)
        return rr

    @LoadSim.Decorators.check_pickle
    def read_density_pdf(self, num, prefix='density_pdf',
                         savdir=None, force_override=False):
        """
        Read 1d pdf of density
        """

        bins = np.logspace(-3, 7, 101)
        ds = self.load_vtk(num)
        dd = ds.get_field(['nH','specific_scalar_CL','xn'])
        # Select neutral cloud gas
        idx = np.logical_and(dd['xn'].data > 0.5, dd['specific_scalar_CL'].data > 5e-1)
        nH_cl = (dd['nH']*dd['specific_scalar_CL']).data[idx]
        x = np.log(nH_cl)

        res = dict()
        res['time_code'] = ds.domain['time']

        try:
            res['nH_cl_meanV'] = np.mean(nH_cl)
            res['nH_cl_meanM'] = np.average(nH_cl, weights=nH_cl)
            res['muV'] = np.sum(x)/len(nH_cl)
            res['muM'] = np.sum(x*nH_cl)/np.sum(nH_cl)
            res['sigmaV'] = np.std(x)
            res['sigmaM'] = np.sqrt(np.sum((x - res['muM'])**2*nH_cl)/np.sum(nH_cl))
            res['histV'], res['bineV'] = np.histogram(nH_cl, bins=bins)
            res['histM'], res['bineM'] = np.histogram(nH_cl, bins=bins, weights=nH_cl)

        except ZeroDivisionError:
            pass

        return res

def plt_pdf2d_one_model(s, dt_Myr=[-0.2,2,5,8], yvar='chi_PE_tot', alpha=1.0,
                        force_override=False):
    """Function to plot 2d histograms at different snapshots
    """

    minmax = dict(chi_PE_tot=(1e-4,1e4),
                  chi_FUV_tot=(1e-4,1e4),
                  pok=(1e2,1e7),
                  nH=(1e-2,3e4),
                  T=(1e1,3e4),
                  Bmag=(1e-7,1e-4),
                  Erad_LyC=(1e-4,1e4),
    )

    ylabels = dict(chi_PE_tot=r'$\chi_{\rm PE}$',
                   chi_FUV_tot=r'$\chi_{\rm FUV}$',
                   pok=r'$P/k_{\rm B}\;[{\rm cm}^{-3}\,{\rm K}]$',
                   T=r'$T\,{\rm K}$',
                   Bmag=r'$|\mathbf{B}|\;[\mu{\rm G}]$',
                   Erad_LyC=r'$\mathcal{E}_{\rm LyC}\;[10^{-13}\,{\rm erg}\,{\rm cm}^{-3}]$',
    )

    pcargs = dict(edgecolor='face', linewidth=0, rasterized=True)
    norm = [mpl.colors.LogNorm(1e-6,5e-2),
            mpl.colors.LogNorm(1e-5,5e-2),
            mpl.colors.LogNorm(1e-5,5e-2),
            mpl.colors.LogNorm(1e-5,5e-2)]

    nums = s.get_nums(dt_Myr=dt_Myr)
    cm0 = plt.cm.viridis
    # cm1 = cmap_apply_alpha('Blues')
    # cm2 = cmap_apply_alpha('Greens')
    # cm3 = cmap_apply_alpha('Oranges')
    cm1 = plt.cm.Blues
    cm2 = plt.cm.Greens
    cm3 = plt.cm.Oranges

    fig = plt.figure(figsize=(15, 12))
    nr = 4
    nc = len(dt_Myr)
    imgrid_args = dict(nrows_ncols=(nr,nc), direction='row', aspect=False,
                       label_mode='L', axes_pad=0.2, cbar_mode='edge', cbar_location='right')
    g1 = ImageGrid(fig, [0.02, 0.05, 0.90, 0.90], **imgrid_args)

    for ic,num in enumerate(nums):
        print(num, end=' ')
        rr = s.read_pdf2d_phase(num, force_override=force_override)
        k0 = f'nH-{yvar}'
        k = f'nH-{yvar}'
        im0 = g1[ic].pcolormesh(rr[k]['xe'], rr[k]['ye'],
                                rr[k]['Hw'].T/rr[k0]['Hw'].T.sum(),
                                norm=norm[0], cmap=cm0, alpha=alpha, **pcargs)
        k = f'nH2-{yvar}'
        im1 = g1[nc+ic].pcolormesh(rr[k]['xe'], rr[k]['ye'],
                                   rr[k]['Hw'].T/rr[k0]['Hw'].T.sum(),
                                   norm=norm[1], cmap=cm1, alpha=alpha, **pcargs)
        k = f'nHI-{yvar}'
        im2 = g1[2*nc+ic].pcolormesh(rr[k]['xe'], rr[k]['ye'],
                                     rr[k]['Hw'].T/rr[k0]['Hw'].T.sum(),
                                     norm=norm[2], cmap=cm2, alpha=alpha, **pcargs)

        if yvar == 'chi_FUV_tot':
            k0 = r'nH-Erad_LyC'
            k = r'nHII-Erad_LyC'
            im3 = g1[3*nc+ic].pcolormesh(rr[k]['xe'], rr[k]['ye']*1e13,
                                         rr[k]['Hw'].T/rr[k0]['Hw'].T.sum(),
                                         norm=norm[3], cmap=cm3, alpha=alpha, **pcargs)
        else:
            k = f'nHII-{yvar}'
            im3 = g1[3*nc+ic].pcolormesh(rr[k]['xe'], rr[k]['ye'],
                                         rr[k]['Hw'].T/rr[k0]['Hw'].T.sum(),
                                         norm=norm[3], cmap=cm3, alpha=alpha, **pcargs)

    for i, ax in enumerate(g1):
        if yvar == 'pok':
            # Plot lines of constant temperature 8000/40K for ionized/molecular gas
            nH = np.logspace(np.log10(minmax['nH'][0]), np.log10(minmax['nH'][1]))
            for T,xe,xH2,c,label in zip((20.0,8000.0),(0.0,1.0),\
                                  (0.5,0.0),('blue','orange'),
                                  (r'$T=20\,{\rm K} (x_{\rm H_2}=0.5)$',
                                   r'$T=8000\,{\rm K} (x_{\rm H^+}=1)$')):
                l, = ax.loglog(nH, (1.1 + xe - xH2)*nH*T, c=c,
                               lw=0.75, ls='-', label=label)

        if yvar == 'chi_FUV_tot' and i >= (nr - 1)*nc:
            # Plot lines of constant ionization parameter
            hnui = (s.par['radps']['hnu_PH']*au.eV).cgs.value
            Uion = (1e0, 1e-2, 1e-4)
            nH = np.logspace(np.log10(minmax['nH'][0]), np.log10(minmax['nH'][1]))
            for U in Uion:
                Erad = hnui*U*nH
                ax.loglog(nH, Erad*1e13, c='grey', lw=0.75, ls='--')

            ax.set(xscale='log', yscale='log', xlim=minmax['nH'], ylim=minmax['Erad_LyC'],
                   xlabel=r'$n_{\rm H}\;[{\rm cm^{-3}}]$', ylabel=ylabels['Erad_LyC'])
        else:
            ax.set(xscale='log', yscale='log', xlim=minmax['nH'], ylim=minmax[yvar],
                   xlabel=r'$n_{\rm H}\;[{\rm cm^{-3}}]$', ylabel=ylabels[yvar])
        ax.grid()


    # Annotate time
    for ic, dt_ in zip(range(nc),dt_Myr):
        if dt_ < 0.0:
            g1[ic].set_title(r'$t_{*,0}-$' + r'{0:.1f}'.format(np.abs(dt_)) + r' Myr')
        else:
            g1[ic].set_title(r'$t_{*,0}+$' + r'{0:.1f}'.format(dt_) + r' Myr')

    for i,(im,cm) in enumerate(zip((im0,im1,im2,im3),(cm0,cm1,cm2,cm3))):
        plt.colorbar(im, cax=g1[(i+1)*nc-1].cax, label='mass fraction',
                     norm=norm[i], cmap=cm)

    savefig = True
    if savefig:
        basedir = '/tigress/jk11/figures/GMC/paper/pdf/'
        name = 'pdf2d-{0:s}-{1:s}.png'.format('nH', yvar)
        savname = osp.join(basedir, name)
        fig.savefig(savname, dpi=200, bbox_inches='tight')
        scp_to_pc(savname, target='GMC-AB')
        print('saved to', savname)

    return fig
