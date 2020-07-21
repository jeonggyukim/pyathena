# pdf.py

import os.path as osp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as au
import astropy.constants as ac

from mpl_toolkits.axes_grid1 import ImageGrid

from ..plt_tools.cmap_custom import get_my_cmap
from ..util.scp_to_pc import scp_to_pc
from ..load_sim import LoadSim

class PDF:
    
    bins=dict(nH=np.logspace(-5,3,81),
              nHI=np.logspace(-2,5,71),
              nH2=np.logspace(-2,5,71),
              nHII=np.logspace(-5,3,101),
              T=np.logspace(1,8,141),
              pok=np.logspace(0,7,71),
              chi_PE=np.logspace(-3,4,71),
              chi_FUV=np.logspace(-3,4,71),
              Lambda_cool=np.logspace(-30,-20,101),
              xi_CR=np.logspace(-17,-15,41)
    )
    
    @LoadSim.Decorators.check_pickle
    def read_pdf2d(self, num,
                   bin_fields=None, bins=None, prefix='pdf2d',
                   savdir=None, force_override=False):
        if self.par['configure']['radps'] == 'ON':
            bin_fields_def = [['nH', 'pok'], ['nH', 'T'], ['nH','chi_FUV'],
                              ['T','Lambda_cool'], ['nH','xi_CR']]
        else:
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

        return res

    def plt_pdf2d(self, ax, dat, bf='nH-pok',
                  cmap='cubehelix_r',
                  norm=mpl.colors.LogNorm(1e-6,2e-2),
                  kwargs=dict(alpha=1.0, edgecolor='face', linewidth=0, rasterized=True),
                  weighted=True,
                  xscale='log', yscale='log'):
        
        if weighted:
            hist = 'Hw'
        else:
            hist = 'H'

        ax.pcolormesh(dat[bf]['xe'], dat[bf]['ye'], dat[bf][hist].T/dat[bf][hist].sum(),
                      norm=norm, cmap=cmap, **kwargs)

        kx, ky = bf.split('-')
        ax.set(xscale=xscale, yscale=yscale,
               xlabel=self.dfi[kx]['label'], ylabel=self.dfi[ky]['label'])
        
    
#     @LoadSim.Decorators.check_pickle
#     def read_pdf2d_phase(self, num, prefix='pdf2d_phase',
#                          savdir=None, force_override=False):
#         """
#         Read 2d pdf of density, chi_FUV, pok
#         """
        
#         r = dict()
#         ds = self.load_vtk(num)
#         fields = ['nH','xH2','xHII','xHI','pok','T']
        
#         self.logger.info('Reading fields {0:s}'.format(', '.join(fields)))
#         dd = self.get_chi(ds, fields=fields,
#                           freq=['LW','PE'])

#         bins = (np.logspace(-2,5,71), np.logspace(-4,5,91))
#         # Masked array
#         idx_HII = dd['xHII'].data.flatten() > 0.5
#         idx_HI = (dd['xHI'].data.flatten() > 0.5)
#         idx_H2 = (dd['xH2'].data.flatten() > 0.25)
#         #idx_HI = ~idx_HII & ~idx_H2

#         dat_all = {
#             'nH-chi_PE_tot': (dd['nH'].data.flatten(),
#                               (dd['chi_PE_ext'] + dd['chi_PE']).data.flatten(),
#                               dd['nH'].data.flatten()),
#             'nH2-chi_PE_tot': (dd['nH'].data.flatten()[idx_H2],
#                                (dd['chi_PE_ext'] + dd['chi_PE']).data.flatten()[idx_H2],
#                                dd['nH'].data.flatten()[idx_H2]),
#             'nHI-chi_PE_tot': (dd['nH'].data.flatten()[idx_HI],
#                                (dd['chi_PE_ext'] + dd['chi_PE']).data.flatten()[idx_HI],
#                                dd['nH'].data.flatten()[idx_HI]),
#             'nHII-chi_PE_tot': (dd['nH'].data.flatten()[idx_HII],
#                                 (dd['chi_PE_ext'] + dd['chi_PE']).data.flatten()[idx_HII],
#                                 dd['nH'].data.flatten()[idx_HII]),

#             'nH-chi_FUV_tot': (dd['nH'].data.flatten(),
#                                (dd['chi_FUV_ext'] + dd['chi_FUV']).data.flatten(),
#                                dd['nH'].data.flatten()),
#             'nH2-chi_FUV_tot': (dd['nH'].data.flatten()[idx_H2],
#                                 (dd['chi_FUV_ext'] + dd['chi_FUV']).data.flatten()[idx_H2],
#                                 dd['nH'].data.flatten()[idx_H2]),
#             'nHI-chi_FUV_tot': (dd['nH'].data.flatten()[idx_HI],
#                                 (dd['chi_FUV_ext'] + dd['chi_FUV']).data.flatten()[idx_HI],
#                                 dd['nH'].data.flatten()[idx_HI]),
#             'nHII-chi_FUV_tot': (dd['nH'].data.flatten()[idx_HII],
#                                  (dd['chi_FUV_ext'] + dd['chi_FUV']).data.flatten()[idx_HII],
#                                  dd['nH'].data.flatten()[idx_HII]),

#             'nH-pok': (dd['nH'].data.flatten(),
#                        dd['pok'].data.flatten(),
#                        dd['nH'].data.flatten()),
#             'nH2-pok': (dd['nH'].data.flatten()[idx_H2],
#                        dd['pok'].data.flatten()[idx_H2],
#                         dd['nH'].data.flatten()[idx_H2]),
#             'nHI-pok': (dd['nH'].data.flatten()[idx_HI],
#                         dd['pok'].data.flatten()[idx_HI],
#                         dd['nH'].data.flatten()[idx_HI]),
#             'nHII-pok': (dd['nH'].data.flatten()[idx_HII],
#                          dd['pok'].data.flatten()[idx_HII],
#                          dd['nH'].data.flatten()[idx_HII]),                   
#                   }

#         for k, (xdat,ydat,wdat) in dat_all.items():
#             r[k] = dict()
#             kx, ky = k.split('-')
#             bins = (self.bins[kx], self.bins[ky])
#             H, xe, ye = np.histogram2d(xdat, ydat, bins=bins, weights=None)
#             Hw, _, _ = np.histogram2d(xdat, ydat, bins=bins, weights=wdat)
#             r[k]['H'] = H
#             r[k]['Hw'] = Hw
#             r[k]['xe'] = xe
#             r[k]['ye'] = ye

#         return r    


# def plt_pdf2d_one_model(s, dt_Myr=[-0.2,2,5,8], yvar='chi_PE_tot', alpha=1.0, force_override=False):
#     """Function to plot 2d histograms at different snapshots
#     """
    
#     minmax = dict(chi_PE_tot=(1e-4,1e4),
#                   chi_FUV_tot=(1e-4,1e4),
#                   pok=(1e2,1e7),
#                   nH=(1e-2,3e4))

#     ylabels = dict(chi_PE_tot=r'$\chi_{\rm PE}$',
#                    chi_FUV_tot=r'$\chi_{\rm FUV}$',
#                    pok=r'$P/k_{\rm B}\;[{\rm cm}^{-3}\,{\rm K}]$')
    
#     pcargs = dict(edgecolor='face', linewidth=0, rasterized=True)
#     norm = mpl.colors.LogNorm(1e-6,2e-2)
    
#     nums = s.get_nums(dt_Myr=dt_Myr)
#     cm0 = plt.cm.viridis
#     cm1 = get_my_cmap('Blues')
#     cm2 = get_my_cmap('Greens')
#     cm3 = get_my_cmap('Oranges')

#     fig = plt.figure(figsize=(15, 12))
#     nr = 4
#     nc = len(dt_Myr)
#     imgrid_args = dict(nrows_ncols=(nr,nc), direction='row', aspect=False,
#                        label_mode='L', axes_pad=0.2, cbar_mode='edge', cbar_location='right')
#     g1 = ImageGrid(fig, [0.02, 0.05, 0.90, 0.90], **imgrid_args)

#     for ic,num in enumerate(nums):
#         print(num, end=' ')
#         rr = s.read_pdf2d_phase(num, force_override=force_override)
#         k0 = f'nH-{yvar}'
#         k = f'nH-{yvar}'
#         im0 = g1[ic].pcolormesh(rr[k]['xe'], rr[k]['ye'],
#                                 rr[k]['Hw'].T/rr[k0]['Hw'].T.sum(),
#                                 norm=norm, cmap=cm0, alpha=alpha, **pcargs)
#         k = f'nH2-{yvar}'
#         im1 = g1[nc+ic].pcolormesh(rr[k]['xe'], rr[k]['ye'],
#                                    rr[k]['Hw'].T/rr[k0]['Hw'].T.sum(),
#                                    norm=norm, cmap=cm1, alpha=alpha, **pcargs)
#         k = f'nHI-{yvar}'
#         im2 = g1[2*nc+ic].pcolormesh(rr[k]['xe'], rr[k]['ye'],
#                                      rr[k]['Hw'].T/rr[k0]['Hw'].T.sum(),
#                                      norm=norm, cmap=cm2, alpha=alpha, **pcargs)
#         k = f'nHII-{yvar}'
#         im3 = g1[3*nc+ic].pcolormesh(rr[k]['xe'], rr[k]['ye'],
#                                      rr[k]['Hw'].T/rr[k0]['Hw'].T.sum(),
#                                      norm=norm, cmap=cm3, alpha=alpha, **pcargs)

#     for ax in g1:
#         ax.set(xscale='log', yscale='log', xlim=minmax['nH'], ylim=minmax[yvar],
#                xlabel=r'$n_{\rm H}\;[{\rm cm^{-3}}]$', ylabel=ylabels[yvar])
#         ax.grid()
        
#     # Annotate time
#     for ic, dt_ in zip(range(nc),dt_Myr):
#         if dt_ < 0.0:
#             g1[ic].set_title(r'$t_{*,0}-$' + r'{0:.1f}'.format(np.abs(dt_)) + r' Myr')
#         else:
#             g1[ic].set_title(r'$t_{*,0}+$' + r'{0:.1f}'.format(dt_) + r' Myr')

#     for i,(im,cm) in enumerate(zip((im0,im1,im2,im3),(cm0,cm1,cm2,cm3))):
#         plt.colorbar(im, cax=g1[(i+1)*nc-1].cax, label='mass fraction',
#                      norm=norm, cmap=cm)

#     basedir='/tigress/jk11/figures/GMC/paper/pdf/'
#     name = 'pdf2d-{0:s}-{1:s}-{2:s}.png'.format(s.basename, 'nH', yvar)
#     savname = osp.join(basedir, name)
#     fig.savefig(savname, dpi=200, bbox_inches='tight')
#     scp_to_pc(savname, target='GMC-MHD-Results')
#     print('saved to', savname)

#     return fig
