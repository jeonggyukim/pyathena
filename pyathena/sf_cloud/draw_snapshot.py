#!/usr/bin/env python

import os
import time
import pprint
import gc
from mpi4py import MPI
import os.path as osp

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

import numpy as np
import astropy.units as au
import astropy.constants as ac

import pyathena as pa

from ..util.split_container import split_container
# from ..plt_tools.make_movie import make_movie

from ..fields.xray_emissivity import XrayEmissivityIntegrator, get_xray_emissivity
from astropy.visualization import make_lupton_rgb

def legend_sp(ax, norm_factor, mass=[1e2, 1e3], location="top", fontsize='medium',
              bbox_to_anchor=None):
    """
    Add legend for sink particle mass
    mass: sink particle mass in solar mass
    location: "top" or "right"
    """
    if bbox_to_anchor is None:
        bbox_to_anchor = dict(top=(0.1, 0.95),
                              right=(0.88, 0.83))
    else:
        if location not in bbox_to_anchor:
            raise(
                "bbox_to_anchor[localtion] must be a tuple specifying legend location")

    ext = ax.images[0].get_extent()

    ss = []
    labels = []
    for m in mass:
        label = r"$10^{0:g}\;M_\odot$".format(np.log10(m))
        s = ax.scatter(ext[1]*2, ext[3]*2,
                       s=np.sqrt(m)/norm_factor, lw=1.0,
                       color='k', alpha=1.0, label=label, facecolors='k')
        ss.append(s)
        labels.append(label)

    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[3], ext[2])
    if location == 'top':
        legend = ax.legend(ss, labels, scatterpoints=1, fontsize=fontsize,
                           ncol=len(ss), frameon=False, loc=2,
                           bbox_to_anchor=bbox_to_anchor[location],
                           bbox_transform=plt.gcf().transFigure,
                           columnspacing=0.1, labelspacing=0.1, handletextpad=0.05)
        # for t in legend.get_texts():
        #         t.set_va('bottom')
    else:
        legend = ax.legend(ss, labels, scatterpoints=1, fontsize=fontsize,
                           frameon=False, loc=2,
                           columnspacing=0.02, labelspacing=0.02, handletextpad=0.02,
                           bbox_to_anchor=bbox_to_anchor[location],
                           bbox_transform=plt.gcf().transFigure)

    return legend


def plt_snapshot_2panel(s, num, dim='z', agemax_sp=8.0, savdir=None, savfig=False):

    from ..classic.plot_tools.scatter_sp import scatter_sp
    from ..plt_tools.cmap import cmap_apply_alpha

    # Read data
    ds = s.load_vtk(num)
    fields = ['nH2','nHI']
    if s.par['feedback']['iWind'] > 0:
        fields += ['j_X']
    if s.par['configure']['ionrad'] == 'ON':
        fields += ['nesq']

    dd = ds.get_field(fields) # read 3d fields nH2 = xH2*nH, nHI = xHI*nH
    dfi = dd.dfi # dictionary containing derived field info
    sp = s.load_starpar_vtk(num) # read starpar vtk as pandas DataFrame

    dim_to_idx = dict(z=2,y=1,x=0)
    idx = dim_to_idx[dim]
    if dim == 'z':
        lx = s.domain['Lx'][0]
        ly = s.domain['Lx'][1]
    elif dim == 'y':
        lx = s.domain['Lx'][2]
        ly = s.domain['Lx'][0]
    elif dim == 'x':
        lx = s.domain['Lx'][1]
        ly = s.domain['Lx'][2]

    xticks = [-0.5*lx,-0.25*lx,0.0,0.25*lx,0.5*lx]
    yticks = [-0.5*ly,-0.25*ly,0.0,0.25*ly,0.5*ly]


    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Get neutral gas surface density in Msun/pc^2
    to_Sigma = (s.u.density*ds.domain['dx'][idx]*s.u.length).to('Msun/pc2').value
    dd['Sigma_neu'] = to_Sigma*(2.0*dd.sum(dim=dim)['nH2'] + dd.sum(dim=dim)['nHI'])
    # Plot using xarray imshow
    im0 = dd['Sigma_neu'].plot.imshow(ax=axes[0], cmap='pink_r', norm=LogNorm(1e-1,1e3),
                                      extend='neither', add_labels=False,
                                      xticks=xticks, yticks=yticks,
                                      cbar_kwargs=dict(label=r'$\Sigma_{\rm n}\;[M_{\odot}\,{\rm pc}^{-2}]$'))

    # X-ray intensity
    # j_X = X-ray volume emissivity in the 0.5-7.0keV band [erg/s/cm^3]
    # X-ray intensity: I_X = 1/(4*pi) * \int j_X ds
    if s.par['feedback']['iWind'] > 0:
        dd['I_X'] = 1.0/(4.0*np.pi)*dd['j_X'].sum(dim=dim)*dd.domain['dx'][2]*s.u.length.cgs.value
        im1 = dd['I_X'].plot.imshow(ax=axes[1], norm=LogNorm(1e-10,1e-6), extend='neither',cmap='Blues',
                                    add_labels=False, xticks=xticks, yticks=yticks,
                 cbar_kwargs=dict(label=r'$I_X\;[{\rm erg\,{\rm cm}^{-2}\,{\rm s}^{-1}\,{\rm sr}^{-1}}]$'))
    else:
        dd['EM'] = dd['nesq'].sum(dim=dim)*dd.domain['dx'][2]
        im1 = dd['EM'].plot.imshow(ax=axes[1], norm=LogNorm(1e1,1e5), extend='neither',cmap='plasma',
                                   add_labels=False, xticks=xticks, yticks=yticks,
                 cbar_kwargs=dict(label=r'${\rm EM}\;[{\rm cm^{-6}\,{\rm pc}}]$'))

    for ax in axes:
        ax.set_aspect('equal')

    # Scatter plot star particles (color: age, mass: size)
    if not sp.empty:
        for ax in (axes[0],axes[1]):
            scatter_sp(sp, ax, axis=0, norm_factor=1.0,
                       type='proj', kpc=False, runaway=True, agemax=agemax_sp)
            extent = (ds.domain['le'][0], ds.domain['re'][0])
            ax.set_xlim(*extent)
            ax.set_ylim(*extent)

    # Add starpar age colorbar
    norm = mpl.colors.Normalize(vmin=0., vmax=agemax_sp)
    cmap = mpl.cm.cool_r
    cax = fig.add_axes([0.125, 0.9, 0.1, 0.015])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                   orientation='horizontal', ticks=[0, 4, 8])
    # cbar_sp.ax.tick_params(labelsize=14)
    cb.set_label(r'${\rm age}\;[{\rm Myr}]$', fontsize=14)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    # Add legends for starpar mass
    legend_sp(axes[0], norm_factor=1.0, mass=[1e2, 1e3], location='top', fontsize='medium',
              bbox_to_anchor=dict(top=(0.22, 0.97),
                                  right=(0.48, 0.91)))

    # Set simulation time [code] as suptitle
    plt.suptitle(r'time={0:.2f}'.format(ds.domain['time']), x=0.5, y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

    if savfig:
        if savdir is None:
            savdir = osp.join('/tigress/jk11/figures/GMC', s.basename)
        if not osp.exists(savdir):
            os.makedirs(savdir)

        plt.savefig(osp.join(savdir, 'snapshot_2panel_{0:04d}.png'.format(num)), dpi=200)
        #plt.close(fig)

    return ds,dd,sp,axes,(im0,im1)


def draw_snapshot_rgb(s, num, ax, minimum=20.0, stretch=15000, Q=15):
    ds = s.load_vtk(num)
    dd = ds.get_field(['nH2','nHI','nH','j_X', 'nesq'])
    dd['NH_neu'] = (2.0*dd['nH2'].sum(dim='z') + 2.0*dd['nHI'].sum(dim='z'))*dd.domain['dx'][2]
    dd['I_X'] = dd['j_X'].sum(dim='z')*dd.domain['dx'][2]*s.u.length.cgs.value
    dd['EM'] = dd['nesq'].sum(dim='z')*dd.domain['dx'][2]

    # Make r and g to have the same maximum
    #norm_b = dd['NH_neu'].data.max()/dd['I_X'].data.max()
    #norm_g = dd['NH_neu'].data.max()/dd['EM'].data.max()
    #norm_b = 1.0
    #norm_g = 1.0
    norm_b = 1625343998.93
    norm_g = 0.04300
    r = dd['NH_neu'].data
    b = dd['I_X'].data*norm_b
    g = dd['EM'].data*norm_g
    #print(r.max(),g.max(),b.max(),r.min(),g.min(),b.min())
    #print(norm_b,norm_g)

    image = make_lupton_rgb(r, g, b, minimum=minimum, stretch=stretch, Q=Q)
    ax.imshow(image, interpolation='bicubic', origin='lower')

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xticklabels(''); ax.set_yticklabels('')

    return ax, dd, image


def plt_snapshot2(s, num, axis='z', savdir=None, savfig=False):

    from ..classic.plot_tools.scatter_sp import scatter_sp
    from ..plt_tools.cmap import cmap_apply_alpha

    cm1 = cmap_apply_alpha('Blues')
    cm2 = cmap_apply_alpha('Greens')
    cm3 = cmap_apply_alpha('Oranges')

    ds = s.load_vtk(num)
    dd = ds.get_field(['nH2','nHI','nHII','ne','T'])
    #dd = ds.get_field(['nH2','nHI','nHII','ne','T','j_X'])
    dfi = dd.dfi
    sp = s.load_starpar_vtk(num*10)

    fig, axes = plt.subplots(2,2,figsize=(15, 12))
    axes = axes.flatten()

    # Neutral gas surface density
    to_surf = (s.u.density*ds.domain['dx'][0]*s.u.length).to('Msun/pc2').value
    dd['Sigma_neu'] = to_surf*(2.0*dd.sum(axis=0)['nH2']+dd.sum(axis=0)['nHI'])
    dd['Sigma_neu'].plot.imshow(ax=axes[0], cmap='pink_r', norm=LogNorm(1e0,1e3), extend='neither',
               add_labels=False, xticks=[-40,-20,0,20,40], yticks=[-40,-20,0,20,40],
               cbar_kwargs=dict(label=r'$\Sigma_{\rm neu}\;[M_{\odot}\,{\rm pc}^{-2}]$'))

    # Emission measure
    ((dd['ne']**2).sum(axis=0)*ds.domain['dx'][2]).plot.\
        imshow(ax=axes[1],cmap='plasma',norm=LogNorm(1e1,1e5), extend='neither',
               add_labels=False, xticks=[-40,-20,0,20,40],yticks=[-40,-20,0,20,40],
               cbar_kwargs=dict(label=r'${\rm EM}\equiv\int n_e^2 d\ell\;[{\rm cm}^{-6}\,{\rm pc}]$'))

    # Density slices
    slc = dd.sel(**{axis:0.0},method='nearest')
    im = slc['nHI'].plot.imshow(ax=axes[2], cmap=cm2, norm=LogNorm(1e-2,1e4),
                                add_labels=False,add_colorbar=True, extend='neither')
    im.colorbar.remove()
    im = slc['nH2'].plot.imshow(ax=axes[2], cmap=cm1, norm=LogNorm(1e-2,1e4),
                                add_labels=False,add_colorbar=False, extend='neither')
    im = slc['nHII'].plot.imshow(ax=axes[2], cmap=cm3, norm=LogNorm(1e-2,1e4),
                                 add_labels=False,add_colorbar=False, extend='neither')
    im = slc['T'].plot.imshow(ax=axes[3], cmap=dfi['T']['cmap'], norm=LogNorm(1e1,1e6),
                              add_labels=False, add_colorbar=True)
    im.colorbar.remove()

    # # X-ray intensity
    # dd['I_X'] = 1.0/(4.0*np.pi)*dd['j_X'].sum(dim='z')*dd.domain['dx'][2]*s.u.length.cgs.value
    # dd['I_X'].plot.imshow(ax=axes[2], norm=LogNorm(1e-10,1e-6),
    #                       extend='neither',cmap='Blues',
    #                       add_labels=False, xticks=[-40,-20,0,20,40],yticks=[-40,-20,0,20,40],
    #                       cbar_kwargs=dict(label=r'$I_X\;[{\rm erg\,{\rm cm}^{-2}\,{\rm s}^{-1}}]$'))

    # im = dd['I_X'].plot.imshow(ax=axes[3], norm=LogNorm(1e-10,1e-6),
    #                        extend='neither',cmap='Blues',
    #                        add_labels=False, xticks=[-40,-20,0,20,40],yticks=[-40,-20,0,20,40],
    #                        cbar_kwargs=dict(label=r'$I_X\;[{\rm erg\,{\rm cm}^{-2}\,{\rm s}^{-1}}]$'))
    # im.colorbar.remove()
    # _, _, _ = draw_snapshot_rgb(s, num, axes[3], minimum=0., stretch=500, Q=20)

    d = ds.get_field(['nH','pok'])
    axes[3].hist2d(d['nH'].data.flatten(),
                   d['pok'].data.flatten(),
                   bins=(np.logspace(np.log10(1e-3),np.log10(5e4),100),
                         np.logspace(2,7,100)),
                   norm=LogNorm(), weights=d['nH'].values.flatten())
    plt.xlabel(dfi['nH']['label']); plt.ylabel(dfi['pok']['label'])
    plt.xscale('log'); plt.yscale('log')
    plt.xlim(1e-3,5e4) ; plt.ylim(1e0,1e8)

    for ax in axes[:-1]:
        ax.set_aspect('equal')

    if not sp.empty:
        for ax in (axes[0],axes[1]):
            scatter_sp(sp, ax, axis=0, norm_factor=1.0,
                       type='prj', kpc=False, runaway=True, agemax=10.0)
            extent = (ds.domain['le'][0], ds.domain['re'][0])
            ax.set_xlim(*extent)
            ax.set_ylim(*extent)

    plt.suptitle(r'time={0:5.1f}'.format(ds.domain['time']),x=0.5,y=0.95)

    #plt.tight_layout()
    #plt.subplots_adjust(top=0.94)

    if savfig:
        if savdir is None:
            savdir = osp.join('/tigress/jk11/figures/GMC', s.basename)
        if not osp.exists(savdir):
            os.makedirs(savdir)

        plt.savefig(osp.join(savdir, 'snapshot_{0:04d}.png'.format(num)), dpi=200)
        plt.close(fig)

    return ds,axes,im

def plt_snapshot(s, h, num, axis='y', savdir=None, savfig=True):
    mpl.rcParams['font.size'] = 13
    n0 = s.par['problem']['n0']

    dfi = s.dfi
    if s.par['configure']['radps'] == 'ON':
        fields = ['r','nH','2nH2','nHI','nHII','ne','xe','xHI','2xH2','xHII','T', 'pok',
                  'chi_PE', 'chi_LW', 'chi_LW_dust', 'heat_rate','cool_rate']
        new_cool = True
    elif s.par['configure']['new_cooling'] == 'ON':
        fields = ['r','nH','2nH2','nHI','nHII','ne','xe','xHI','2xH2','xHII','T', 'pok',
                  'heat_rate','cool_rate']
        new_cool = True
    else:
        fields = ['r','nH','T', 'pok','heat_rate','cool_rate']
        new_cool = False

    ds = s.load_vtk(num)
    Lx = ds.domain['Lx'][0]
    slc = ds.get_slice(axis, fields)

    fig, axes = plt.subplots(2, 3, figsize=(15,10))
    axes = axes.flatten()

    plt.sca(axes[0])
    if new_cool:
        fields = ('2nH2','nHI','nHII','ne')
    else:
        fields = ('nH',)
    for f in fields:
        plt.scatter(slc['r'], slc[f], s=2.0, label=slc.dfi[f]['label'])

    plt.xlim(0, 0.5*2.0**0.5*Lx)
    plt.ylim(1e-6*n0,20.0*n0)
    plt.yscale('log')
    plt.legend(loc=4)

    plt.sca(axes[1])
    if new_cool:
        fields = ('2xH2','xHI','xHII','xe')
    else:
        fields = ('nH',)
    for f in fields:
        plt.scatter(slc['r'], slc[f], s=2.0, label=slc.dfi[f]['label'])

    plt.xlim(0, 0.5*2.0**0.5*Lx)
    plt.ylim(1e-6,2.0)
    plt.yscale('log')
    plt.legend(loc=4)

    plt.sca(axes[2])
    for f in ('T','pok'):
        plt.scatter(slc['r'], slc[f], s=2.0, label=slc.dfi[f]['label'])
    plt.xlim(0, 0.5*2.0**0.5*Lx)
    plt.ylim(1e1,1e7)
    plt.yscale('log')
    plt.legend(loc=4)

    plt.sca(axes[4])
    for f in ('cool_rate','heat_rate'):
        plt.scatter(slc['r'], slc[f], s=2.0, label=f)

    plt.xlim(0, 0.5*2.0**0.5*Lx)
    plt.ylim(1e-25,1e-18)
    plt.yscale('log')
    plt.legend(loc=1)

    plt.sca(axes[5])
#     d = ds.get_slice(axis, ['r','nH','xHI','xHII','xe','rad_energy_density_PH','T'])
#     hnu_PH = (s.par['radps']['hnu_PH']*au.eV).cgs.value
#     sigma_PH = s.par['radps']['sigma_HI_PH']
#     d['xi_ph'] = ac.c.cgs.value*d['rad_energy_density_PH']*s.u.energy_density.value/hnu_PH*sigma_PH
#     d['tphoti'] = d['xi_ph']*d['xHI']
#     d['treci'] = d['xHII']*d['nH']*d['xe']*2.59e-13*(d['T']*1e-4)**-0.7
#     d['tHII'] = 1.0/np.abs(d['tphoti'] - d['treci'])
#    plt.scatter(d['r'],d['tHII'],s=2.0,label=['tHII'])
#    plt.yscale('log')

    d = ds.get_field(['nH','cool_rate','T'])
    plt.hist2d(d['nH'].data.flatten(),
               d['T'].data.flatten(),
               bins=(np.logspace(np.log10(n0*1e-5),np.log10(n0*1e2),100),
                     np.logspace(1,7,100)),
               norm=LogNorm(), weights=d['nH'].values.flatten())
    plt.xlabel(dfi['nH']['label']); plt.ylabel(dfi['T']['label'])
    plt.xscale('log'); plt.yscale('log')
    plt.xlim(1e-5*n0,1e2*n0) ; plt.ylim(1e1,1e7)

#     for f in ('chi_PE','chi_LW', 'chi_LW_dust'):
#         plt.scatter(slc['r'], slc[f], s=2.0, label=slc.dfi[f]['label'])

#     plt.xlim(0, 0.5*2.0**0.5*Lx)
#     #plt.ylim(1e0,1e6)
#     plt.ylim(1e5,1e15)
#     plt.yscale('log')
#     plt.legend(loc=1)

    # Temperature slice
#     slc['T'].plot.imshow(ax=axes[3], norm=LogNorm(1e1,1e3),
#                          label=slc.dfi['T']['label'],
#                          cbar_kwargs=dict(label=slc.dfi['T']['label']))
    f = 'nH'
    slc[f].plot.imshow(ax=axes[3], norm=LogNorm(1e-4,1e4), # norm=slc.dfi[f]['norm'],
                         label=slc.dfi[f]['label'],cmap='Spectral_r',
                         cbar_kwargs=dict(label=slc.dfi[f]['label']))
    axes[3].set_aspect('equal')

    plt.suptitle(f'Model: {s.basename}  ' + \
             r'num={0:2d} time={1:f}'.format(num, ds.domain['time']))
    print('time:',ds.domain['time'])
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    if savfig:
        if savdir is None:
            savdir = osp.join('/tigress/jk11/figures/FEEDBACK-TEST', s.basename)
        if not osp.exists(savdir):
            os.makedirs(savdir)

        plt.savefig(osp.join(savdir, 'slice_{0:04d}.png'.format(num)), dpi=200)

    return ds

if __name__ == '__main__':

    COMM = MPI.COMM_WORLD

    #basedir = '/scratch/gpfs/jk11/GMC/M1E5R20.R.Binf.A2.N128.again/'
    basedir = '/scratch/gpfs/jk11/GMC/M1E5R20.WS.Binf.A2.N128/'
    s = pa.LoadSimSFCloud(basedir, verbose=False)
    #h = s.read_hst(force_override=True)
    nums = s.nums

    if COMM.rank == 0:
        print('basedir, nums', s.basedir, nums)
        nums = split_container(nums, COMM.size)
    else:
        nums = None

    mynums = COMM.scatter(nums, root=0)
    print('[rank, mynums]:', COMM.rank, mynums)

    time0 = time.time()
    for num in mynums:
        print(COMM.rank, num, end=' ')
        #ds = plt_snapshot2(s, num, savfig=True)
        ds = plt_snapshot_2panel(s, num, savfig=True)
        n = gc.collect()
        print('Unreachable objects:', n)
        print('Remaining Garbage:', end=' ')
        pprint.pprint(gc.garbage)

    # # Make movies
    # if COMM.rank == 0:
    #     fin = osp.join(s.basedir, 'snapshots2/*.png')
    #     fout = osp.join(s.basedir, 'movies/{0:s}_snapshots2.mp4'.format(s.basename))
    #     make_movie(fin, fout, fps_in=15, fps_out=15)
    #     from shutil import copyfile
    #     copyfile(fout, osp.join('/tigress/jk11/public_html/movies',
    #                             osp.basename(fout)))

    COMM.barrier()
    if COMM.rank == 0:
        print('')
        print('################################################')
        print('# Do tasks')
        print('# Execution time [sec]: {:.1f}'.format(time.time()-time0))
        print('################################################')
        print('')
