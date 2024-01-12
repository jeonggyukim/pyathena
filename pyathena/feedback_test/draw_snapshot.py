#!/usr/bin/env python

import os
import os.path as osp
import time
from mpi4py import MPI

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

import numpy as np
import astropy.units as au
import astropy.constants as ac

import pyathena as pa

from pyathena.util.split_container import split_container
# from ..plt_tools.make_movie import make_movie


def plt_snapshot(s, h, num, axis='y', savdir=None, savfig=False):
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
    #basedir = '/scratch/gpfs/jk11/FEEDBACK-TEST/PHLWRPSN.n100.again2/'
    #basedir = '/scratch/gpfs/jk11/FEEDBACK-TEST/n100.SN.N128'
    basedir = '/scratch/gpfs/jk11/FEEDBACK-TEST/n100.M3E3.PHLWRPWNSN.N128/'
    # s = pa.LoadSimFeedbackTest(basedir, verbose=False)

    s = pa.LoadSimFeedbackTest(basedir, verbose=False)
    h = s.read_hst(force_override=True)
    nums = s.nums

    if COMM.rank == 0:
        print('basedir, nums', s.basedir, nums)
        nums = split_container(nums, COMM.size)
    else:
        nums = None

    mynums = COMM.scatter(nums, root=0)
    print('[rank, mynums]:', COMM.rank, mynums)

    for num in mynums:
        print(COMM.rank, num, end=' ')
        ds = plt_snapshot(s, h, num, savfig=True)

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
