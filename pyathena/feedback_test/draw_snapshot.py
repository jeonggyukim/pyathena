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


def draw_snapshot1(s, h, num, axis='y', savdir=None, savfig=False):

    mpl.rcParams['font.size'] = 13
    n0 = s.par['problem']['n0']

    dfi = s.dfi
    fields = ['r','nH','2nH2','nHI','nHII','ne','xe','xHI','2xH2','xHII','T', 'pok',
              'chi_PE', 'chi_LW', 'chi_LW_dust', 'heat_rate','cool_rate']
    
    ds = s.load_vtk(num)
    Lx = ds.domain['Lx'][0]
    slc = ds.get_slice(axis, fields)

    fig, axes = plt.subplots(2, 3, figsize=(15,10))
    axes = axes.flatten()
    
    plt.sca(axes[0])
    for f in ('2nH2','nHI','nHII','ne'):
        plt.scatter(slc['r'], slc[f], s=2.0, label=slc.dfi[f]['label'])
    
    plt.xlim(0, 0.5*2.0**0.5*Lx)
    plt.ylim(1e-6*n0,20.0*n0)
    plt.yscale('log')
    plt.legend(loc=4)

    plt.sca(axes[1])
    for f in ('2xH2','xHI','xHII','xe'):
        plt.scatter(slc['r'], slc[f], s=2.0, label=slc.dfi[f]['label'])
    
    plt.xlim(0, 0.5*2.0**0.5*Lx)
    plt.ylim(1e-6,2.0)
    plt.yscale('log')
    plt.legend(loc=4)

    plt.sca(axes[2])
    for f in ('T','pok'):
        plt.scatter(slc['r'], slc[f], s=2.0, label=slc.dfi[f]['label'])
    plt.xlim(0, 0.5*2.0**0.5*Lx)
    plt.ylim(1e1,3e4*n0)
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
    for f in ('chi_PE','chi_LW', 'chi_LW_dust'):
        plt.scatter(slc['r'], slc[f], s=2.0, label=slc.dfi[f]['label'])
    
    plt.xlim(0, 0.5*2.0**0.5*Lx)
    plt.ylim(1e0,1e6)
    plt.yscale('log')
    plt.legend(loc=1)
    
    # Temperature slice
#     slc['T'].plot.imshow(ax=axes[3], norm=LogNorm(1e1,1e3), 
#                          label=slc.dfi['T']['label'],
#                          cbar_kwargs=dict(label=slc.dfi['T']['label']))
    f = 'nHI'
    slc[f].plot.imshow(ax=axes[3], norm=LogNorm(1e0,1e3), # norm=slc.dfi[f]['norm'], 
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

def draw_snapshot(s, nums, view=True):
    u = pa.Units(kind='LV')
    h = pa.read_hst(s.files['hst'])
    if s.par['configure']['new_cooling'] == 'OFF':
        newcool = False
    else:
        newcool = True
        
    for num in nums:
        print(num, end=' ')
        ds = s.load_vtk(num)
        
        from matplotlib import gridspec
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(nrows=3, ncols=3, wspace=0.5,
                               height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[0,1])
        ax2 = fig.add_subplot(gs[0,2])
        ax3 = fig.add_subplot(gs[1,0])
        ax4 = fig.add_subplot(gs[1,1])
        ax5 = fig.add_subplot(gs[1,2])
        ax6 = fig.add_subplot(gs[2,0])
        ax7 = fig.add_subplot(gs[2,1])
        ax8 = fig.add_subplot(gs[2,2])
        axes = (ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8)
        
        # Plane through which to make slices
        slc = dict(y=0, method='nearest')
        
        # Read data and calculate derived fields
        if newcool:
            d = ds.get_field(['nH', 'nH2', 'temperature', 'pok', 'heat_ratio', 'velocity',
                              'rad_energy_density_PE', 'rad_energy_density_PE_unatt'])
        else:
            d = ds.get_field(['nH', 'temperature', 'pok', 'heat_ratio', 'velocity',
                              'rad_energy_density_PE', 'rad_energy_density_PE_unatt'])
            
        d['AVeff'] = -1.87*np.log(d['rad_energy_density_PE']/d['rad_energy_density_PE_unatt'])
        d['r'] = np.sqrt((d.x**2 + d.y**2 + d.z**2))
        d['vr'] = np.sqrt(d['velocity1']**2+d['velocity2']**2+d['velocity3']**2)

        # Plot slices
        d['nH'].sel(**slc).plot.imshow(ax=ax0, norm=LogNorm(1e-5,1e4), cmap=mpl.cm.Spectral_r)
        d['temperature'].sel(**slc).plot.imshow(ax=ax1, norm=LogNorm(1e1,1e7),
                                   cmap=pa.cmap_shift(mpl.cm.RdYlBu_r, midpoint=3./7.))
        d['pok'].sel(**slc).plot.imshow(ax=ax3, norm=LogNorm(1e1,1e7), cmap=mpl.cm.jet)
        d['heat_ratio'].sel(**slc).plot.imshow(ax=ax4, norm=LogNorm(1e-1,1e4), cmap=mpl.cm.viridis)
        d['vr'].sel(**slc).plot.imshow(ax=ax6, norm=Normalize(-10, 10.0), cmap=mpl.cm.PiYG)
        
        plt.sca(ax2)
        h.plot(x='time', y=['rmom_bub','Rsh', 'Minter', 'Mwarm', 'Mhot', 'Mcold'],
               logy=True, marker='o', markersize=2, ax=ax2)
        plt.ylim(plt.gca().get_ylim()[1]*1e-4, plt.gca().get_ylim()[1])
        plt.axvline(ds.domain['time'], color='grey', linestyle='--')

        plt.sca(ax5)
        plt.hist2d(d['nH'].data.flatten(), d['pok'].data.flatten(),
                   bins=(np.logspace(-3,4,100), np.logspace(1,7,100)), norm=LogNorm());
        cf = pa.classic.cooling.coolftn()
        plt.plot(cf.heat/cf.cool,
                 1.1*cf.get_temp(cf.T1)*cf.heat/cf.cool, c='r', ls=':',label='KI02')
        plt.plot(10.0*cf.heat/cf.cool,
                 1.1*cf.get_temp(cf.T1)*10.0*cf.heat/cf.cool, c='r', ls=':',label='KI02')
        plt.plot(100.0*cf.heat/cf.cool,
                 1.1*cf.get_temp(cf.T1)*100.0*cf.heat/cf.cool, c='r', ls=':',label='KI02')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('nH')
        plt.ylabel('pok')

        # stride for scatter plot
        j = 10
        plt.sca(ax7)
        plt.scatter(d['r'].data.flatten()[::j], d['nH'].data.flatten()[::j],
                    marker='o', s=1.0, alpha=1, c='C0', label='nH')
        plt.scatter(d['r'].data.flatten()[::j], 1e1*d['temperature'].data.flatten()[::j],
                    marker='o', s=1.0, alpha=1, c='C1', label='T*10')
        plt.scatter(d['r'].data.flatten()[::j], d['vr'].data.flatten()[::j],
                        marker='o', s=1.0, alpha=1, c='C2', label='vr')
        plt.scatter(d['r'].data.flatten()[::j], d['AVeff'].data.flatten()[::j],
                    marker='o', s=1.0, alpha=1, c='C3', label='AVeff')
        plt.yscale('log')
        plt.ylim(1e-3, 1e5)
        plt.xlabel('r')
        plt.ylabel('nH, nH2, T*10, AVeff')
        plt.legend(loc=1)

        plt.sca(ax8)
        plt.scatter(d['AVeff'].data.flatten()[::j], d['nH'].data.flatten()[::j],
                    marker='o', s=1.0, alpha=1, c='C0', label='nH')
        if newcool:
            plt.scatter(d['AVeff'].data.flatten()[::j], d['nH2'].data.flatten()[::j],
                        marker='o', s=1.0, alpha=1, c='C2', label='nH2')
        plt.scatter(d['AVeff'].data.flatten()[::j], 1e1*d['temperature'].data.flatten()[::j],
                    marker='o', s=1.0, alpha=1, c='C5', label='T*10')
        plt.scatter(d['AVeff'].data.flatten()[::j], d['heat_ratio'].data.flatten()[::j],
                    marker='o', s=1.0, alpha=1, c='C4', label='heat_ratio')
        plt.yscale('log')
        plt.xlim(0, 15)
        plt.ylim(1e-2, 1e5)
        plt.xlabel('Aveff')
        plt.ylabel('nH, nH2, T*10, heat_ratio')
        plt.legend(loc=1)
        
        for ax in (ax0,ax1,ax3,ax4,ax6):
            ax.set_aspect('equal')

        plt.suptitle(f'{ s.basename }' + 
                     r'  $t$={0:.2f}'.format(d.domain['time']), fontsize='x-large')
        plt.subplots_adjust(top=0.92)

        savdir = osp.join('/tigress/jk11/figures/TIGRESS-FEEDBACK/', s.basename)
        if not osp.exists(savdir):
            os.makedirs(savdir)
        plt.savefig(osp.join(savdir, 'snapshot{0:04d}.png'.format(num)))

        if view:
            break
        else:
            plt.close(plt.gcf())
            
    return ds, savdir

# if __name__ == '__main__':
#     basedir = '/perseus/scratch/gpfs/jk11/FEEDBACK-TEST/n100.M3E3.LWPHRP.N128.test5.dyn/'
#     s = pa.LoadSimFeedbackTest(basedir, verbose=False)
#     for num in s.nums[::2]:
#         print(num, end=' ')
#         ds = draw_snapshot1(s, h, num, savfig=True);
#         # break

if __name__ == '__main__':

    COMM = MPI.COMM_WORLD
    basedir = '/scratch/gpfs/jk11/FEEDBACK-TEST/n100.M3E3.LWPHRP.N128.test5.dyn.tlim2/'
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
        ds = draw_snapshot1(s, h, num, savfig=True)
            
      
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
