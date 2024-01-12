#!/usr/bin/env python

import os
import sys
import time

import os.path as osp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpi4py import MPI

from .load_sim_tigress_single_sn import LoadSimTIGRESSSingleSNAll
from ..util.split_container import split_container
from ..util.units import Units
from ..classic.cooling import coolftn

def plt_rprofiles(ds0, ds1, dat0, dat1, r1d=None, j=100):
    # import pyathena as pa
    u = Units()
    cf = coolftn()

    muH = 1.4271
    nHeq = cf.heat/cf.cool
    Peq = cf.T1*nHeq*muH

    fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                             gridspec_kw=dict(hspace=0.1), constrained_layout=True)
    axes = axes.flatten()

    for dat in (dat0, dat1):
        if r1d is None:
            x = np.tile(dat0.x.data[:, None, None], (1, ds0.domain['Nx'][1], ds0.domain['Nx'][2]))
            y = np.tile(dat0.y.data[None, :, None], (ds0.domain['Nx'][0], 1, ds0.domain['Nx'][2]))
            z = np.tile(dat0.z.data[None, None, :], (ds0.domain['Nx'][0], ds0.domain['Nx'][1], 1))
            r3d = np.sqrt(x**2 + y**2 + z**2)
            r1d = r3d.flatten()

        d = dat['density'].data.flatten()
        p = dat['pressure'].data.flatten()*u.pok
        c = dat['cool_rate'].data.flatten()
        T = dat['temperature'].data.flatten()

        plt_sty = dict(s=2.0, alpha=0.2)
        plt.sca(axes[0])
        plt.scatter(r1d[::j], d[::j], marker='o', **plt_sty)

        plt.sca(axes[1])
        plt.scatter(r1d[::j], p[::j], marker='o', **plt_sty)

        plt.sca(axes[2])
        plt.scatter(d[::j], p[::j], marker='o', **plt_sty)

        plt.sca(axes[3])
        plt.scatter(r1d[::j], c[::j]/d[::j]**2, marker='o', **plt_sty)

        plt.sca(axes[4])
        plt.scatter(r1d[::j], T[::j], marker='o', **plt_sty)

        plt.sca(axes[5])
        plt.scatter(T[::j], c[::j]/d[::j]**2, marker='o', **plt_sty)

    for ax in (axes[0],axes[1],axes[3],axes[4]):
        plt.sca(ax)
        plt.xlabel(r'$r\;[{\rm pc}]$')
        plt.xscale('linear')
        plt.yscale('log')
        plt.xlim(0, ds0.domain['Lx'][0]/2.5)

    plt.sca(axes[0])
    plt.ylim(5e-4, 2e1)
    plt.ylabel(r'$n_{\rm H}$')

    plt.sca(axes[1])
    plt.ylim(1e1, 5e7)
    plt.ylabel(r'$P/k_{\rm B}\;[{\rm cm}^{-3}\,{\rm K}]$')

    plt.sca(axes[2])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(5e-4, 3e3)
    plt.ylim(1e1, 5e7)
    plt.xlabel(r'$n_{\rm H}$')
    plt.ylabel(r'$P/k_{\rm B}\;[{\rm cm}^{-3}\,{\rm K}]$')
    plt.plot(nHeq, Peq, c='grey', alpha=0.7)

    plt.sca(axes[3])
    plt.ylim(1e-27, 1e-18)
    plt.ylabel(r'$\Lambda(T)\;[{\rm cm^3\,s^{-1}}]$')

    plt.sca(axes[4])
    plt.ylim(1e2, 1e7)
    plt.ylabel(r'$T\;[{\rm K}]$')

    plt.sca(axes[5])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e1, 5e7)
    plt.ylim(1e-27, 1e-18)
    plt.xlabel(r'$T\;[{\rm K}]$')
    plt.ylabel(r'$\Lambda(T)\;[{\rm cm^3\,s^{-1}}]$')
    plt.plot(nHeq, Peq, c='grey', alpha=0.7)

    plt.tight_layout()
    plt.suptitle('{0:s} {1:s} time:{2:5.3f}'.format(s0.basename, s1.basename,
                                                    ds0.domain['time']))
    #plt.subplots_adjust(top=0.95)

    return fig, r1d



if __name__ == '__main__':

    COMM = MPI.COMM_WORLD
    sa = LoadSimTIGRESSSingleSNAll()

    mpl.rcParams['font.size'] = 14
    savdir = '/tigress/jk11/notebook/TIGRESS-SINGLE-SN/snapshots_new'
    # if not osp.exists(savdir):
    #     os.makedirs(savdir)

    fields = ['density', 'pressure', 'cool_rate', 'temperature']
    sa = LoadSimTIGRESSSingleSNAll()
    s0 = sa.set_model(sa.models[0]) # old cooling
    s1 = sa.set_model(sa.models[1]) # new cooling
    r1d = None
    nums = s0.nums_id0
    nums = split_container(nums, COMM.size)

    mynums = COMM.scatter(nums, root=0)
    print('[rank, mynums]:', COMM.rank, mynums)

    if COMM.rank == 0:
        print('nums', nums)
    else:
        nums = None

    # Measure execution time
    time0 = time.time()
    for num in mynums:
        print(num, end=' ')
        ds0 = s0.load_vtk(num=num)
        ds1 = s1.load_vtk(num=num)
        dat0 = ds0.get_field(fields, as_xarray=True)
        dat1 = ds1.get_field(fields, as_xarray=True)
        fig, r1d = plt_rprofiles(ds0, ds1, dat0, dat1, r1d=r1d, j=100)
        fig.savefig(osp.join(savdir, 'rprofiles_{0:s}.{1:s}.{2:04d}.png'.format(s0.basename, s1.basename, num)),
                    dpi=200)

    COMM.barrier()
    if COMM.rank == 0:
        print('')
        print('################################################')
        print('# Done with model', model)
        print('# Execution time [sec]: {:.1f}'.format(time.time()-time0))
        print('################################################')
        print('')
