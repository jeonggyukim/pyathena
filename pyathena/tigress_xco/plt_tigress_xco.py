#!/usr/bin/env python

import os
import sys
import time

import os.path as osp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import xarray as xr
from mpi4py import MPI

from .load_sim_tigress_xco import LoadSimTIGRESSXCOAll
from ..util.split_container import split_container
from ..plt_tools.plt_joint_pdf import plt_joint_pdf

#field_def = ['density', 'xH2', 'CR_ionization_rate']

field_def = ['density', 'xH2', 'CR_ionization_rate',
             'rad_energy_density0', 'rad_energy_density_PE',
             'rad_energy_density2', 'rad_energy_density3', 'rad_energy_density4',
             'rad_energy_density_LW', 'rad_energy_density_PE_unatt']

def read_data(sa, model, num,
              field=field_def, zmin=-256.0, zmax=256.0):
    sa.set_model(model)
    s = sa.sim
    ds = s.load_vtk(num=num)
    # Read data
    dat = ds.get_field(field=field, as_xarray=True)
    dat['nH2'] = 2.0*dat.xH2*dat.density
    dat = dat.where(np.logical_and(dat.z < zmax, dat.z > zmin), drop=True)

    #dat = dict(dat)

    # dat['taueff'] = -np.log(dat.rad_energy_density_PE/dat.rad_energy_density_PE_unatt)
    # Mask where taueff = inf with tau_eff_max
    # taueff_max = 10.0
    # dat['taueff'] = xr.where(dat['taueff'] == np.inf, taueff_max, dat['taueff'])

    return s, ds, dat

def plt_pdf_density_CRIR(sa, model, num, dat=None, gs=None, savfig=True):

    if dat is None:
        s, ds, dat = read_data(sa, model, num)

    s = sa.set_model(model)
    ds = s.load_vtk(num=num)

    x = dat['density'].values.flatten()
    y = dat['CR_ionization_rate'].values.flatten()
    hexbin_args = dict(xscale='log', yscale='log', mincnt=1, gridsize=30)
    ax1, ax2, ax3 = plt_joint_pdf(x, y, hexbin_args, weights=x, gs=gs)
    ax1.set_xlabel(r'$n_{\rm H}$')
    ax1.set_ylabel(r'$\xi_{\rm CR}$')
    ax1.set_xlim(1e-3, 1e4)
    ax2.set_xlim(1e-3, 1e4)

    # Set CRIR range
    h = s.read_hst()
    ylim = (h.xi_CR0.iloc[0]*1e-2, h.xi_CR0.iloc[0]*2.0)
    ax1.set_ylim(*ylim)
    ax3.set_ylim(*ylim)
    #ax1.set_ylim(3e-17, 1e-15)
    #ax3.set_ylim(3e-17, 1e-15)
    plt.suptitle('{0:s}, time: {1:.1f}'.format(s.basename, ds.domain['time']))

    if savfig:
        savdir = osp.join('./figures-pdf')
        if not os.path.exists(savdir):
            os.makedirs(savdir)
        plt.savefig(osp.join(savdir, 'pdf-density-CRIR.{0:s}.{1:04d}.png'.format(model, ds.num)))

    return plt.gcf()

def plt_pdf_density_xH2(sa, model, num, dat=None, gs=None, savfig=True):

    if dat is None:
        s, ds, dat = read_data(sa, model, num)

    s = sa.set_model(model)
    ds = s.load_vtk(num=num)

    x = dat['density'].values.flatten()
    y = dat['xH2'].values.flatten()
    hexbin_args = dict(xscale='log', yscale='linear', mincnt=1, gridsize=50,
                       norm=mpl.colors.LogNorm())
    ax1, ax2, ax3 = plt_joint_pdf(x, y, hexbin_args, weights=x, gs=gs)
    ax1.set_xlabel(r'$n_{\rm H}$')
    ax1.set_ylabel(r'$x_{\rm H_2}$')
    ax1.set_xlim(1e-3, 1e4)
    ax2.set_xlim(1e-3, 1e4)
    ax1.set_ylim(0, 0.55)
    ax3.set_ylim(0, 0.55)

    def calc_xH2_equil(n, xi_H=2.0e-16, R_gr=3.0e-17, zeta=5.7e-11):
        a = 2.31*xi_H
        b = -2.0*R_gr*n - 4.95*xi_H - zeta
        c = n*R_gr
        return (-b - np.sqrt(b*b - 4.0*a*c))/(2.0*a)

    n = np.logspace(-3, 4)
    h = s.read_hst()
    xH2eq = calc_xH2_equil(n, h.xi_CR0.iloc[num-1], # num-1 because the first row is delted
                           R_gr=3.0e-17*s.par['problem']['R_gr_amp'], zeta=0.0)

    ax1.semilogx(n, xH2eq, 'r--')
    plt.suptitle('{0:s}, time: {1:.1f}'.format(s.basename,ds.domain['time']))

    if savfig:
        savdir = osp.join('./figures-pdf')
        if not os.path.exists(savdir):
            os.makedirs(savdir)
        plt.savefig(osp.join(savdir, 'pdf-density-xH2.{0:s}.{1:04d}.png'.format(model, ds.num)))

    return plt.gcf()


def plt_hst_mass(mhd_model='R2_2pc'):

    sa = LoadSimTIGRESSXCOAll()

    fig, axes = plt.subplots(3, 1, figsize=(12, 15),
                             sharex=True)

    i = 0
    for mdl in sa.models:
        if not mdl.startswith(mhd_model):
            continue

        s = sa.set_model(mdl, verbose=False)
        h = s.read_hst(merge_mhd=True, force_override=True)
        hmhd = s.read_hst_mhd()
        plt.sca(axes[0])
        if i == 0:
            label = 'total'
            plt.plot(h.time, h.Sigma_gas, 'k-', lw=2, label=label)
        else:
            label = '_nolegend_'
        plt.plot(h.time, h.Sigma_H - h.Sigma_H2, 'o-', label=mdl)
        plt.sca(axes[1])
        plt.plot(h.time, h.Sigma_H2/h.Sigma_H, 'o-', label=mdl)
        plt.sca(axes[2])
        plt.plot(h.time, h.xi_CR0, 'o-')

        i += 1

    plt.sca(axes[0])
    plt.ylabel(r'$\Sigma_{\rm HI} [Msun/pc^2]$')
    plt.legend(loc=1)
    plt.yscale('log')
    plt.title('Gas surface density')
    plt.grid()
    plt.gca().grid(which='minor', alpha=0.2)
    plt.gca().grid(which='major', alpha=0.5)

    # H2 fraction
    plt.sca(axes[1])
    plt.ylim(3e-2, 1)
    plt.yscale('log')
    plt.title('H2 mass fraction')
    plt.ylabel(r'$M_{\rm H_2}/M_{\rm H,tot}$')
    plt.grid()
    plt.gca().grid(which='minor', alpha=0.2)
    plt.gca().grid(which='major', alpha=0.5)

    # CRIR
    plt.sca(axes[2])
    plt.yscale('log')
    plt.title('CRIR')
    plt.xlabel('time [Myr]')
    plt.ylabel(r'$\xi_{\rm CR,0}\;[{\rm s}^{-1}]$')
    plt.grid()
    plt.gca().grid(which='minor', alpha=0.2)
    plt.gca().grid(which='major', alpha=0.5)

    dtime = h.time.iloc[-1] - h.time.iloc[0]
    plt.xlim(h.time.iloc[0]*0.9, h.time.iloc[-1] + 0.8*dtime)

    return fig

def plt_two_joint_pdfs(sa, model, num, savfig=True):

    fig = plt.figure(figsize=(14, 6))
    gs0 = gridspec.GridSpec(1, 2, wspace=0.25)
    gs00 = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs0[0])
    gs01 = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs0[1])

    s, ds, dat = read_data(sa, model, num)
    plt_pdf_density_xH2(sa, model, num, dat, gs=gs00, savfig=False)
    fig = plt_pdf_density_CRIR(sa, model, num, dat, gs=gs01, savfig=savfig)
    if savfig:
        plt.close(fig)
    else:
        return fig

if __name__ == '__main__':

    COMM = MPI.COMM_WORLD
    sa = LoadSimTIGRESSXCOAll()

    models = sa.models

    # Measure execution time
    time0 = time.time()
    for model in models:
        if not model.startswith('R8'):
            continue
        s = sa.set_model(model, verbose=False)
        nums = s.nums

        if COMM.rank == 0:
            print('model, nums', model, nums)
            nums = split_container(nums, COMM.size)
        else:
            nums = None

        mynums = COMM.scatter(nums, root=0)
        print('[rank, mynums]:', COMM.rank, mynums)

        for num in mynums:
            print(num, end=' ')
            plt_two_joint_pdfs(sa, model, num)
            # break

        COMM.barrier()
        if COMM.rank == 0:
            print('')
            print('################################################')
            print('# Done with model', model)
            print('# Execution time [sec]: {:.1f}'.format(time.time()-time0))
            print('################################################')
            print('')
