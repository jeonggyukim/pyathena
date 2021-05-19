"""
Read athena zprof file using pandas and xarray
"""

from __future__ import print_function

import os
import os.path as osp
import glob
import numpy as np
import pandas as pd
import xarray as xr

def read_zprof_all(dirname, problem_id, phase='whole', savdir=None,
                   force_override=False):
    """Function to read all zprof files in directory and make a Dataset object
    and write to a NetCDF file.

    Note: An xarray DataArray holds a single multi-dimensional variable and its
    coordinates, while a xarray Dataset holds multiple variables that
    potentially share the same coordinates.

    Parameters
    ----------
    dirname : str
        Name of the directory where zprof files are located
    problem_id : str
        Prefix of zprof files
    phase : str
        Name of thermal phase
        ex) whole, phase1, ..., phase5 (cold, intermediate, warm, hot1, hot2)
    savdir : str
        Name of directory to save pickle data as a netcdf file
        Default value is dirname.
    force_override : bool
        Flag to force read of hst file even when netcdf exists

    Returns
    -------
       ds: xarray dataset

    """

    # Find all files with "/dirname/problem_id.xxxx.phase.zprof"
    fname_base = '{0:s}.????.{1:s}.zprof'.format(problem_id, phase)
    fnames = sorted(glob.glob(osp.join(dirname, fname_base)))

    fnetcdf = '{0:s}.{1:s}.zprof.nc'.format(problem_id, phase)
    if savdir is not None:
        fnetcdf = osp.join(savdir, fnetcdf)
    else:
        fnetcdf = osp.join(dirname, fnetcdf)

    print(fnetcdf)

    # Check if netcdf file exists and compare last modified times
    mtime_max = np.array([osp.getmtime(fname) for fname in fnames]).max()
    if not force_override and osp.exists(fnetcdf) and \
        osp.getmtime(fnetcdf) > mtime_max:
        da = xr.open_dataset(fnetcdf)
        return da

    # If here, need to create a new dataarray
    time = []
    df_all = []
    for i, fname in enumerate(fnames):
        # Read time
        with open(fname, 'r') as f:
            h = f.readline()
            time.append(float(h[h.rfind('t=') + 2:]))

        # read pickle if exists
        df = read_zprof(fname, force_override=False)
        if i == 0: # save z coordinates
            z = (np.array(df['z'])).astype(float)
        df.drop(columns='z', inplace=True)
        df_all.append(df)

        # For test
        # if i > 10:
        #     break

    fields = np.array(df.columns)

    # Combine all data
    # Coordinates: time and z
    time = (np.array(time)).astype(float)
    fields = np.array(df.columns)
    df_all = np.stack(df_all, axis=0)
    data_vars = dict()
    for i, f in enumerate(fields):
        data_vars[f] = (('z', 'time'), df_all[...,i].T)

    ds = xr.Dataset(data_vars, coords=dict(z=z, time=time))

    # Somehow overwriting using mode='w' doesn't work..
    if osp.exists(fnetcdf):
        os.remove(fnetcdf)

    try:
        ds.to_netcdf(fnetcdf, mode='w')
    except IOError:
        pass

    return ds

def read_zprof(filename, force_override=False, verbose=False):
    """
    Function to read one zprof file and pickle

    Parameters
    ----------
    filename : string
        Name of the file to open, including extension
    force_override: bool
        Flag to force read of zprof file even when pickle exists

    Returns
    -------
    df : pandas dataframe
    """

    skiprows = 2

    fpkl = filename + '.p'
    if not force_override and osp.exists(fpkl) and \
       osp.getmtime(fpkl) > osp.getmtime(filename):
        df = pd.read_pickle(fpkl)
        if verbose:
            print('[read_zprof]: reading from existing pickle.')
    else:
        if verbose:
            print('[read_zprof]: pickle does not exist or zprof file updated.' + \
                      ' Reading {0:s}'.format(filename))

        with open(filename, 'r') as f:
            # For the moment, skip the first line which contains information about
            # the time at which the file is written
            # "# Athena vertical profile at t=xxx.xx"
            h = f.readline()
            time = float(h[h.rfind('t=') + 2:])
            h = f.readline()
            vlist = h.split(',')
            if vlist[-1].endswith('\n'):
                vlist[-1] = vlist[-1][:-1]    # strip \n

        # c engine does not support regex separators
        df = pd.read_csv(filename, names=vlist, skiprows=skiprows,
                         comment='#', sep=',', engine='python')
        try:
            df.to_pickle(fpkl)
        except IOError:
            pass

    return df

class ReadZprofBase:

    def read_zprof(self, phase='all', savdir=None, force_override=False):
        """Wrapper function to read all zprof output

        Parameters
        ----------
        phase : str or list of str
            List of thermal phases to read. Possible phases are
            'whole' (entire gas)
            'c': (phase1, cold, T < 180),
            'u': (phase2, unstable, 180 <= T < 5050),
            'w': (phase3, warm, 5050 <= T < 2e4),
            'h1' (phase4, warm-hot, 2e4 < T < 5e5),
            'h2' (phase5, hot-hot, T >= 5e5)
            'pi' = photoionized
            'h' = h1 + h2
            '2p' = c + u + w
            'cu' = c + u
            If 'all', read all phases.
        savdir: str
            Name of the directory where pickled data will be saved.
        force_override: bool
            If True, read all (pickled) zprof profiles and dump in netCDF format.

        Returns
        -------
        zp : dict or xarray DataSet
            Dictionary containing xarray DataSet objects for each phase.
        """

        # Mapping from thermal phase name to file suffix
        dct = dict(c='phase1',
                   u='phase2',
                   w='phase3',
                   h1='phase4',
                   h2='phase5',
                   H2='phase6',
                   pi='phase7',
                   HIcc='phase8',
                   HIc='phase9',
                   HIcu='phase10',
                   HIu='phase11',
                   HIw='phase12',
                   HIww='phase13',
                   whole='whole')

        if phase == 'all':
            phase = list(dct.keys()) + ['h', '2p', 'cu']
        else:
            phase = np.atleast_1d(phase)

        zp = dict()
        for ph in phase:
            if ph == 'h':
                zp[ph] = \
                self._read_zprof(phase=dct['h1'], savdir=savdir,
                                 force_override=force_override) + \
                self._read_zprof(phase=dct['h2'], savdir=savdir,
                                 force_override=force_override)
            elif ph == '2p':
                zp[ph] = \
                self._read_zprof(phase=dct['c'], savdir=savdir,
                                 force_override=force_override) + \
                self._read_zprof(phase=dct['u'], savdir=savdir,
                                 force_override=force_override) + \
                self._read_zprof(phase=dct['w'], savdir=savdir,
                                 force_override=force_override)
            elif ph == 'cu':
                zp[ph] = \
                self._read_zprof(phase=dct['c'], savdir=savdir,
                                 force_override=force_override) + \
                self._read_zprof(phase=dct['u'], savdir=savdir,
                                 force_override=force_override)
            else:
                zp[ph] = self._read_zprof(phase=dct[ph], savdir=savdir,
                                          force_override=force_override)

        if len(phase) == 1:
            self.zp = zp[ph]
        else:
            self.zp = zp

        return self.zp

def recal_zprof(sim,num,write=False):
    """Recalculate horizontally averaged vertical profile from vtk dumps"""

    ds = sim.load_vtk(num)
    zprof_data = ds.get_field(field=ds.field_list+['T'])

    # rename variables to use them for the horizontal average
    hydro_rename=dict(density='d',velocity1='v1',velocity2='v2',velocity3='v3',
                      pressure='P',gravitational_potential='Phisg')
    mhd_rename=dict(cell_centered_B1='B1',cell_centered_B2='B2',cell_centered_B3='B3')
    cool_rename=dict(cool_rate='cool',heat_rate='heat')

    if 'density' in zprof_data: zprof_data=zprof_data.rename(hydro_rename)
    if 'cell_centered_B1' in zprof_data: zprof_data=zprof_data.rename(mhd_rename)
    if 'cool_rate' in zprof_data: zprof_data=zprof_data.rename(cool_rename)
    if sim.par['configure']['nscalars'] > 3:
        zprof_data['specific_scalar[3]']=zprof_data['xHI']
    if sim.par['configure']['nscalars'] > 4:
        zprof_data['specific_scalar[4]']=zprof_data['xH2']
    if sim.par['configure']['nscalars'] > 5:
        zprof_data['specific_scalar[5]']=zprof_data['xe']

    # add additional variables to be averaged
    _set_zprof_classic(sim,zprof_data)

    # v2 is set to dv2
    zprof_data['v2'] = zprof_data['dv2']

    # find phases
    idx=_get_phase(zprof_data,sim.par['configure']['new_cooling']=='ON')
    plist = list(idx.keys())

    dA=sim.domain['dx'][0]*sim.domain['dx'][1]

    savdir = sim.savdir+'/zprof'
    if not os.path.isdir(savdir): os.mkdir(savdir)
    for phase in plist:
        zp = _get_mean(zprof_data*idx[phase],dA)
        fname = '{}/{}.{:04d}.{}.zprof.nc'.format(savdir,sim.problem_id,num,phase)
        print(fname)
        if write: zp.to_netcdf(fname)
        zp.close()

def _set_zprof_classic(sim,zprof):
    """Setting data variables to be integrated horizontally (used in TIGRESS classic)
    """
    from ..util.tigress_extpot import TigressExtPot
    par = sim.par
    if par['configure']['ShearingBox'] == 'yes': shearing_box=True
    if par['configure']['gas'] == 'mhd': MHD=True
    nscalars = par['configure']['nscalars']
    Gamma = par['problem']['gamma']
    if shearing_box:
        qshear = par['problem']['qshear']
        Omega = par['problem']['Omega']
        vy0=-qshear*Omega*zprof['x']
        Phit=-qshear*Omega**2*zprof['x']**2
        if 'pattern' in par['problem']:
            vy0 += par['problem']['R0']*(Omega-par['problem']['pattern'])
            vx0 += (Omega-par['problem']['pattern'])

    v1,v2,v3 = zprof['v1'],zprof['v2'],zprof['v3']
    if MHD:
        B1,B2,B3 = zprof['B1'],zprof['B2'],zprof['B3']
        dsqrt=np.sqrt(zprof['d'])
        vdotB=B1*v1+B2*v2+B3*v3
        Emag=0.5*(B1**2+B2**2+B3**2)
        meanB={'1':B1.mean(dim=['x','y']),'2':B2.mean(dim=['x','y']),'3':B3.mean(dim=['x','y'])}

    for vec in ['1','2','3']:
        zprof['M%s' % vec] = zprof['d']*zprof['v%s' % vec]
        zprof['Ek%s' % vec] = 0.5*zprof['d']*zprof['v%s' % vec]**2
        if MHD:
            zprof['PB%s' % vec] = 0.5*zprof['B%s' % vec]**2
            zprof['vA%s' % vec] = zprof['B%s' % vec]/dsqrt
            zprof['dB%s' % vec] = zprof['B%s' % vec] - meanB[vec]
            zprof['dPB%s' % vec] = 0.5*zprof['dB%s' % vec]**2
            zprof['dvA%s' % vec] = zprof['dB%s' % vec]/dsqrt
            zprof['S%s' % vec] = 2.0*Emag*zprof['v%s' % vec] - zprof['B%s' % vec]*vdotB

    if shearing_box:
        dv2 = zprof['v2'] - vy0
        zprof['dv2'] = dv2
        zprof['dM2'] = zprof['d']*dv2
        zprof['dEk2'] = 0.5*zprof['d']*dv2**2
        if 'pattern' in par['problem']:
            dv1 = zprof['v1'] - vx0
            zprof['dv1'] = dv1
            zprof['M1'] = zprof['d']*dv1
            zprof['Ek1'] = 0.5*zprof['d']*dv1**2

    for ns in range(nscalars):
        if 'specific_scalar[{}]'.format(ns) in zprof:
            zprof['s{}'.format(ns +1)]=zprof['specific_scalar[{}]'.format(ns)]*zprof['d']

    dz = sim.domain['dx'][2]
    z = zprof['z'].data
    z_unitary = zprof['z']/zprof['z']
    unitary = zprof['d']/zprof['d']

    phiext=TigressExtPot(par['problem']).phiext
    Phie=phiext(z).to('km**2/s**2').value
    gext=(phiext((z+dz/2))-phiext((z-dz/2))).to('km**2/s**2').value/dz
    zprof['Phie']=Phie*z_unitary
    zprof['gext']=gext*z_unitary
    zprof['dWext']=zprof['d']*zprof['gext']

    Phi=zprof['Phisg'].data
    dPhi=np.zeros_like(Phi)
    dPhi[1:-1,:,:]=(Phi[2:,:,:]-np.roll(Phi,2,axis=0)[2:,:,:])/2.0/dz
    dPhi[0,:,:]=(Phi[1,:,:]-Phi[0,:,:])/dz
    dPhi[-1,:,:]=(Phi[-1,:,:]-Phi[-2,:,:])/dz
    zprof['gsg']=dPhi*unitary
    zprof['dWsg']=zprof['d']*zprof['gsg']
    cs2 = zprof['P']/zprof['d']
    zprof['Ber'] = 0.5*(v1**2+v2**2+v3**2) + Gamma/(Gamma-1)*cs2 + zprof['Phie'] + zprof['Phisg'] + Phit

    dA = sim.domain['dx'][0]*sim.domain['dx'][1]

    for pm in ['p','m']:
        zprof[pm+'A'] = unitary
        zprof[pm+'d'] = zprof['d'].copy(deep=True)
        zprof[pm+'P'] = zprof['P'].copy(deep=True)
        zprof[pm+'vz'] = zprof['v3'].copy(deep=True)
        for f in ['d','M1','M2','M3']:
            zprof['%sFz%s' % (pm,f)] = zprof[f]*v3
        for f in ['E1','E2','E3','Ege','Egsg','Etidal']:
            if f in ['E1','E2','E3']:
                zf='%sk%s' %(f[0],f[1])
                tmp = zprof[zf]*v3
            elif f == 'Ege': tmp = zprof['M3']*zprof['Phie']
            elif f == 'Egsg': tmp = zprof['M3']*zprof['Phisg']
            elif f == 'Etidal': tmp = zprof['M3']*Phit
            zprof['%sFz%s' % (pm,f)] = tmp
        zprof['%sFzP' % pm] = Gamma/(Gamma-1)*zprof['P']*v3
        zprof['%sFzEWsg' % pm] = zprof['M3']*zprof['gsg']
        zprof['%sFzEWext' % pm] = zprof['M3']*zprof['gext']
        if MHD:
            zprof['%sSzEm1' % pm] = 2.0*zprof['PB1']*v3
            zprof['%sSzEm2' % pm] = 2.0*zprof['PB2']*v3
            zprof['%sSzvB1' % pm] = -B3*B1*v1
            zprof['%sSzvB2' % pm] = -B3*B2*v2
        for ns in range(nscalars):
            if 's{}'.format(ns+1) in zprof:
                zprof['%sFzs%s' % (pm,ns+1)]=zprof['s{}'.format(ns+1)]*zprof['v3']
    if shearing_box:
        zprof['Rxy']=zprof['d']*v1*dv2
        if MHD: zprof['Mxy']=-B1*B2

    nv3=zprof['v3']<0
    pv3=~nv3
    for k in zprof:
        if k.startswith('p'): zprof[k]=zprof[k].where(pv3).fillna(0.)
        if k.startswith('m'): zprof[k]=zprof[k].where(nv3).fillna(0.)

    zprof['A']=unitary*dA

def _get_phase(data,new_cool=False):
    """Define phases as done in the original code"""
    temp=data['T']

    idx={}
    if new_cool:
        idx['phase1']=temp < 250
        idx['phase2']=(temp >= 250) * (temp <6000)
        idx['phase3']=(temp >= 6000) * (temp <3.e4)
        idx['phase4']=(temp >= 3.e4) * (temp <5.e5)
        idx['phase5']=temp >= 5.e5
        idx['phase6']=data['xH2']>= 0.25 # H2
        idx['phase7']=((data['xHI'] + 2.0*data['xH2']) < 0.5) * (temp<3.e4)
        idx['phase8']=(data['xHI'] > 0.5) * (temp < 125)
        idx['phase9']=(data['xHI'] > 0.5) * (temp <=125) * (temp < 250)
        idx['phase10']=(data['xHI'] > 0.5) * (temp <=250) * (temp < 1000)
        idx['phase11']=(data['xHI'] > 0.5) * (temp <=1000) * (temp < 6000)
        idx['phase12']=(data['xHI'] > 0.5) * (temp <=6000) * (temp < 12000)
        idx['phase13']=(data['xHI'] > 0.5) * (temp <=12000) * (temp < 30000)
        idx['whole']=temp>0
    else:
        idx['phase1']=temp < 184
        idx['phase2']=(temp >= 184) * (temp <5050)
        idx['phase3']=(temp >= 5050) * (temp <2.e4)
        idx['phase4']=(temp >= 2.e4) * (temp <5.e5)
        idx['phase5']=temp >= 5.e5

    return idx

def _get_mean(dset,dA):
    """Function to calculate the horizontal averages
    with proper handling of stresses at x boundaries
    """
    zp=dset.sum(dim=['x','y'])*dA

    zp['A']=dset['A'].sum(dim=['x','y'])

    Nx=len(dset.x)

    if 'Rxy' in zp:
        zp=zp.drop('Rxy')

        stress=dset['Rxy'].sum(axis=1)
        zp['RxyL']=stress.isel(x=0).drop('x')*Nx*dA
        zp['RxyR']=stress.isel(x=-1).drop('x')*Nx*dA
        if 'Mxy' in dset:
            zp=zp.drop('Mxy')
            stress=dset['Mxy'].sum(axis=1)
            zp['MxyL']=stress.isel(x=0).drop('x')*Nx*dA
            zp['MxyR']=stress.isel(x=-1).drop('x')*Nx*dA
    return zp
