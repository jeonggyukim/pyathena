import numpy as np
import xarray as xr
import astropy.units as au
import astropy.constants as ac
import sys, os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse

sys.path.insert(0,'../../')

import pyathena as pa
import gc

from pyathena.tigress_ncr.get_cooling import get_cooling_heating,get_pdfs,set_bins_default,get_pdf_xarray
from pyathena.fields.xray_emissivity import get_xray_emissivity

#s=pa.LoadSimTIGRESSNCR('/tigress/changgoo/RPS_4pc_ICM1_newacc',verbose=True)
#nums = s.nums
if len(sys.argv) == 2:
    suite = sys.argv[1]
else:
    suite = 'classic'
print(suite)
if suite == 'classic':
    basedir='/tigress/changgoo/TIGRESS-default/'
    dirs = os.listdir(basedir)
    dirs.sort()
    models = dict()
    for d in dirs:
        if os.path.isdir(os.path.join(basedir,d)):
            if d.startswith('R16'): continue
            models['-'.join(d.split('_')[:2])] = os.path.join(basedir,d)
elif suite == 'icm':
    basedir='/tigress/changgoo/'
    dirs = os.listdir(basedir)
    dirs.sort()
    models = dict()
    for d in dirs:
        if os.path.isdir(os.path.join(basedir,d)):
            if ('ICM' in d) and ('newacc' in d):
                models['-'.join(d.split('_')[:3])] = os.path.join(basedir,d)
else:
    raise KeyError("suite must be 'classic' or 'icm'")

sa = pa.LoadSimTIGRESSNCRAll(models)
for m in models:
    s = sa.set_model(m)
    ds = s.load_vtk(s.nums[0])
    pid = ds.problem_id
    if suite == 'classic':
        nums = range(200,501)
    elif suite == 'icm':
        nums = s.nums
    for num in nums:
        if not (num in s.nums): continue
        savdir = '{}/jointpdf/cooling_heating/'.format(s.savdir)
        if not os.path.isdir(savdir): os.makedirs(savdir)
        coolpdfname = '{}.{:04d}.cool.pdf.nc'.format(pid,num)
        sxname = '{}.{:04d}.SX.nc'.format(pid,num)
        if os.path.isfile(os.path.join(savdir,coolpdfname)): continue

        ds = s.load_vtk(num)
        print(m, num, ds.domain['time'], end=' ')

        data = s.get_classic_cooling_rate(ds).rename(density='nH')

        emin = 0.3 # keV
        emax = 2.0 # keV
        em = get_xray_emissivity(data['T'].data, 1.0, emin, emax, energy=True)
        data['soft-Xray']=xr.DataArray(em*data['nH']**2,coords=[data.z,data.y,data.x])
        emax = 7.0 # keV
        em = get_xray_emissivity(data['T'].data, 1.0, emin, emax, energy=True)
        data['full-Xray']=xr.DataArray(em*data['nH']**2,coords=[data.z,data.y,data.x])
    
        # get total cooling from vtk output for normalization
        total_cooling=data['cool_rate'].sum().data
        # get total heating from vtk output for normalization
        total_heating=data['heat_rate'].sum().data

        pdf = get_pdfs('nH','T',data,data[['cool_rate','heat_rate','net_cool_rate','soft-Xray','full-Xray']])/total_cooling

        pdf.attrs['total_cooling'] = total_cooling
        pdf.attrs['total_heating'] = total_heating
        pdf.assign_coords(time=ds.domain['time'])
        pdf.to_netcdf(os.path.join(savdir,coolpdfname))

        S_X = xr.Dataset()
        S_X['soft_yz']=(data['soft-Xray'].sum(dim='x')*s.domain['dx'][0]*s.u.cm/4/np.pi)
        S_X['soft_xz']=(data['soft-Xray'].sum(dim='y')*s.domain['dx'][1]*s.u.cm/4/np.pi)
        S_X['soft_xy']=(data['soft-Xray'].sum(dim='z')*s.domain['dx'][2]*s.u.cm/4/np.pi)
        S_X['full_yz']=(data['full-Xray'].sum(dim='x')*s.domain['dx'][0]*s.u.cm/4/np.pi)
        S_X['full_xz']=(data['full-Xray'].sum(dim='y')*s.domain['dx'][1]*s.u.cm/4/np.pi)
        S_X['full_xy']=(data['full-Xray'].sum(dim='z')*s.domain['dx'][2]*s.u.cm/4/np.pi)
        S_X.assign_coords(time=ds.domain['time'])
        S_X.to_netcdf(os.path.join(savdir,sxname))

        gc.collect()
