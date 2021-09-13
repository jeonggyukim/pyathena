import pathlib
import os.path as osp
import numpy as np
import astropy.units as au
import astropy.constants as ac
import xarray as xr

local = pathlib.Path(__file__).parent.absolute()

def read_FG20():
    """Function to read Faucher-Giguère (2020) UV background as functions of nu and
    z 

    See : https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.1614F/abstract
    https://galaxies.northwestern.edu/uvb-fg20/

    Returns
    -------
    r : dict

    """
    fname = osp.join(local, '../../data/fg20_spec_lambda.dat')
    with open(fname) as fp:
        ll = fp.readlines()

    # Get redshift
    z = np.array(list(map(lambda zz: np.round(float(zz),3), ll[0].split())))
    zstr = np.array(list(map(lambda zz: str(np.round(float(zz),3)), ll[0].split())))
    nz = len(zstr)

    # Get wavelengths
    wav = []
    for l in ll[1:]:
        wav.append(float(l.split()[0]))

    wav = np.array(wav)*au.angstrom
    nwav = len(wav)

    # Read Jnu
    Jnu = np.zeros((nz,nwav))
    for i,l in enumerate(ll[1:]):
        Jnu[:,i] = np.array(list(map(float, l.split())))[1:]

    r = dict()
    r['nwav'] = nwav
    r['wav'] = wav
    r['nz'] = nz
    r['z'] = z
    r['Jnu'] = Jnu
    r['nu'] = (ac.c/wav).to('Hz').value
    
    da = xr.DataArray(data=r['Jnu'], dims=['z', 'wav'], coords=[r['z'],r['wav']],
                      attrs=dict(description='FG20 UVB', units='ergs/s/cm^2/Hz/sr'))
    r['ds'] = xr.Dataset(dict(Jnu=da))
    
    return r
