import h5py
import astropy.units as au
import astropy.constants as ac
import numpy as np

def read_P20(shielded=True, z=0.0, logZ=0.0, verbose=False):
    """Function to read the fiducial model of Ploeckinger et al. (2020) cooling
    table (UVB_dust1_CR1_G1_shield1)

    https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.4857P/abstract
    """

    if shielded:
        fname='/tigress/jk11/code/Ploeckinger20/UVB_dust1_CR1_G1_shield1.hdf5'
    else:
        fname='/tigress/jk11/code/Ploeckinger20/UVB_dust1_CR1_G1_shield0.hdf5'

    f = h5py.File(fname, mode='r')
    if verbose:
        for key in f.keys():
            print(f[key]) #Names of the groups in HDF5 file.

        for k in f['NumberOfBins'].keys():
            print('Key:',k)
            print('Number of bins', f['NumberOfBins'][k], f['NumberOfBins'][k][()])

    n = 10.0**f['TableBins']['DensityBins'][()]
    zz = f['TableBins']['RedshiftBins'][()]
    ZZ = f['TableBins']['MetallicityBins'][()]

    T = 10.0**f['ThermEq/Temperature'][()]
    xe = 10.0**f['ThermEq/ElectronFractions'][()] # 14 = 11 individual element + (prim + metal + total)
    Gamma = f['ThermEq/GammaHeat'][()]
    U = 10.0**f['ThermEq/U_from_T'][()]
    mu = f['ThermEq/MeanParticleMass'][()]
    muH = (1.4*au.u).cgs.value
    pok = (Gamma-1.0)*(muH*n)*U/ac.k_B.cgs.value
    heat = 10.0**f['ThermEq/Heating'][()]
    cool = 10.0**f['ThermEq/Cooling'][()]

    def _find_nearest(array, value):
        array = np.asarray(array)
        return (np.abs(array - value)).argmin()

    idx_z = _find_nearest(zz, z)
    idx_Z = _find_nearest(ZZ, logZ)

    r = dict()
    r['n'] = n
    r['pok'] = pok[idx_z,idx_Z,:]
    r['T'] = T[idx_z,idx_Z,:]
    r['xe'] = xe[idx_z,idx_Z,:,-1]
    r['heatPE'] = heat[idx_z,idx_Z,:,22]
    r['heatTotalPrim'] = heat[idx_z,idx_Z,:,23]
    #r['heatTotalMetal'] = heat[idx_z,idx_Z,:,24]
    r['heat'] = heat[idx_z,idx_Z,:,:]
    r['cool'] = cool[idx_z,idx_Z,:,:]

    r['IdentifierHeating'] = list(f['IdentifierHeating'])
    r['IdentifierCooling'] = list(f['IdentifierCooling'])

    if verbose:
        print('IdentifierHeating')
        print(r['IdentifierHeating'])
        print('IdentifierCooling')
        print(r['IdentifierCooling'])

    f.close()

    return r
