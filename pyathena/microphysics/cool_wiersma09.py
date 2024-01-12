# Read Wiersma+2009 cooling tables

import os
import os.path as osp
import pathlib
import h5py
import numpy as np

class CoolWiersma09(object):
    """Class to read Wiersma cooling tables Need to first run idl scripts to
    generate tables or download hdf5 files (for non-zero redshift)

    # Data from:
    # https://www.strw.leidenuniv.nl/WSS08
    # https://www.strw.leidenuniv.nl/WSS08/z_collis.txt
    # https://www.strw.leidenuniv.nl/WSS08/coolingtables_highres.tar.gz
    """

    def __init__(self, z=0.0, CIE=False):

        self.basedir = osp.join(pathlib.Path(__file__).parent.absolute(),
                                '../../data/microphysics/Wiersma09/tables')

        self.r = self.read_table(z=z, CIE=CIE)

    def read_table(self, z=0.0, CIE=False):

            if CIE:
                fname = osp.join(self.basedir, 'z_collis.hdf5')
            else:
                fname = osp.join(self.basedir, 'z_{0:.3f}.hdf5'.format(z))

            with h5py.File(fname, 'r') as f:
                h = f.get('Header')
                abd = h.get('Abundances')
                solar = f.get('Solar')
                r = dict()
                r['xe'] = np.array(solar['Electron_density_over_n_h'])
                r['mu'] = np.array(solar['Mean_particle_mass'])
                r['Lambda'] = np.array(solar['Net_cooling'])
                r['T'] = np.array(solar['Temperature_bins'])
                try:
                    r['nH'] = np.array(solar['Hydrogen_density_bins'])
                except KeyError:
                    r['nH'] = None

                metalf = f.get('Metal_free')
                metalt = f.get('Total_Metals')
                r['xe_Z0'] = np.array(metalf['Electron_density_over_n_h'])
                r['mu_Z0'] = np.array(metalf['Mean_particle_mass'])
                r['Lambda_Z0'] = np.array(metalf['Net_Cooling'])
                r['T_Z0'] = np.array(metalf['Temperature_bins'])
                r['Y_Z0'] = np.array(metalf['Helium_mass_fraction_bins'])
                r['xHe_Z0'] = np.array(metalf['Helium_number_ratio_bins'])

                r['T_metal'] = np.array(metalt['Temperature_bins'])
                r['Lambda_metal'] = np.array(metalt['Net_cooling'])

                #print(f.keys(), h.keys(), solar.keys())

                r['abd'] = dict()
                for k in abd.keys():
                    r['abd'][k] = abd.get(k)[()]

                r['header'] = dict()
                for k in h.keys():
                    if k == 'Abundances':
                        continue
                    r['header'][k] = h.get(k)[()]

                X = abd.get('Solar_mass_fractions')[()]
                r['abd']['muH'] = X.sum()/X[0]

            return r
