import numpy as np
import pandas as pd


class AbundanceSolar(object):
    """Class to read protosolar abundances of the elements with Z <= 32
    (Table 1.4 in Draine 2011; based on Asplund et al. 2009) and compute
    mean molecular weight, etc.

    Properties
    ----------
    df : pandas DataFrame
        Table 1.4 in Draine (2011)

    Parameters
    ----------
    xHe : float
        Elemental abundance of helium. Default value is 0.0955 (Asplund+2009)
    Zprime : float
        Metallicity scaled to solar abundances.
    """

    def __init__(self, xHe=9.55e-2, Zprime=1.0):

        self.Zprime = Zprime
        data = {
            'Z': np.arange(32) + 1,
            'X': np.array([ 'H', 'He', 'Li', 'Be',  'B',  'C',  'N',  'O',
                            'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',  'P',  'S',
                            'Cl', 'Ar',  'K', 'Ca', 'Sc', 'Ti',  'V', 'Cr',
                            'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge']),
            'mX_amu': np.array([1.008, 4.0026,  6.941,  9.012, 10.811, 12.011, 14.007, 15.999,
                                18.998, 20.180, 22.990, 24.305, 26.982, 28.086, 30.974, 32.065,
                                35.453, 39.948, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996,
                                54.938, 55.845, 58.933, 58.693, 63.546, 65.380, 69.723, 72.640]),
            'NX_NH': np.array([1.0, 9.55e-2, 2.00e-9, 2.19e-11, 6.76e-10, 2.95e-4, 7.41e-5, 5.37e-4,
                               2.88e-8, 9.33e-5, 2.04e-6, 4.37e-5, 2.95e-6, 3.55e-5, 3.23e-7, 1.45e-5,
                               1.86e-7, 2.75e-6, 1.32e-7, 2.14e-6, 1.23e-9, 8.91e-8, 1.00e-8, 4.79e-7,
                               3.31e-7, 3.47e-5, 8.13e-8, 1.74e-6, 1.95e-8, 4.68e-8, 1.32e-9, 4.17e-9])
        }
        df = pd.DataFrame.from_dict(data)

        # Adjust Helium abundance
        df.loc[df['X'] == 'He', 'NX_NH'] = xHe

        # Scale metal abundance by Zprime
        df.loc[df['Z'] > 2, 'NX_NH'] *= self.Zprime

        # TODO: Check these again
        # Mass [amu] for nH=1
        df['MX_per_H'] = df['mX_amu']*df['NX_NH']
        # Mass fraction
        # divide by MH = 1.008 amu so that MX_MH = 1.0 for H
        df['MX_MH'] = df['MX_per_H']/df.loc[0, 'MX_per_H']

        self.df = df

    def get_XYZ_muH_mu(self, verbose=False):
        """Compute mass fractions of hydrogen (X), helium (Y), and metals (Z),
        mean molecular weight per H (muH), and mean molecular weight per
        particle (mu). Dust depletion is ignored.

        Parameters
        ----------
        verbose : bool
            Print summary of results.
        """

        df = self.df
        xHe = df[df['X'] == 'He']['NX_NH'].iloc[0]
        xMetal = df[df['Z'] > 2]['NX_NH'].sum()

        # Total mass (in units of amu) per H
        Mtot = df['MX_per_H'].sum()

        # Mean molecular weight per H (in units of mH)
        muH = Mtot/1.008

        # Mass fraction of H, He, and Metals
        X = df[df['X'] == 'H']['MX_MH'].iloc[0]/Mtot
        Y = df[df['X'] == 'He']['MX_MH'].iloc[0]/Mtot
        Z = df[df['Z'] > 2]['MX_MH'].sum()/Mtot

        # Mean molecular weight (per particle)
        # Fully atomic
        mu_atom = df['MX_MH'].sum()/df['NX_NH'].sum()

        # Fully ionized
        xe_ion = (df['NX_NH']*df['Z']).sum()
        mu_ion = df['MX_MH'].sum()/(df['NX_NH']*(1.0 + df['Z'])).sum()

        res = dict()
        res['xHe'] = xHe
        res['xMetal'] = xMetal
        res['X'] = X
        res['Y'] = Y
        res['Z'] = Z
        res['muH'] = muH
        res['mu_atom'] = mu_atom
        res['mu_ion'] = mu_ion
        res['xe_ion'] = xe_ion
        res['df'] = df

        if verbose:
            print('Elemental abundance of Helium xHe :', res['xHe'])
            print('xMetal :', res['xMetal'])
            print('Hydrogen mass fraction X:', res['X'])
            print('Helium mass fraction Y:', res['Y'])
            print('Metal mass fraction Z:', res['Z'])
            print('Mean molecular weight per H:', res['muH'])
            print('Mean molecular weight per ptcl (atomic gas):', res['mu_atom'])
            print('Mean molecular weight per ptcl (fully ionized):', res['mu_ion'])
            print('Electron fraction x_e (fully ionized):', res['xe_ion'])

        return res
