import astropy.units as au
import astropy.constants as ac
import numpy as np

class Units(object):
    """Simple class for simulation unit.

    Msun, Lsun, etc.: physical constants in code unit
    """
    def __init__(self, kind='LV', muH=1.4271, units_dict=None):
        """
        Parameters
        ----------
        kind : str
           "LV" for (pc, km/s) or "LT" for (pc, Myr) or "cgs" for (cm, s), or
           "code" for code unit. For athena++ simulations, set to "custom"
           and provide units_dict (<units> block in athinput file)
        muH : float
            Mean particle mass per H. Default value is 1.4271 (TIGRESS-classic).
            See examples for other cases.
        units_dict : dict
            Dictionary containing <units> block in tigris athinput file.

        Returns
        -------
        u : dict
            Units instance

        Example
        -------
        # Athena-TIGRESS with classic cooling
        >>> u = Units(kind='LV', muH=1.4271)
        # Athena-TIGRESS with NCR cooling
        >>> u = Units(kind='LV', muH=1.4)
        # TIGRIS "ism" unit system or any other custom unit system
        >>> units_dict = {'units_system': 'ism',
                          'mass_cgs': 4.91615563682836e+31,
                          'length_cgs': 3.085678e+18,
                          'time_cgs': 30856780000000,
                          'mean_mass_per_hydrogen': 2.34262e-24}
        >>> u = Units('custom', units_dict=units_dict)
        """
        mH = (1.008*au.u).to('g')
        # If code units, set [L]=[M]=[T]=1 and return.
        if kind == 'code':
            self.length = 1.0
            self.mass = 1.0
            self.time = 1.0
            self.units_override = None
            return
        elif kind == 'LV':
            self.muH = muH
            self.length = (1.0*au.pc).to('pc')
            self.velocity = (1.0*au.km/au.s).to('km s-1')
            self.time = (self.length/self.velocity).cgs
        elif kind == 'LT':
            self.muH = muH
            self.length = (1.0*au.pc).to('pc')
            self.time = (1.0*au.Myr).to('Myr')
            self.velocity = (self.length/self.time).to('km s-1')
        elif kind == 'cgs':
            self.length = 1.0*au.cm
            self.time = 1.0*au.s
            self.velocity = (self.length/self.time).to('km s-1')
        elif kind == 'custom':
            if units_dict is None:
                raise ValueError(f"Provide units_dict for athena++ simulations")

            # Try manually setting the unit for the ism unit system
            if units_dict['unit_system'] == 'ism':
                self.mass = (units_dict['mass_cgs']*au.g).to('Msun')
                self.length = (units_dict['length_cgs']*au.cm).to('pc')
                self.time = (units_dict['time_cgs']*au.s).to('Myr')
                self.velocity = (self.length/self.time).to('km s-1')
            else:
                self.mass = units_dict['mass_cgs']*au.g
                self.length = units_dict['length_cgs']*au.cm
                self.time = units_dict['time_cgs']*au.s
                self.velocity = self.length/self.time
            try:
                self.muH = units_dict['mean_mass_per_hydrogen']
            except KeyError:
                pass
        else:
            raise ValueError(f"Unrecognized unit system: {kind}")

        self.mH = mH.to('g')
        # mass unit set here if not "custom"
        if not hasattr(self, 'mass'):
            self.mass = (self.muH*mH*(self.length.to('cm').value)**3).to('Msun')
        self.density = (self.mass/self.length**3).cgs
        self.momentum = (self.mass*self.velocity).to('Msun km s-1')
        self.energy = (self.mass*self.velocity**2).cgs
        self.pressure = (self.density*self.velocity**2).cgs
        self.energy_density = self.pressure.to('erg cm-3')

        self.mass_flux = (self.density*self.velocity).to('Msun kpc-2 yr-1')
        self.momentum_flux = (self.density*self.velocity**2
                              ).to('Msun km s-1 kpc-2 yr-1')
        self.energy_flux = (self.density*self.velocity**3).to('erg kpc-2 yr-1')

        # Define (physical constants in code units)^-1
        #
        # Opposite to the convention chosen by set_units function in
        # athena/src/units.c This is because in post-processing we want to
        # convert from code units to more convenient ones by "multiplying"
        # these constants
        self.cm = self.length.to('cm').value
        self.pc = self.length.to('pc').value
        self.kpc = self.length.to('kpc').value
        self.Myr = self.time.to('Myr').value
        self.kms = self.velocity.to('km s-1').value
        self.Msun = self.mass.to('Msun').value
        self.Lsun = (self.energy/self.time).to('Lsun').value
        self.erg = self.energy.to('erg').value
        self.eV = self.energy.to('eV').value
        self.s = self.time.to('s').value
        self.pok = ((self.pressure/ac.k_B).to('cm-3 K')).value
        self.muG = np.sqrt(4*np.pi*self.energy_density.cgs.value)/1e-6

        # For yt
        self.units_override = dict(length_unit=(
                                       self.length.to('pc').value, 'pc'),
                                   time_unit=(
                                       self.time.to('Myr').value, 'Myr'),
                                   mass_unit=(
                                       self.mass.to('Msun').value, 'Msun'))
