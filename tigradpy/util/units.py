import astropy.units as au
import astropy.constants as ac

class Units(object):
    """Simple class for simulation unit.

    Msun, Lsun, etc.: physical constants in code unit
    """
    def __init__(self, kind='LV', muH=1.4271):
        """
        Parameters
        ----------
           kind: string
              "LV" for (pc, km/s) or "LT" for (pc, Myr)
           muH: float
              mean particle mass per H (for neutral gas).
              Default value is 1.4271 (assuming solar metallicity).
        """
        
        mH = 1.00794*au.u
        
        self.muH = muH
        self.length = (1.0*au.pc).to('pc')
        if kind == 'LV':
            self.velocity = (1.0*au.km/au.s).to('km/s')
            self.time = (self.length/self.velocity).cgs
        elif kind == 'LT':
            self.time = (1.0*au.Myr).to('Myr')
            self.velocity = (self.length/self.time).to('km/s')

        self.mass = (self.muH*mH*(self.length.to('cm').value)**3).to('Msun')
        self.density = (self.mass/self.length**3).cgs
        self.energy = (self.mass*self.velocity**2).cgs
        self.pressure = (self.density*self.velocity**2).cgs
        self.energy_density = self.pressure
        
        self.mass_flux = (self.density*self.velocity).to('Msun/(pc**2*Myr)')
        
        # Define (physical constants in code units)^-1
        # Opposite to the convention chosen by set_units function in athena/src/units.c
        # because in post-processing we want to convert from code units to
        # more convenient ones by multiplying these constants
        self.pc = self.length.to('pc').value
        self.Myr = self.time.to('Myr').value
        self.kms = self.velocity.to('km/s').value
        self.Msun = self.mass.to('Msun').value
        self.Lsun = (self.energy/self.time).to('Lsun').value
        self.eV = self.energy.to('eV').value
        self.s = self.time.to('s').value
        self.pok = ((self.pressure/ac.k_B).to('cm**-3*K')).value

        # For yt
        self.units_override = dict(length_unit=(self.length.to('pc').value, 'pc'),
                                   time_unit=(self.time.to('Myr').value, 'Myr'),
                                   mass_unit=(self.mass.to('Msun').value, 'Msun'))
