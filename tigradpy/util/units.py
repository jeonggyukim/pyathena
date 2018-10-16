import astropy.units as au
import astropy.constants as ac

class units(object):
    """Simple class for simulation unit.

    lunit: length
    munit: mass
    vunit: velocity
    tunit: time
    dunit: density
    eunit: energy
    punit: pressure (energy density)

    Msun, Lsun, etc.: physical constants in code unit
    """
    def __init__(self, kind='LV', muH=1.4272):
        """
        Parameters
        ----------
           kind: string
              "LV" for (pc, km/s) or "LT" for (pc, Myr)
           muH: float
              mean molecular weight
        """
        self.mH = 1.00794*au.u
        self.muH = muH
        
        self.lunit = (1.0*au.pc).to('pc')
        if kind == 'LV':
            self.vunit = (1.0*au.km/au.s).to('km/s')
            self.tunit = (self.lunit/self.vunit).cgs     
        elif kind == 'LT':
            self.tunit = (1.0*au.Myr).to('Myr')
            self.vunit = (self.lunit/self.tunit).to('km/s')

        self.munit = (self.muH*self.mH*(au.pc.to('cm'))**3).cgs
        self.dunit = (self.munit/self.lunit**3).cgs    
        self.eunit = (self.munit*self.vunit**2).cgs
        self.punit = (self.dunit*self.vunit**2).cgs
        
        self.pc = self.lunit.to('pc').value
        self.Myr = self.tunit.to('Myr').value
        self.kms = self.vunit.to('km/s').value
        self.Msun = self.munit.to('Msun').value
        self.Lsun = (self.eunit/self.tunit).to('Lsun').value

        # For yt
        self.units_override = dict(length_unit=(self.lunit.to('pc').value, 'pc'),
                                   time_unit=(self.tunit.to('Myr').value, 'Myr'),
                                   mass_unit=(self.munit.to('Msun').value, 'Msun'))
