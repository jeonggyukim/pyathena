import numpy as np
# astropy
import astropy.constants as ac
import astropy.units as au

class TigressExtPot(object):
    """TIGRESS external potential class

    Parameters
    ----------
    par : dict
        problem block of input pararmeters
        or a separately created dictionary for parameters in the TIGRESS code units.

        * SurfS : stellar surface density
        * rhodm : dark matter volume density
        * zstar : stellar disk scale height
        * R0 : galactocentric radius
    """
    def __init__(self,param):
        self.surf_s=param['SurfS']*au.M_sun/au.pc**2
        self.rho_dm=param['rhodm']*au.M_sun/au.pc**3
        self.z0=param['zstar']*au.pc
        self.r0=param['R0']*au.pc

    def gext(self,zcode):
        """Calculate gravitational acceleration

        Parameters
        ----------
        zcode : array
            z coordinate in pc

        Returns
        -------
        gext : array
            gravitational acceleration with units
        """
        z = zcode*ac.pc
        a1=2*np.pi*ac.G*self.surf_s
        a2=4*np.pi*ac.G*self.rho_dm
        g1=-a1*z/np.sqrt(z**2+self.z0**2)
        g2=-a2*z/(z**2/self.r0**2+1)
        g_new=g1+g2

        return g_new

    def phiext(self,zcode):
        """Calculate gravitational potential

        Parameters
        ----------
        zcode : array
            z coordinate in pc

        Returns
        -------
        array
            gravitational potential with units
        """
        z = zcode*ac.pc
        phi=2*np.pi*ac.G*(self.surf_s*(np.sqrt(z**2+self.z0**2)-self.z0)
                         +self.rho_dm*self.r0**2*np.log(1+z**2/self.r0**2))
        return phi

    def vesc(self,z):
        """Calculate escape velocity from z to zmax
        """
        return np.sqrt(2*(self.phiext(z.max()).to('km^2/s^2')-self.phiext(z).to('km^2/s^2')))