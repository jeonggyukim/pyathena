import numpy as np
from .tes import TES


class Tools:

    def get_tJeans(self, lmb, rho=None):
        """e-folding time of the fastest growing mode of the Jeans instability
        lmb = wavelength of the mode
        """
        if rho is None:
            rho = self.rho0
        tJeans = 1/np.sqrt(4*np.pi*self.G*rho)*lmb/np.sqrt(lmb**2 - 1)
        return tJeans

    def get_tcr(self, lscale, dv):
        """crossing time for a length scale lscale and velocity dv"""
        tcr = lscale/dv
        return tcr

    def get_Lbox(self, Mach):
        """Return box size at which t_cr = t_Jeans,
        where t_cr = (Lbox/2)/Mach
        """
        dv = Mach*self.cs
        Lbox = np.sqrt(1 + dv**2/(np.pi*self.G*self.rho0))
        return Lbox

    def get_sonic(self, Mach, p=0.5):
        """returns sonic scale for periodic box with Mach number Mach
        assume linewidth-size relation v ~ R^p
        """
        if Mach==0:
            return np.inf
        Lbox = self.get_Lbox(Mach)
        lambda_s = Lbox*Mach**(-1/p)
        return lambda_s

    def get_RLP(self, M):
        """Returns the LP radius enclosing  mass M"""
        RLP = self.G*M/8.86/self.cs**2
        return RLP

    def get_rhoLP(self, r):
        """Larson-Penston asymptotic solution in dimensionless units"""
        rhoLP = 8.86*self.cs**2/(4*np.pi*self.G*r**2)
        return rhoLP

    def get_critical_TES(self, rhoe, lmb_sonic, p=0.5):
        """
        Calculate critical turbulent equilibrium sphere

        Description
        -----------
        Critical mass of turbulent equilibrium sphere is given by
            M_crit = M_{J,e}m_crit(xi_s)
        where m_crit is the dimensionless critical mass and M_{J,e}
        is the Jeans mass at the edge density rho_e.
        This function assumes unit system:
            [L] = L_{J,0}, [M] = M_{J,0}

        Parameters
        ----------
        rhoe : edge density
        lmb_sonic : sonic radius
        p (optional) : power law index for the linewidth-size relation

        Returns
        -------
        rhoc : central density
        R : radius of the critical TES
        M : mass of the critical TES
        """
        LJ_e = 1.0*(rhoe/self.rho0)**-0.5
        MJ_e = 1.0*(rhoe/self.rho0)**-0.5
        xi_s = lmb_sonic / LJ_e
        tes = TES(p, xi_s)
        rat, xi0, m = tes.get_crit()
        rhoc = rat*rhoe
        R = LJ_e*xi0
        M = MJ_e*m
        return rhoc, R, M
