import numpy as np

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
