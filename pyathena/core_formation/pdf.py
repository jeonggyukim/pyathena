import numpy as np
from scipy.special import erfinv

class LognormalPDF:
    """
    Lognormal density PDF
    b is the order unity coefficient that depends on the ratio of the
    compressive and solenoidal modes in the turbulence
    (see Federrath 2010, fig. 8; zeta=0.5 corresponds to the natural
    mixture, at which b~0.4)

    methods
    -------
    fx(x):
    get_contrast(frac)
    """
    def __init__(self, Mach, b=0.4, weight='mass'):
        self.mu = 0.5*np.log(1 + b**2*Mach**2)
        self.var = 2*self.mu
        self.sigma = np.sqrt(self.var)
        if weight=='mass':
            pass
        elif weight=='volume':
            self.mu *= -1
        else:
            ValueError("weight must be either mass or volume")

    def fx(self,x):
        """The mass fraction between x and x+dx, where x = ln(rho/rho_0)"""
        f = (1/np.sqrt(2*np.pi*self.var))*np.exp(-(x - self.mu)**2/(2*self.var))
        return f

    def get_contrast(self, frac):
        """
        Returns rho/rho_0 below which frac (0 to 1) of the total mass
        is contained.
        """
        x = self.mu + np.sqrt(2)*self.sigma*erfinv(2*frac - 1)
        return np.exp(x)
