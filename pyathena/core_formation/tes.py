from scipy.integrate import odeint, quad
from scipy.optimize import minimize_scalar, brentq
import numpy as np
import functools


def vectorize(otypes=None, signature=None):
    """Numpy vectorization wrapper that works with instance methods."""
    def decorator(fn):
        vectorized = np.vectorize(fn, otypes=otypes, signature=signature)
        @functools.wraps(fn)
        def wrapper(*args):
            return vectorized(*args)
        return wrapper
    return decorator


class TES:
    """Turbulent Equilibrium Sphere

    Description
    -----------
    Turbulent equilibrium spheres are characterized by two parameters:
      p: power-law index for the linewidth-size relation
      xi_s: the sonic radius
    Given these two parameters, the equilibrium profile is obtained by solving
    a hydrostatic equation in the following dimensionless variables:

    xi = r / L_{J,e}, where L_{J,e} the Jeans length at the edge density rho_e.
    m = M / M_{J,e}
    u = ln(rho/rho_e)

    Assume the density-weighted, angle-averaged radial velocity square has a
    power-law dependence on the radius:
        <v_r^2> = c_s^2 (r / r_s)^{2p} = c_s^2 (xi / xi_s)^{2p}
    where xi_s = np.inf corresponds to the usual Bonner-Ebert sphere.
    """
    def __init__(self, p=0.5, xi_s=np.inf):
        self.p = p
        self.xi_s = xi_s
        self.xi_min = 1e-5
        self.xi_max = 10

    def solve(self, xi, rat):
        """Solve equilibrium equation

        Returns
        -------
        u : array_like
            log density u = log(rho/rho_e)
        du : derivative of u: d(u)/d(xi)
        """
        xi = np.array(xi, dtype='float64')
        y0 = np.array([np.log(rat),0])
        if xi.min() > self.xi_min:
            xi = np.insert(xi, 0, self.xi_min)
            istart = 1
        else:
            istart = 0
        if np.all(xi<=self.xi_min):
            u = y0[0]*np.ones(xi.size)
            du = y0[1]*np.ones(xi.size)
        else:
            y = odeint(self._dydx, y0, xi)
            u = y[istart:,0]
            du = y[istart:,1]
        return u, du

    @vectorize(signature="(),()->()")
    def computeRadius(self, rat):
        """Calculate the dimensionless radius where rho = rho_e"""
        xi0 = brentq(lambda x: self.solve(x, rat)[0], self.xi_min, self.xi_max)
        return xi0

    @vectorize(signature="(),()->()")
    def computeMass(self, rat):
        """Calculate dimensionless mass

        Description
        -----------
        The dimensionless mass is defined as
            M(xi) = m(xi)M_{J,e}
        where M_{J,e} is the Jeans mass at the edge density rho_e
        """
        xi0 = self.computeRadius(rat)
        u, du = self.solve(xi0, rat)
        f = 1 + (xi0/self.xi_s)**(2*self.p)
        m = -(xi0**2*f*du + 2*self.p*(f-1)*xi0)/np.pi
        return m.squeeze()[()]

    def get_crit(self):
        """
        Find density contrast at which pressure become maximum at the P-V curve
        Note that the pressure at the given mass = P = pi^3 c_s^8 / (G^3 M^2) m^2

        Returns
        -------
        rat_c: critical density contrast
        r_c: critical radius
        m_c: critical mass
        """
        res = minimize_scalar(lambda x: -self.computeMass(x)**2,
                              bounds=(1e0, 1e3), method='Bounded')
        rat_c = res.x
        r_c = self.computeRadius(rat_c)
        m_c = self.computeMass(rat_c)
        if rat_c >= 999:
            raise Exception("critical density contrast is out-of-bound")
        return rat_c, r_c, m_c

    def _dydx(self, y, x):
        """Hydrostatic equilibrium equation

        Parameters
        ----------
        y : array_like
            vector of dependent variables
        x : independent variable
        """
        y1, y2 = y
        dy1 = y2
        f = 1 + (x/self.xi_s)**(2*self.p)
        dy2 = -2/x*(1 + self.p*(1 - 1/f))*y2\
              - 2*self.p*(2*self.p + 1)*(1 - 1/f)/x**2\
              - 4*np.pi**2*np.exp(y1)/f
        return np.array([dy1, dy2])
