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
    """Turbulent equilibrium sphere

    A family of turbulent equilibrium spheres at a fixed external pressure.
    Default parameters correspond to Bonner-Ebert spheres.

    Parameters
    ----------
    p : float, optional
        power-law index of the velocity dispersion.
    xi_s : float, optional
        dimensionless sonic radius.

    Attributes
    ----------
    p : float
        power-law index of the velocity dispersion.
    xi_s : float
        dimensionless sonic radius.

    Notes
    -----
    A family of turbulent equilibrium spheres parametrized by the center-
    to-edge density contrast, at a fixed edge density (or equivalently,
    external pressure).

    The angle-averaged hydrostatic equation is solved using the following
    dimensionless variables,
        xi = r / L_{J,e},
        u = ln(rho/rho_e),
    where L_{J,e} is the Jeans length at the edge density rho_e.

    Density-weighted, angle-averaged radial velocity dispersion is assumed
    to be a power-law in radius, such that
        <v_r^2> = c_s^2 (r / r_s)^{2p} = c_s^2 (xi / xi_s)^{2p}.

    m = M / M_{J,e},

    Examples
    --------
    >>> import tes
    >>> ts = tes.TES()
    >>> # Find critical parameters
    >>> rat_crit, r_crit, m_crit = ts.get_crit()
    >>> xi = np.logspace(-2, np.log10(r_crit))
    >>> u, du = ts.solve(xi, rat_crit)
    >>> # plot density profile
    >>> plt.loglog(xi, np.exp(u))
    """
    def __init__(self, p=0.5, xi_s=np.inf):
        self.p = p
        self.xi_s = xi_s
        self._xi_min = 1e-5
        self._xi_max = 1e5

    def solve(self, xi, rat):
        """Solve equilibrium equation

        Parameters
        ----------
        xi : array
            Dimensionless radii
        rat : float
            Center-to-edge density contrast.

        Returns
        -------
        u : array
            Log density u = log(rho/rho_e)
        du : array
            Derivative of u: d(u)/d(xi)
        """
        xi = np.array(xi, dtype='float64')
        y0 = np.array([np.log(rat),0])
        if xi.min() > self._xi_min:
            xi = np.insert(xi, 0, self._xi_min)
            istart = 1
        else:
            istart = 0
        if np.all(xi<=self._xi_min):
            u = y0[0]*np.ones(xi.size)
            du = y0[1]*np.ones(xi.size)
        else:
            y = odeint(self._dydx, y0, xi)
            u = y[istart:,0]
            du = y[istart:,1]
        return u, du

    @vectorize(signature="(),()->()")
    def get_radius(self, rat):
        """Calculates the dimensionless radius of a TES.

        The maximum radius of a TES is the radius at which rho = rho_e.

        Parameters
        ----------
        rat : float
            Center-to-edge density contrast.

        Returns
        -------
        float
            Dimensionless maximum radius.
        """
        logxi0 = brentq(lambda x: self.solve(10**x, rat)[0],
                        np.log10(self._xi_min), np.log10(self._xi_max))
        return 10**logxi0

    @vectorize(signature="(),()->()")
    def get_mass(self, rat, xi0=None):
        """Calculates dimensionless enclosed mass.

        The dimensionless mass enclosed within the dimensionless radius xi
        is defined by
            M(xi) = m(xi)M_{J,e}
        where M_{J,e} is the Jeans mass at the edge density rho_e

        Parameters
        ----------
        rat : float
            Center-to-edge density contrast.
        xi0 : float, optional
            Radius within which the enclosed mass is computed. If None, use
            the maximum radius of a sphere.

        Returns
        -------
        float
            Dimensionless enclosed mass.
        """
        if xi0 is None:
            xi0 = self.computeRadius(rat)
        u, du = self.solve(xi0, rat)
        f = 1 + (xi0/self.xi_s)**(2*self.p)
        m = -(xi0**2*f*du + 2*self.p*(f-1)*xi0)/np.pi
        return m.squeeze()[()]

    def get_crit(self):
        """Finds critical TES parameters.

        Critical point is defined as a maximum point in the P-V curve.

        Returns
        -------
        rat_c : float
            Critical density contrast
        r_c : float
            Critical radius
        m_c : float
            Critical mass

        Notes
        -----
        The pressure at a given mass is P = pi^3 c_s^8 / (G^3 M^2) m^2, so
        the minimization is done with respect to m^2.
        """
        # do minimization in log space for robustness and performance
        upper_bound = 6
        res = minimize_scalar(lambda x: -self.computeMass(10**x)**2,
                              bounds=(0, upper_bound), method='Bounded')
        rat_c = 10**res.x
        r_c = self.computeRadius(rat_c)
        m_c = self.computeMass(rat_c)
        if rat_c >= 0.999*10**upper_bound:
            raise Exception("critical density contrast is out-of-bound")
        return rat_c, r_c, m_c

    def _dydx(self, y, x):
        """Differential equation for hydrostatic equilibrium.

        Parameters
        ----------
        y : array
            Vector of dependent variables
        x : array
            Independent variable

        Returns
        -------
        array
            Vector of (dy/dx)
        """
        y1, y2 = y
        dy1 = y2
        f = 1 + (x/self.xi_s)**(2*self.p)
        dy2 = -2/x*(1 + self.p*(1 - 1/f))*y2\
              - 2*self.p*(2*self.p + 1)*(1 - 1/f)/x**2\
              - 4*np.pi**2*np.exp(y1)/f
        return np.array([dy1, dy2])
