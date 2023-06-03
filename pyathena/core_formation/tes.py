from scipy.integrate import odeint
from scipy.optimize import minimize_scalar, brentq
import matplotlib.pyplot as plt
import numpy as np


class TESe:
    """Turbulent equilibrium sphere of a fixed external pressure.

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
    >>> ts = tes.TESe()
    >>> # Find critical parameters
    >>> u_crit, r_crit, m_crit = ts.get_crit()
    >>> r = np.logspace(-2, np.log10(r_crit))
    >>> u, du = ts.solve(r, u_crit)
    >>> # plot density profile
    >>> plt.loglog(r, np.exp(u))
    """
    def __init__(self, p=0.5, xi_s=np.inf):
        self.p = p
        self.xi_s = xi_s
        self._xi_min = 1e-5
        self._xi_max = 1e3

    def solve(self, xi, u0):
        """Solve equilibrium equation

        Parameters
        ----------
        xi : array
            Dimensionless radii
        u0 : float
            Dimensionless logarithmic central density

        Returns
        -------
        u : array
            Log density u = log(rho/rho_e)
        du : array
            Derivative of u: d(u)/d(xi)
        """
        xi = np.array(xi, dtype='float64')
        y0 = np.array([u0, 0])
        if xi.min() > self._xi_min:
            xi = np.insert(xi, 0, self._xi_min)
            istart = 1
        else:
            istart = 0
        if np.all(xi <= self._xi_min):
            u = y0[0]*np.ones(xi.size)
            du = y0[1]*np.ones(xi.size)
        else:
            y = odeint(self._dydx, y0, xi)
            u = y[istart:, 0]
            du = y[istart:, 1]
        return u, du

    def get_radius(self, u0):
        """Calculates the dimensionless radial extent of a TES.

        The radial extent of a TES is the radius at which rho = rho_e.

        Parameters
        ----------
        u0 : float
            Dimensionless logarithmic central density

        Returns
        -------
        float
            Dimensionless radial extent
        """
        logxi0 = brentq(lambda x: self.solve(10**x, u0)[0],
                        np.log10(self._xi_min), np.log10(self._xi_max))
        return 10**logxi0

    def get_mass(self, u0, xi0=None):
        """Calculates dimensionless enclosed mass.

        The dimensionless mass enclosed within the dimensionless radius xi
        is defined by
            M(xi) = m(xi)M_{J,e}
        where M_{J,e} is the Jeans mass at the edge density rho_e

        Parameters
        ----------
        u0 : float
            Dimensionless logarithmic central density
        xi0 : float, optional
            Radius within which the enclosed mass is computed. If None, use
            the maximum radius of a sphere.

        Returns
        -------
        float
            Dimensionless enclosed mass.
        """
        if xi0 is None:
            xi0 = self.get_radius(u0)
        u, du = self.solve(xi0, u0)
        f = 1 + (xi0/self.xi_s)**(2*self.p)
        m = -(xi0**2*f*du + 2*self.p*(f-1)*xi0)/np.pi
        return m.squeeze()[()]

    def get_crit(self):
        """Finds critical TES parameters.

        Critical point is defined as a maximum point in the R-M curve.

        Returns
        -------
        u_c : float
            Critical logarithmic central density
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
        upper_bound = 8
        res = minimize_scalar(lambda x: -self.get_mass(x),
                              bounds=(0, upper_bound), method='Bounded')
        u_c = res.x
        r_c = self.get_radius(u_c)
        m_c = self.get_mass(u_c)
        if u_c >= 0.999*upper_bound:
            raise Exception("critical density contrast is out-of-bound")
        return u_c, r_c, m_c

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
        a = x**2*f
        b = 2*x*((1+self.p)*f - self.p)
        c = 2*self.p*(2*self.p+1)*(f-1) + 4*np.pi**2*x**2*np.exp(y1)
        dy2 = -(b/a)*y2 - (c/a)
        return np.array([dy1, dy2])


class TESm:
    """Turbulent equilibrium sphere of a fixed mass.

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
    """
    def __init__(self, p=0.5, xi_s=np.inf):
        self.p = p
        self.xi_s = xi_s
        self._xi_min = 1e-7
        self._xi_max = 1e3

    def solve(self, xi, u0):
        """Solve equilibrium equation

        Parameters
        ----------
        xi : array
            Dimensionless radii
        u0 : float
            Dimensionless logarithmic central density

        Returns
        -------
        u : array
            Log density u = log(rho/rho_e)
        du : array
            Derivative of u: d(u)/d(xi)
        """
        xi = np.array(xi, dtype='float64')
        y0 = np.array([u0, 0])
        if xi.min() > self._xi_min:
            xi = np.insert(xi, 0, self._xi_min)
            istart = 1
        else:
            istart = 0
        if np.all(xi <= self._xi_min):
            u = y0[0]*np.ones(xi.size)
            du = y0[1]*np.ones(xi.size)
        else:
            y = odeint(self._dydx, y0, xi)
            u = y[istart:, 0]
            du = y[istart:, 1]
        return u, du

    def get_crit(self):
        """Find critical TES

        Returns
        -------
        float
            Critical logarithmic central density
        """
        umin, umax = -2, 12
        res = minimize_scalar(lambda x: -self.get_rhoe(x),
                              bounds=(umin, umax), method='Bounded')
        u0_crit = res.x
        if np.isclose(u0_crit, umax):
            # Try expanding umax once.
            umax = 16
            res = minimize_scalar(lambda x: -self.get_rhoe(x),
                                  bounds=(umin, umax), method='Bounded')
            u0_crit = res.x
        if np.any(np.isclose(u0_crit, (umin, umax))):
            raise ValueError("There is no local maximum within "
                             "(umin, umax) = ({}, {})".format(umin, umax))
        return u0_crit

    def get_rhoe(self, u0):
        """Find edge density

        Parameters
        ----------
        u0 : float
            Dimensionless logarithmic central density

        Returns
        -------
        array
            Dimensionless edge density
        """
        rmax = self.get_radius(u0)
        u, du = self.solve(rmax, u0)
        return np.exp(u[-1])

    def get_radius(self, u0):
        """Calculates the dimensionless radial extent of a TES.

        Parameters
        ----------
        u0 : float
            Dimensionless logarithmic central density

        Returns
        -------
        float
            Dimensionless radial extent.
        """
        logxi0 = brentq(lambda x: self._get_mass(10**x, u0) - 1,
                        np.log10(self._xi_min), np.log10(self._xi_max))
        return 10**logxi0

    def _get_mass(self, xi0, u0):
        """Calculates dimensionless enclosed mass.

        For fixed mass TES family, the dimensionless enclosed mass
        must be unity by definition. This imposes boundary condition
        for the radial extent of the cloud.

        Parameters
        ----------
        xi0 : float
            Radius within which the enclosed mass is computed.
        u0 : float
            Dimensionless logarithmic central density

        Returns
        -------
        float
            Dimensionless enclosed mass.
        """
        u, du = self.solve(xi0, u0)
        f = 1 + (xi0/self.xi_s)**(2*self.p)
        m = -(xi0**2*f*du + 2*self.p*(f-1)*xi0)
        return m.squeeze()[()]

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
        a = x**2*f
        b = 2*x*((1+self.p)*f - self.p)
        c = 2*self.p*(2*self.p+1)*(f-1) + 4*np.pi*x**2*np.exp(y1)
        dy2 = -(b/a)*y2 - (c/a)
        return np.array([dy1, dy2])


class TESc:
    """Turbulent equilibrium sphere of a fixed central density.

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
    """
    def __init__(self, p=0.5, xi_s=np.inf):
        self.p = p
        self.xi_s = xi_s
        self._xi_min = 1e-5
        self._xi_max = 1e3

    def solve(self, xi):
        """Solve equilibrium equation

        Parameters
        ----------
        xi : array
            Dimensionless radii

        Returns
        -------
        u : array
            Log density u = log(rho/rho_c)
        du : array
            Derivative of u: d(u)/d(xi)
        """
        xi = np.array(xi, dtype='float64')
        y0 = np.array([0, 0])
        if xi.min() > self._xi_min:
            xi = np.insert(xi, 0, self._xi_min)
            istart = 1
        else:
            istart = 0
        if np.all(xi <= self._xi_min):
            u = y0[0]*np.ones(xi.size)
            du = y0[1]*np.ones(xi.size)
        else:
            y = odeint(self._dydx, y0, xi)
            u = y[istart:, 0]
            du = y[istart:, 1]
        return u, du

    def get_crit(self):
        """Find critical TES

        Returns
        -------
        float
            Critical radius
        """
        def func(xi0):
            menc = self.get_mass(xi0)
            u0_TESm = np.log(np.pi**3*menc**2)
            xi_s_TESm = self.xi_s / np.pi / menc
            tsm = TESm(xi_s=xi_s_TESm)
            u0_crit = tsm.get_crit()
            return u0_TESm - u0_crit
        logrmax = brentq(lambda x: func(10**x), 0, np.log10(32))
        return 10**logrmax

    def get_mass(self, xi0):
        """Calculates dimensionless enclosed mass.

        The dimensionless mass enclosed within the dimensionless radius xi
        is defined by
            M(xi) = m(xi)M_{J,c}
        where M_{J,c} is the Jeans mass at the central density rho_c

        Parameters
        ----------
        xi0 : float
            Radius within which the enclosed mass is computed. If None, use
            the maximum radius of a sphere.

        Returns
        -------
        float
            Dimensionless enclosed mass.
        """
        u, du = self.solve(xi0)
        f = 1 + (xi0/self.xi_s)**(2*self.p)
        m = -(xi0**2*f*du + 2*self.p*(f-1)*xi0)/np.pi
        return m.squeeze()[()]

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
        a = x**2*f
        b = 2*x*((1+self.p)*f - self.p)
        c = 2*self.p*(2*self.p+1)*(f-1) + 4*np.pi**2*x**2*np.exp(y1)
        dy2 = -(b/a)*y2 - (c/a)
        return np.array([dy1, dy2])


def get_critical_tes(rhoe, lmb_sonic, p=0.5):
    """
    Calculate critical turbulent equilibrium sphere

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
    LJ_e = rhoe**-0.5
    MJ_e = rhoe**-0.5
    xi_s = lmb_sonic / LJ_e
    tes = TESe(p, xi_s)
    rat, xi0, m = tes.get_crit()
    rhoc = rat*rhoe
    R = LJ_e*xi0
    M = MJ_e*m
    return rhoc, R, M


def get_pv_diagram(rsonic, u0s=None):
    """Construct p-v diagram of a TES

    Parameters
    ----------
    rsonic : float
        Dimensionless sonic radius with the TESm normalization.
    u0s : array, optional
        Logarithmic central densities

    Returns
    -------
    vol : float
        Dimensionless volume of a TES.
    prs : float
        Dimensionless boundary pressure of a TES.

    Notes
    -----
    dimensionless length = length / (GM / c_s^2)
    dimensionless density = density / (c_s^6 / G^3 / M^2)
    dimensionless pressure = pressure / (c_s^8 / G^3 / M^2)
    """
    if u0s is None:
        u0s = np.linspace(-2, 18)
    tsm = TESm(xi_s=rsonic)
    vol, prs = [], []
    for u0 in u0s:
        rmax = tsm.get_radius(u0)
        vol.append(4*np.pi/3*rmax**3)
        u, du = tsm.solve(rmax, u0)
        prs.append(np.exp(u[-1]))
    vol = np.array(vol)
    prs = np.array(prs)
    return vol, prs


def plot_pv_diagram_for_fixed_rhoc(rmax0, rsonic0):
    """Plot p-v diagram of a TES

    Parameters
    ----------
    rmax0 : float
        r_max / L_{J,c}
    rsonic0 : float
        r_s / L_{J,c}
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    ts = TESc(xi_s=rsonic0)
    menc0 = ts.get_mass(rmax0)

    # Plot the density profile
    plt.sca(axs[0, 0])
    rds = np.logspace(-2, 2)
    u, du = ts.solve(rds)
    plt.loglog(rds, np.exp(u), c='k')
    plt.axvline(rsonic0, ls=':', color='k', label=r'$r_\mathrm{sonic}/L_{J,c}$')
    u, du = ts.solve(rmax0)
    plt.plot(rmax0, np.exp(u[0]), 'r+', mew=2, ms=10)
    plt.xlabel(r'$r/L_{J,c}$')
    plt.ylabel(r'$\rho/\rho_c$')
    plt.legend()

    # Plot the r-M diagram
    plt.sca(axs[0, 1])
    rmax_arr = np.logspace(-1, 2)
    u, du = ts.solve(rmax_arr)
    rho_e = np.exp(u)
    M = ts.get_mass(rmax_arr)
    plt.loglog(rmax_arr, M, label=r'$M/M_{J,c}$')
    plt.loglog(rmax_arr, M*np.sqrt(rho_e), label=r'$M/M_{J,e}$')
    M = ts.get_mass(rmax0)
    u, du = ts.solve(rmax0)
    rho_e = np.exp(u[0])
    plt.plot(rmax0, M, 'r+', mew=2, ms=10)
    plt.plot(rmax0, M*np.sqrt(rho_e), 'r+', mew=2, ms=10)
    plt.xlabel(r'$R/L_{J,c}$')
    plt.ylabel('Mass')
    plt.legend(loc='upper left')

    # Construct a fixed-mass TES family corresponding to menc0
    rsonic = rsonic0 / np.pi / menc0
    tsm = TESm(xi_s=rsonic)

    # visualization
    plt.sca(axs[1, 0])
    # density profiles for a selected u0
    # initial condition corresponding to the unvaried sphere
    u00 = np.log(np.pi**3*menc0**2)
    for u0 in np.linspace(u00-2, u00+2, 5):
        rmax = tsm.get_radius(u0)
        r = np.logspace(-6, np.log10(rmax))
        u, du = tsm.solve(r, u0)
        color = 'r' if np.isclose(u0, u00) else 'k'
        lw = 2 if np.isclose(u0, u00) else 1
        plt.loglog(r, np.exp(u), color=color, lw=lw)
    plt.xlabel(r'$r/(GMc_s^{-2})$')
    plt.ylabel(r'$\rho/(c^6G^{-3}M^{-2})$')

    # P-V diagram
    vol, prs = get_pv_diagram(rsonic, np.linspace(u00-10, u00+10))
    plt.sca(axs[1, 1])
    plt.loglog(vol, prs, c='k')
    rmax = tsm.get_radius(u00)
    vol = 4*np.pi/3*rmax**3
    u, du = tsm.solve(rmax, u00)
    prs = np.exp(u[-1])
    plt.plot(vol, prs, 'r+', mew=2, ms=10)
    plt.xlabel(r'$V/(G^3M^3c_s^{-6})$')
    plt.ylabel(r'$P/(c_s^8G^{-3}M^{-2})$')
    plt.tight_layout()
    plt.xlim(vol/10, vol*10)
    plt.ylim(prs/10, prs*10)
    return fig


def plot_pv_diagram_for_fixed_pressure(logrhoc, rsonic0):
    """Plot p-v diagram of a TES

    Parameters
    ----------
    logrhoc : float
        log(rho_c / rho_e)
    rsonic0 : float
        r_s / L_{J,e}
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    ts = TESe(xi_s=rsonic0)
    menc0 = ts.get_mass(logrhoc)

    # Plot the density profile
    plt.sca(axs[0, 0])
    rds = np.logspace(-2, np.log10(ts.get_radius(logrhoc)))
    u, du = ts.solve(rds, logrhoc)
    plt.loglog(rds, np.exp(u))
    r = np.logspace(-2, 0)
    plt.loglog(r, 8.86/(4*np.pi**2*r**2), 'k--', label=r'$\rho_\mathrm{LP}$')
    plt.axvline(rsonic0, ls=':', label=r'$r_\mathrm{sonic}/L_{J,c}$')
    plt.xlabel(r'$r/L_{J,e}$')
    plt.ylabel(r'$\rho/\rho_e$')
    plt.xlim(1e-2, 1e0)
    plt.ylim(1e0, 2e3)
    plt.legend()

    # Plot the r-M diagram
    plt.sca(axs[0, 1])
    u0_arr = np.linspace(0.1, 15, 100)
    R, M = [], []
    for u0 in u0_arr:
        R.append(ts.get_radius(u0))
        M.append(ts.get_mass(u0))
    plt.plot(R, M)
    r0 = ts.get_radius(logrhoc)
    plt.plot(r0, menc0, 'r+', mew=2, ms=10)
    plt.xlabel(r'$R/L_{J,e}$')
    plt.ylabel(r'$M/M_{J,e}$')

    # Construct a fixed-mass TES family corresponding to menc0
    rsonic = rsonic0 / np.pi / menc0
    tsm = TESm(xi_s=rsonic)

    # visualization
    plt.sca(axs[1, 0])
    # density profiles for a selected u0
    # initial condition corresponding to the unvaried sphere
    u00 = logrhoc + np.log(np.pi**3*menc0**2)
    for u0 in np.linspace(u00-2, u00+2, 5):
        rmax = tsm.get_radius(u0)
        r = np.logspace(-6, np.log10(rmax))
        u, du = tsm.solve(r, u0)
        color = 'r' if np.isclose(u0, u00) else 'k'
        lw = 2 if np.isclose(u0, u00) else 1
        plt.loglog(r, np.exp(u), color=color, lw=lw)
    plt.xlabel(r'$r/(GMc_s^{-2})$')
    plt.ylabel(r'$\rho/(c^6G^{-3}M^{-2})$')

    # P-V diagram
    vol, prs = get_pv_diagram(rsonic, np.linspace(u00-10, u00+10))
    plt.sca(axs[1, 1])
    plt.loglog(vol, prs)
    rmax = tsm.get_radius(u00)
    vol = 4*np.pi/3*rmax**3
    u, du = tsm.solve(rmax, u00)
    prs = np.exp(u[-1])
    plt.plot(vol, prs, 'r+', mew=2, ms=10)
    plt.xlabel(r'$V/(G^3M^3c_s^{-6})$')
    plt.ylabel(r'$P/(c_s^8G^{-3}M^{-2})$')
    plt.tight_layout()
    plt.xlim(vol/10, vol*10)
    plt.ylim(prs/10, prs*10)
    return fig
