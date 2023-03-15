import numpy as np
from scipy.optimize import fsolve,least_squares
from scipy.integrate import odeint,cumulative_trapezoid
import astropy.units as au
import astropy.constants as ac

from ..util.cloud import Cloud

######################################################################
# Model-dependent function definitions
# (e.g., mass loading, density profile, setting bounds for r_c, etc.)
######################################################################

def mdot(r,p_mdot):
    a, = p_mdot
    return np.where(r > 1.0, 1.0, r**2/(r**2 + a**2*(1 - r)**2))

def dmdot_dr(r,p_mdot):
    a, = p_mdot
    return np.where(r > 1.0, 0.0,
                    2*a**2*r*(1-r)/(r**2 + a**2*(1 - r)**2)**2)

def d2mdot_dr2(r,p_mdot):
    a, = p_mdot
    return np.where(r > 1.0, 0.0,
                    2*(a**4*(1-r)**2*(2*r+1) + a**2*r**2*(2*r-3))/(r**2 + a**2*(1 - r)**2)**3)

def dlogmdot_dx(r,p_mdot):
    a, = p_mdot
    return np.where(r > 1.0, 0.0,
                    2*a**2*(1.0 - r)/(r**2 + a**2*(1.0 - r)**2))

def d2logmdot_dx2(r,p_mdot):
    a, = p_mdot
    return np.where(r > 1.0, 0.0,
                    r*(2*a**2*(a**2*(1 - r)**2 + (-2 + r)*r))/(a**2*(1 - r)**2 + r**2)**2)

def f(r,p_f):
    eps_star, eps_gas, r_star, k_rho = p_f
    return np.where(r <= 1.0,
                    eps_star*r**3/(r**2 + r_star**2)**1.5 + eps_gas*r**(3.0 - k_rho),
                    eps_star*r**3/(r**2 + r_star**2)**1.5 + eps_gas)

def df_dr(r,p_f):
    eps_star, eps_gas, r_star, k_rho = p_f
    return np.where(r <= 1.0,
                    eps_star*3.0*r_star**2*r**2/(r**2 + r_star**2)**2.5 + eps_gas*(3.0 - k_rho)*r**(2.0 - k_rho),
                    eps_star*3.0*r_star**2*r**2/(r**2 + r_star**2)**2.5)

def set_bounds_and_r_c_guess(r_g, p_mdot, p_f, r_c_guess=None):
    a, = p_mdot
    bounds = (a/(1.0 + a**2)**0.5, max(1.0,r_g + 2e-1))
    if r_c_guess is None:
        if r_g > 1.0:
            r_c_guess = r_g
        else:
            r_c_guess = bounds[0]+1e-2

    return bounds,r_c_guess


###############################
# General function definitions
###############################

def q(r,p_mdot):
    return dmdot_dr(r,p_mdot)/(4.0*np.pi*r**2)

def dydx(y, x, r_g, p_mdot, p_f):
    return (-(1 + np.exp(2*y))*dlogmdot_dx(np.exp(x),p_mdot) +
            2*(1 - r_g*np.exp(-x)*f(np.exp(x),p_f))) / (np.exp(2*y) - 1.0)

def func_r_c(r, r_g, p_mdot, p_f):
    return 1.0 - dlogmdot_dx(r,p_mdot) - r_g/r*f(r,p_f)

def find_r_c(func_set_bounds_and_r_c_guess,
             r_g, p_mdot, p_f, r_c_guess=None):
    # Setting bounds depends on the functional form of mdot
    bounds, r_c_guess = func_set_bounds_and_r_c_guess(r_g, p_mdot, p_f, r_c_guess=r_c_guess)
    sol = least_squares(func_r_c,r_c_guess,bounds=bounds,
                        args=(r_g,p_mdot,p_f))
    # print(sol.cost)
    return float(sol.x)

def find_dydx_c(r_c,r_g,p_mdot,p_f):
    aa = 1.0
    bb = dlogmdot_dx(r_c,p_mdot)
    cc = d2logmdot_dx2(r_c,p_mdot) + r_g*(df_dr(r_c, p_f) - f(r_c, p_f)/r_c)
    root = (-bb + (bb**2 - 4.0*aa*cc)**0.5)/(2.0*aa)
    if np.isnan(root):
        raise ValueError('dydx is NaN. r_c: {0:g}, r_g: {1:g}'.format(r_c,r_g))
    return root

def solve_ode(r_g, p_mdot, p_f, x_c, dydx_c):
    Nx = 1000
    r_c = np.exp(x_c)
    xp = np.linspace(x_c + 1e-3, 1.0, Nx)
    xm = np.linspace(x_c - 1e-3,-4.0, Nx)
    yp = np.zeros_like(xp)
    yp0 = dydx_c*(xp[0] - x_c)
    ym0 = dydx_c*(xm[0] - x_c)
    yp = odeint(dydx, yp0, xp, args=(r_g,p_mdot,p_f)).flatten()
    ym = odeint(dydx, ym0, xm, args=(r_g,p_mdot,p_f)).flatten()
    x = np.append(np.append(np.flip(xm),x_c),xp)
    y = np.append(np.append(np.flip(ym),0.0),yp)
    r = np.exp(x)
    u = np.exp(y)
    
    # Density (unscaled since mdot is unscaled)
    rho = mdot(r,p_mdot)/(4.0*np.pi*r**2*u)
    idx_c = np.where(x >= x_c)[0][0]
    idx_0 = np.where(x >= 0.0)[0][0]
    
    # Velocity at r=1 and 2
    u1 = u[np.where(r >= 1.0)[0][0]]
    u2 = u[np.where(r >= 2.0)[0][0]]
    
    res = dict(r_g=r_g,p_mdot=p_mdot,p_f=p_f,
               r=r,u=u,x=x,y=y,rho=rho,
               idx_0=idx_0,
               idx_c=idx_c,r_c=r_c,x_c=x_c,dydx_c=dydx_c,
               u1=u1,u2=u2)

    return res

def rescale_density(r, Qi, R0, idx, Ti=8000*au.K):
    from copy import deepcopy
    
    rr = deepcopy(r)
    mH = 1.008*ac.u
    muH = (1.4*mH).to('gram')
    ci = (np.sqrt(2.1*ac.k_B*Ti/(muH))).to('cm s-1')
    alphaB = 2.7e-13*(Ti/(1e4*au.K))**-0.7*au.cm**-3/au.s

    rr['Qi'] = Qi
    # Total recombination rate as a function r (unscaled since rho is unscaled)
    rr['recomb_cumul'] = cumulative_trapezoid(4.0*np.pi*rr['r']**2*rr['rho']**2,
                        rr['r'],initial=0.0)*R0.cgs.value**3*alphaB
    scale_rec = (rr['Qi']/rr['recomb_cumul'][idx]).value

    # Rescale density such that Qi is just enough to balance recombination up to r_c
    rr['cumulative_rec_c'] = scale_rec*rr['recomb_cumul']
    # Assume clumping factor of order unity
    scale_rho = scale_rec**0.5
    rr['ni'] = rr['rho']*scale_rho/au.cm**3
    
    # Number density of ionized gas at r_c and R_0
    rr['ni_c'] = rr['ni'][rr['idx_c']]
    
    # Mass loss rate
    # rr['mdot_c'] = (4*np.pi*(rr['r_c']*R0_pc*au.pc)**2*ci*(rr['ni_c']*muH)).to('Msun Myr-1')
    rr['mdot'] = (4*np.pi*(rr['r']*R0)**2*(ci*rr['u'])*(rr['ni']*muH)).to('Msun Myr-1')
    rr['mdot0'] = rr['mdot'].max()

#     # Time required to evaporate gas mass (1-eps_star)*M0, assuming constant mdot
#     rr['tevap_c'] = (1.0 - eps_star)*self.M/rr['mdot_c']
#     # SFE during tevap, assuming constant eps_ff
#     rr['SFE_tevap_c'] = self.calc_eps_ff(self.alpha_vir)*rr['tevap_c']/self.tff
#     # Velocity at 2*R0
#     rr['u_at_2'] = rr['u'][np.where(rr['r'] >= 2.0)[0][0]]
#     self.rr = rr
    
    return rr
