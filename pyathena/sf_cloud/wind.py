import os
import os.path as osp
import sys

import numpy as np
from scipy.optimize import fsolve,least_squares
from scipy.integrate import odeint,cumulative_trapezoid
import astropy.units as au
import astropy.constants as ac
import pandas as pd

from ..microphysics.rec_rate import RecRate
from ..util.mass_to_lum import mass_to_lum
from ..util.cloud import Cloud

######################################################################
# Model-dependent function definitions
# (e.g., mass loading, density profile, setting bounds for r_c, etc.)
######################################################################

# What if mdot \propto r^4 at small r?
# def mdot(r,p_mdot):
#     a, = p_mdot
#     return np.where(r > 1.0, 1.0, r**4/(r**2 + a**2*(1 - r)**2))

# def dlogmdot_dx(r,p_mdot):
#     a, = p_mdot
#     return np.where(r > 1.0, 0.0,
#                     2*(a**2*(r**2 - 3.0*r + 2.0) + r**2)/(r**2 + a**2*(1.0 - r)**2))

# def d2logmdot_dx2(r,p_mdot):
#     a, = p_mdot
#     return np.where(r > 1.0, 0.0,
#                     2*r*(a**4*(1 - r)**2 + a**2*(r - 2.0)*r)/(a**2*(1 - r)**2 + r**2)**2)

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
    f_star, r_star, k_rho = p_f
    return np.where(r <= 1.0,
                    f_star*r**3/(r**2 + r_star**2)**1.5 + (1.0 - f_star)*r**(3.0 - k_rho),
                    f_star*r**3/(r**2 + r_star**2)**1.5 + (1.0 - f_star))

def df_dr(r,p_f):
    f_star, r_star, k_rho = p_f
    return np.where(r <= 1.0,
                    f_star*3.0*r_star**2*r**2/(r**2 + r_star**2)**2.5 + (1 - f_star)*(3.0 - k_rho)*r**(2.0 - k_rho),
                    f_star*3.0*r_star**2*r**2/(r**2 + r_star**2)**2.5)

def set_bounds_and_r_c_guess1(r_g, p_mdot, p_f, r_c_guess=None):
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
    if np.isnan(root) or root < 0.0:
        print('r_c = {0:g}, func_r_c(r_c) = {1:g}'.\
              format(r_c,func_r_c(r_c, r_g, p_mdot, p_f)))
        print('dydx is NaN. r_c: {0:g}, r_g: {1:g}, root: {2:g}, b, c, b^2-4c=({3:g},{4:g},{5:g})'.\
              format(r_c,r_g,root,bb,cc,bb**2 - 4.0*cc))
        raise ValueError()
    return root

def solve_ode_one(r_g, p_mdot, p_f, x_c, dydx_c):

    
    odeint_kwargs = dict(atol=None, rtol=None, full_output=False)
    #, hmin=1e-4)
    
    Nx = 2000
    r_c = np.exp(x_c)
    xp = np.linspace(x_c + 1e-5, 1.5, Nx)
    xm = np.linspace(x_c - 1e-5,-4.0, Nx)
    yp = np.zeros_like(xp)
    yp0 = dydx_c*(xp[0] - x_c)
    ym0 = dydx_c*(xm[0] - x_c)

    yp = odeint(dydx, yp0, xp, args=(r_g,p_mdot,p_f), **odeint_kwargs).flatten()
    ym = odeint(dydx, ym0, xm, args=(r_g,p_mdot,p_f), **odeint_kwargs).flatten()

    # def fake_odeint(func, y0, t, Dfun=None):
    #     ig = ode(func, Dfun)
    #     ig.set_integrator('lsoda',
    #                       method='adams')
    #     ig.set_initial_value(y0, t=0.)
    #     y = []
    #     for tt in t:
    #         y.append(ig.integrate(tt))
    #         return np.array(y)

        
    x = np.append(np.append(np.flip(xm),x_c),xp)
    y = np.append(np.append(np.flip(ym),0.0),yp)
    r = np.exp(x)
    u = np.exp(y)
    
    # Density (unscaled since mdot is unscaled)
    rho = mdot(r,p_mdot)/(4.0*np.pi*r**2*u)
    idx_c = np.where(x >= x_c)[0][0]
    idx_1 = np.where(x >= 0.0)[0][0]
    idx_2 = np.where(r >= 2.0)[0][0]

    # Rescale density so that rho=1 at r=1. Alternatively, we can normalize it so that
    # rho=1 at the sonic point r_c, but we should carry around some other constants (e.g.,
    # mdot(r_c)) in scaling relations. It should make no practical difference anyway.
    rho = rho/rho[idx_1]
    
    # Velocity at r=1 and 2
    u1 = u[np.where(r >= 1.0)[0][0]]
    u2 = u[np.where(r >= 2.0)[0][0]]

    # Both integrals should give the same results
    int_rec = 3.0*u1**2*cumulative_trapezoid(mdot(r,p_mdot)**2/(r*u)**2,r,
                                             initial=0.0)
    int_rec_alt = 3.0*cumulative_trapezoid((r*rho)**2,r,initial=0.0)

    F_rec_m1 = int_rec[-1]
    F_rec_1 = int_rec[idx_1]
    F_rec_2 = int_rec[idx_2]
    F_rec_c = int_rec[idx_c]

    F_evap_m1 = u1*int_rec[-1]**-0.5
    F_evap_1 = u1*int_rec[idx_1]**-0.5
    F_evap_2 = u1*int_rec[idx_2]**-0.5
    F_evap_c = u1*int_rec[idx_c]**-0.5

    # r_g/R0 = vesc^2/(4*cion^2)
    vesc = 2.0*r_g**0.5

    res = dict(r_g=r_g,
               vesc=vesc,
               p_mdot=p_mdot,p_f=p_f,
               r=r,u=u,x=x,y=y,rho=rho,
               idx_1=idx_1,idx_2=idx_2,
               idx_c=idx_c,r_c=r_c,x_c=x_c,dydx_c=dydx_c,
               u1=u1,u2=u2,
               int_rec=int_rec,int_rec_alt=int_rec_alt,
               F_rec_m1=F_rec_m1,F_rec_1=F_rec_1,F_rec_2=F_rec_2,F_rec_c=F_rec_c,
               F_evap_m1=F_evap_m1,F_evap_1=F_evap_1,F_evap_2=F_evap_2,F_evap_c=F_evap_c,               
               )

    return res


def solve_odes(r_g, p_mdot_dict, p_f_dict,
               func_set_bounds_and_r_c_guess=set_bounds_and_r_c_guess1):
    """
    Function to get ode solution for arrays of input parameters

    Parameters
    ----------
    r_g : float or 1d array
        Dimensionless radius
    p_mdot_dict : dict
        Each item contains a float or 1d array for mdot parameters
    p_f_dict : dict
        Each item contains a float or 1d array for f parameters

    Returns
    -------
    (dict, pandas DataFrame)

    Example
    -------
    r_g = np.logspace(-2.5,0.5,99)
    p_mdot_dict = dict(a=1.0)
    p_f_dict = dict(f_star=0.05,
                    r_star=0.2,
                    k_rho=1.0)

    res, df = solve_odes(r_g, p_mdot_dict, p_f_dict)
    """    

    # Make float to arrays
    r_g = np.asarray(r_g)
    for k in p_mdot_dict.keys():
        p_mdot_dict[k] = np.broadcast_to(p_mdot_dict[k],r_g.shape)
    for k in p_f_dict.keys():
        p_f_dict[k] = np.broadcast_to(p_f_dict[k],r_g.shape)

    res = np.empty((len(r_g),), dtype=dict)
    res2 = dict()
    r_c_guess = None
    
    for i,r_g_ in enumerate(r_g):
        # print(r_g_,end=' ')
        p_mdot = tuple(v[i] for k,v in p_mdot_dict.items())
        p_f = tuple(v[i] for k,v in p_f_dict.items())

        p_find_r_c = (r_g_, p_mdot, p_f)
        r_c_ = find_r_c(set_bounds_and_r_c_guess1,
                        r_g_,p_mdot,p_f,r_c_guess=r_c_guess)

        # print('r_g_ ={0:g}, r_c = {1:g}, func_r_c(r_c) = {2:g}'.\
        #       format(r_g_,r_c_,func_r_c(r_c_, r_g_, p_mdot, p_f)))

        dydx_c_ = find_dydx_c(r_c_,r_g_,p_mdot,p_f)
        res[i] = solve_ode_one(r_g_, p_mdot, p_f, np.log(r_c_), dydx_c_)
        
        # Save to dictionaries
        if i == 0:
            for k in res[0].keys():
                if k == 'p_mdot':
                    for kk in p_mdot_dict.keys():
                        res2[kk] = []
                elif k == 'p_f':
                    for kk in p_f_dict.keys():
                        res2[kk] = []
                else:
                    res2[k] = []
            
        for k in res[i].keys():
            if k == 'p_mdot':
                for kk in p_mdot_dict.keys():
                    res2[kk].append(p_mdot_dict[kk][i])
            elif k == 'p_f':
                for kk in p_f_dict.keys():
                    res2[kk].append(p_f_dict[kk][i])
            else:
                res2[k].append(res[i][k])

    df = pd.DataFrame(res2)

    return res,df

def solve_odes_fig_mdot0(r_g=None, a=1.0, f_star=0.1, 
                         r_star=0.2, k_rho=1.0, force_override=False):
    """
    Solve ODEs for making figure on mdot0 (or F_ev)
    """
    
    if r_g is None:
        fname = './pickle/wind_ode_a{0:03d}_fstar{1:03d}_rstar{2:03d}_krho{3:03d}.p'.\
                        format(int(100.0*a),int(100.0*f_star),int(100.0*r_star),int(100.0*k_rho))
    else:
        fname = './pickle/wind_ode_a{0:03d}_fstar{1:03d}_rg{2:03d}_krho{3:03d}.p'.\
                        format(int(100.0*a),int(100.0*f_star),int(100.0*r_g),int(100.0*k_rho))

    if osp.exists(fname) and not force_override:
        df = pd.read_pickle(fname)
        return df
    
    if r_g is None:
        vesc_orig = np.linspace(0.04,3.0,200)
        r_g_orig = 0.25*vesc_orig**2
        if a < 1.0:
            r_g = np.delete(r_g_orig,
                            np.where(np.logical_and(r_g_orig > 0.98,
                                                    r_g_orig < 1.08))[0])
        else:
            r_g = r_g_orig
    else:
        r_g = np.repeat(r_g,r_star.shape)
    
    p_mdot_dict = dict(a=a)
    p_f_dict = dict(f_star=f_star, r_star=r_star, k_rho=k_rho)
    res,df = solve_odes(r_g, p_mdot_dict, p_f_dict)
    df.to_pickle(fname)
    
    return df


class CloudEvap(Cloud):

    mH = 1.008*ac.u
    muH = (1.4*mH).to('gram')
    Xi0 = 5.0e46/au.s/au.M_sun

    def __init__(self, M=None, R=None, Sigma=None, alpha_vir=4.0,
                 Ti=8000*au.K, age_const=True, 
                 a=2.0, r_star=0.2, k_rho=1.0, dt_factor=100.0):

        super(CloudEvap, self).__init__(M=M, R=R, Sigma=Sigma, alpha_vir=alpha_vir)
        
        self.Ti = Ti
        self.alphaB = RecRate().get_rec_rate_H_caseB_Dr11(self.Ti.value)*au.cm**3/au.s
        # self.alphaB = 3.0e-13*(self.Ti/(8e3*au.K))**-0.7*au.cm**-3/au.s
        self.ci = (np.sqrt(2.1*ac.k_B*self.Ti/(self.muH))).to('cm s-1')
        self.Sigma_ev0 = (CloudEvap.muH*self.ci*(CloudEvap.Xi0/(8.0*self.alphaB*ac.G))**0.5).to('Msun pc-2')
        self.r_g0 = ((ac.G*self.M/(2.0*self.ci**2)).to('pc')/self.R).value
        
        self._initialize_arrays()
        self.dt = self.tff/dt_factor
        
        self.mtl = mass_to_lum(model='SB99')
        self.age_max_sb99 = 40.0*au.Myr
        self.age_const = age_const
        
        self.a = a
        self.r_star = r_star
        self.k_rho = k_rho
        self.fion = 1.0
        
    def _initialize_arrays(self):
        self.time = np.array((0,),dtype=au.Quantity)*au.Myr
        self.eps_neu = np.array((1,))
        self.eps_star = np.zeros((1,))
        self.deps_star = np.zeros((1,))
        self.eps_ev = np.zeros((1,))
        self.deps_ev = np.zeros((1,))
        self.F_evap_m1 = np.zeros((1,))

        self.Qi = np.zeros((1,),dtype=au.Quantity)/au.s
        self.Qieff = np.zeros((1,),dtype=au.Quantity)/au.s

    @staticmethod
    def calc_eps_ff(alpha_vir, b=2.02):
        return np.exp(-b*alpha_vir**0.5)
    
    def calc_fion(self):
        return self.fion
    
    def evolve_dt(self):
        
        dt = self.dt
        t0 = self.time[-1]
        t1 = t0 + dt
        
        # Values at the last time step
        eps_neu = self.eps_neu[-1]
        eps_star = self.eps_star[-1]
        tff = self.tff
        alpha_vir = self.alpha_vir
        
        # Increment of stellar mass
        eps_ff = self.calc_eps_ff(alpha_vir)
        deps_star = (eps_ff*eps_neu/tff*dt).value
        
        self.time = np.append(self.time, t1)
        self.eps_neu = np.append(self.eps_neu, self.eps_neu[-1] - deps_star)
        self.eps_star = np.append(self.eps_star, self.eps_star[-1] + deps_star)
        self.deps_star = np.append(self.deps_star, deps_star)
        
        # Calculate age and Qi
        if self.age_const:
            age_sb99 = 1e-4
        else:
            age_sb99 = np.where(t1 - self.time > self.age_max_sb99,
                                self.age_max_sb99,
                                t1 - self.time).value
        
        Qi = self.mtl.calc_Qi_SB99(self.deps_star*self.M.value, age=age_sb99).sum()/au.s
        fion = self.calc_fion()
        Qieff = Qi*self.fion
        self.Qi = np.append(self.Qi, Qi)
        self.Qieff = np.append(self.Qieff, Qieff)

        # Solve ODE solution and calculate the mass loss rate
        res = self.solve_wind_ode()
        mdot0 = (self.muH*self.ci*(12.0*np.pi*Qieff*self.R/self.alphaB)**0.5).\
                    to('Msun Myr-1')*res['F_evap_m1'] # Mass loss rate when F_ev=1
        deps_ev = (mdot0*dt/self.M).value
        self.deps_ev = np.append(self.deps_ev, deps_ev)
        self.eps_ev = np.append(self.eps_ev, self.eps_ev[-1] + deps_ev)
        #self.deps_ev = np.append(self.eps_ev, self.eps_ev[-1] + deps_ev)
        self.F_evap_m1 = np.append(self.F_evap_m1, res['F_evap_m1'])
        
        # Decrease neutral gas mass
        self.eps_neu[-1] -= deps_ev
        
    def solve_wind_ode(self):
        # Set parameters
        r_g = self.r_g0*(self.eps_star[-1] + self.eps_neu[-1])
        f_star = self.eps_star[-1]/(self.eps_star[-1] + self.eps_neu[-1]) 
        p_mdot = (self.a,)
        p_f = (f_star,self.r_star,self.k_rho)
        p_find_r_c = (r_g, p_mdot, p_f)
        r_c = find_r_c(set_bounds_and_r_c_guess1,
                        r_g,p_mdot,p_f,r_c_guess=None)
        dydx_c = find_dydx_c(r_c,r_g,p_mdot,p_f)
        self.res = solve_ode_one(r_g, p_mdot, p_f, np.log(r_c), dydx_c)
        
        return self.res

    def evolve(self, tlim=50.0*au.Myr):
        
        while True:
            self.evolve_dt()
            if self.time[-1] > tlim or self.eps_neu[-1] < 0.01:
                break

# def get_scaled_result(r, Qieff, R0, idx, Tion=8000*au.K):
#     from copy import deepcopy
    
#     rr = deepcopy(r)
#     mH = 1.008*ac.u
#     muH = (1.4*mH).to('gram')
#     cion = (np.sqrt(2.1*ac.k_B*Tion/muH)).to('km s-1')

#     if not isinstance(Qieff, au.quantity.Quantity):
#         Qieff = Qieff/au.s
#     if not isinstance(R0, au.quantity.Quantity):
#         R0 = R0*au.pc

#     alphaB = rec_rate.RecRate().get_rec_rate_H_caseB_Dr11(Tion.value)*au.cm**3/au.s

#     rr['Qieff'] = Qieff
#     rr['nion0'] = (((3.0*Qieff)/(4.0*np.pi*R0**3*rr['int_rec'][idx]*alphaB))**0.5).to('cm-3')
#     rr['mdot0'] = (4.0*np.pi*R0**2*(muH*rr['nion0'])*rr['u1']*cion).to('Msun Myr-1')
    
    # rr['intrec'] = 3.0*ucumulative_trapezoid()
    # # Total recombination rate as a function r (unscaled since rho is unscaled)
    # rr['recomb_cumul'] = cumulative_trapezoid(4.0*np.pi*rr['r']**2*rr['rho']**2,
    #                     rr['r'],initial=0.0)*R0.cgs.value**3*alphaB
    # scale_rec = (rr['Qi']/rr['recomb_cumul'][idx]).value

    # # Rescale density such that Qi is just enough to balance recombination up to r_c
    # rr['cumulative_rec_c'] = scale_rec*rr['recomb_cumul']
    # # Assume clumping factor of order unity
    # scale_rho = scale_rec**0.5
    # rr['ni'] = rr['rho']*scale_rho/au.cm**3
    
    # # Number density of ionized gas at r_c and R_0
    # rr['ni_c'] = rr['ni'][rr['idx_c']]
    
    # # Mass loss rate
    # # rr['mdot_c'] = (4*np.pi*(rr['r_c']*R0_pc*au.pc)**2*ci*(rr['ni_c']*muH)).to('Msun Myr-1')
    # rr['mdot'] = (4*np.pi*(rr['r']*R0)**2*(ci*rr['u'])*(rr['ni']*muH)).to('Msun Myr-1')
    # rr['mdot0'] = rr['mdot'].max()

#     # Time required to evaporate gas mass (1-eps_star)*M0, assuming constant mdot
#     rr['tevap_c'] = (1.0 - eps_star)*self.M/rr['mdot_c']
#     # SFE during tevap, assuming constant eps_ff
#     rr['SFE_tevap_c'] = self.calc_eps_ff(self.alpha_vir)*rr['tevap_c']/self.tff
#     # Velocity at 2*R0
#     rr['u_at_2'] = rr['u'][np.where(rr['r'] >= 2.0)[0][0]]
#     self.rr = rr
    
    # return rr
