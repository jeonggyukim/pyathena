import os
import os.path as osp
import astropy.units as au
import astropy.constants as ac
import numpy as np
import pandas as pd
import cPickle as pickle
import matplotlib.pyplot as plt
from scipy.integrate import odeint,ode
from scipy.special import erf

from ..util.cloud import Cloud

# def rk4(x, h, y, f):
#     k1 = h * f(x, y)
#     k2 = h * f(x + 0.5*h, y + 0.5*k1)
#     k3 = h * f(x + 0.5*h, y + 0.5*k2)
#     k4 = h * f(x + h, y + k3)
#     return y + (k1 + 2.0*(k2 + k3) + k4)/6.0


class CloudEvol(Cloud):
    def __init__(self, M=None, R=None, Sigma=None, alpha_vir=2.0,
                 SFE=None, sigmad=1e-21, eps_ff=0.1, sigma_lnSigma=1.0,
                 p=0.0, Mach=None, radp=True, eta=0.2, selfg_shell=True):
    """Thompson & Krumholz (2016) model
    """

        super(Cloud_evol, self).__init__(M=M, R=R, Sigma=Sigma, alpha_vir=alpha_vir)

        # width of log-normal surface-density distribution
        if Mach is not None:
            self.sigma_lnSigma = self.calc_sigma_lnSigma(Mach)
        else:
            self.sigma_lnSigma = sigma_lnSigma

        self.eps_ff = eps_ff
        self.radp = radp
        self.phot = phot

        self.selfg_shell = selfg_shell

        # initial value of ycrit
        self.ycrit_M = -np.inf
        self.ycrit_A = -np.inf
        self.Sigma_Edd = 0.0*au.Msun/au.pc**2
        # instantaneous gas fraction M_gas/(M_gas + M_*)
        self.f_gas = 1.0
        self.Lrec = eta*self.R
        self.SFE = 0.0
        self.p=p # see Thompson & Krumholz, R=R_0*[(M_*+M_gas)/M]**p
        self.eps_ej=0.0 # ejection efficiency  M_of/M
        if self.phot:
            self.nHII=0.0/au.cm**3
            self.rhoII=0.0*au.gram/au.cm**3

    @staticmethod
    def calc_sigma_lnSigma(Mach):
        """Eq. 12 in Thompson & Krumholz
        """
        alpha = 2.5
        R = 0.5*(3.0 - alpha)/(2.0 - alpha)*\
            (1.0 - Mach**(2.0*(2.0 - alpha)))/(1.0 - Mach**(2.0*(3.0 - alpha)))
        return np.sqrt(np.log(1.0 + R*Mach**2/4.0))

    def get_ycrit(self):
        if self.SFE > 0.0:
            # Mean value of (circum-cluster) surface density
            #self.Sigma_c_mean=(1.0 - self.SFE)*self.Sigma/4.0
            self.Sigma_c_mean=self.f_gas*(1.0 - self.eps_ej)**(1.0 - 2.0*self.p)*self.Sigma
            # Mean of the mass distribution of ln Sigma_c
            mu_M=np.log(self.Sigma_c_mean.value) + 0.5*self.sigma_lnSigma**2
            mu_A=np.log(self.Sigma_c_mean.value) - 0.5*self.sigma_lnSigma**2
            self.ycrit_M=(np.log(self.Sigma_crit.value) - mu_M)/(np.sqrt(2.0)*self.sigma_lnSigma)
            self.ycrit_A=(np.log(self.Sigma_crit.value) - mu_A)/(np.sqrt(2.0)*self.sigma_lnSigma)
        else:
            self.ycrit_M=-np.inf
            self.ycrit_A=-np.inf

    def update_state(self,M_gas,M_star,M_of):
        # update stellar mass, luminosity, SFE, gas fraction
        self.set_SFE(M_star/self.M.value,iPsi=0)
        self.f_gas=M_gas/(M_star + M_gas)
        self.eps_ej=M_of/self.M.value
        # Case 1: rad pressure only
        # Compute Eddington surface density
        if self.radp and not self.phot:
            # Sigma_E from Raskutti et al.
            #self.Sigma_Edd=(self.SFE/(1.0 + self.SFE)*self.Psi/(2.0*np.pi*ac.c*ac.G)).to('Msun/pc**2')
            # Sigma_E from Thompson & Krumholz

            if not self.shell_selfgrav:
            # dv/dt=-GM_*/r^2 - GM_gas/r^2 + L/(4 pi r^2 c Sigma)
                self.Sigma_Edd=(M_star/(M_star + M_gas)*self.Psi/(4.0*np.pi*ac.c*ac.G)).to('Msun/pc**2')
            # If we consider self-gravity
            else:
                self.Sigma_Edd=(M_star/(M_star + 0.5*M_gas)*self.Psi/(4.0*np.pi*ac.c*ac.G)).to('Msun/pc**2')
            self.Sigma_crit=self.Sigma_Edd
            self.get_ycrit()
        elif not self.radp and self.phot:
        # Case 2: photoionization only
            if self.SFE > 0.0:
                self.Sigma_crit=(1.0/(self.f_gas + 2.0*self.SFE)*self.rhoII*self.cion**2/(np.pi*ac.G*self.Sigma)).to('Msun/pc**2')
                self.get_ycrit()
            else:
                self.sigma_tau=np.inf

            self.Phi=(self.Qi/(4.0*np.pi*self.R**2)).to('cm**(-2)/s')
            self.nHII=np.sqrt(self.Phi/(self.alphaB*self.Lrec)).to('cm**(-3)')
            self.rhoII=self.muH*self.nHII

    @staticmethod
    def ODE(t,y,p):
        cl=p['cl']
        tff0=cl.tff.value
        if cl.radp and not cl.phot:
            M_gas,M_star,M_of=y
            Mdot_of=M_gas*0.5*(1.0 + erf(cl.ycrit_M))/tff0
            Mdot_star=M_gas*cl.eps_ff/tff0
            Mdot_gas=-Mdot_star-Mdot_of
            dydt=[Mdot_gas,Mdot_star,Mdot_of]
        elif not cl.radp and cl.phot:
            M_gas,M_star,M_evap=y
            f_esc=0.0
            #f_esc=0.5*(1.0 + erf(cl.ycrit_A))
            Mdot_evap=(cl.rhoII*cl.cion*(cl.R**2)*cl.f_gas).to('Msun/Myr').value
            Mdot_star=M_gas*cl.eps_ff/tff0
            Mdot_gas=-Mdot_star-Mdot_evap
            dydt=[Mdot_gas,Mdot_star,Mdot_evap]

        cl.update_state(M_gas,M_star,M_of)
        #print cl.ycrit_M,(1.0 - 0.5*(1.0 + erf(cl.ycrit_M)))
        return dydt

    def evol_ODE(self):
        """
        rad pressure only
        y=[M_gas,M_star,M_of]
        photevap only
        y=[M_gas,M_star,M_evap]
        """
        # set initial condition
        y0=[self.M.value,0.0,0.0]
        tff=self.tff.value
        # integration time range (in units of free-fall time)
        tau0=0.0 ; t0=tau0*self.tff.value
        tau1=20.0 ; t1=tau1*self.tff.value

        N=100
        tau=np.linspace(tau0, tau1, N)
        t=tau*self.tff.value
        sol=np.empty((N, 3))
        sol[0]=y0
        ycrit_M=np.empty(N)
        ycrit_M[0]=-np.inf

        p=dict(cl=self)
        solver=ode(Cloud_evol.ODE)
        #solver.set_integrator('lsoda',rtol=1e-6,atol=1e-6*self.M.value)
        solver.set_integrator('dopri5',rtol=5e-3,verbosity=1)
        solver.set_initial_value(y0,t0)
        solver.set_f_params(p)

        # Repeatedly call the `integrate` method to advance the
        # solution to time t[k], and save the solution in sol[k].
        k=1
        while solver.successful() and solver.t < t[-1]:
            solver.integrate(t[k])
            sol[k]=solver.y

            ycrit_M[k]=self.ycrit_M
            # quit if gas fraction is less than 0.5%
            if sol[k][0]/self.M.value < 5e-3:
                break
            k+=1

        if self.radp and not self.phot:
            self.hst=pd.DataFrame([t[0:k],tau[0:k],sol[0:k,0],sol[0:k,1],sol[0:k,2],
                                  ycrit_M[0:k]],
                                  index=['time','tau','M_gas','M_star','M_of',
                                         'ycrit_M']).T
        elif not self.radp and self.phot:
            self.hst=pd.DataFrame([t[0:k],tau[0:k],sol[0:k,0],sol[0:k,1],sol[0:k,2]],
                                  index=['time','tau','M_gas','M_star','M_evap']).T

        self.SFE_final=self.hst.M_star.iloc[-1]/self.M.value

    def plt_sol(self):
        hst=self.hst
        M=self.M.value
        plt.plot(hst.tau,hst.M_gas/M,'b-')
        plt.plot(hst.tau,hst.M_star/M,'g-')
        if self.radp:
            plt.plot(hst.tau,hst.M_of/M,'r-')
        if self.phot:
            plt.plot(hst.tau,hst.M_evap/M,'r-')

        plt.ylim(bottom=0.0)


def calc_cloud_evol(radp=True, eps_ff=0.2, p=0.0,
                    sigma_lnSigma=1.0,M=1e5,eta=0.2,Mach=None,
                    selfg_shell=True, force_override=False, verbose=False):
    """
    radp, phot: turn on radiation pressure and photoionization
    sigma_lnSigma: width of lognormal
    eps_ff: star formation efficiency per free-fall time
    """

    if radp and not phot:
        feedback='radp'
    elif not radp and phot:
        feedback='phot'
    elif radp and phot:
        feedback='phrp'

    Sigma=10.0**np.linspace(1.0,np.log10(2000.0),25)
    sigma=sigma_lnSigma # width of lognormal

    if Mach is not None:
        fpkl=osp.join(osp.expanduser('~'),
                        'Dropbox/gmc/notes/ms-science-01/ms2/cloud-evol/',
                        '{0:s}'.format(feedback) +
                        '-Mach{0:3.1f}-SFRff{1:4.2f}-logM{2:3.1f}'.format(Mach,eps_ff,np.log10(M)))
    else:
        fpkl=osp.join(osp.expanduser('~'),
                        'Dropbox/gmc/notes/ms-science-01/ms2/cloud-evol/',
                        '{0:s}'.format(feedback) +
                        '-sigma{0:3.1f}-SFRff{1:4.2f}-logM{2:3.1f}'.format(sigma,eps_ff,np.log10(M)))

    if not shell_selfgrav:
        fpkl += '-no-selfgrav'

    if phot:
        fpkl += '-eta{0:4.2f}'.format(eta)
    fpkl += '.p'

    print fpkl
    if not force_override and osp.isfile(fpkl):
        dat=pickle.load(open(fpkl,'rb'))
        if verbose:
            print '[calc_cloud_evol]: read from pickle {0:s}'.format(fpkl)

        return dat

    # If here, need to calculate cloud evolution
    dat={}
    dat['cl']=[]
    dat['SFE_final']=[]
    dat['sigma_lnSigma'] = sigma_lnSigma
    dat['M']=M
    dat['Sigma']=Sigma
    dat['eps_ff']=eps_ff
    dat['feedback']=feedback

    if Mach is not None:
        dat['sigma_lnSigma']=Cloud_evol.calc_sigma_lnSigma(Mach)

    print 'M',M,'Sigma',
    for S in Sigma:
        print S,
        cl=Cloud_evol(M=M, Sigma=S, eps_ff=eps_ff, p=p, sigma_lnSigma=sigma_lnSigma,
                      Mach=Mach, radp=radp, selfg_shell=selfg_shell)
        cl.evol_ODE()
        dat['cl'].append(cl)

    for cl in dat['cl']:
        dat['SFE_final'].append(cl.SFE_final)
    dat['SFE_final']=np.array(dat['SFE_final'])

    pickle.dump(dat,open(fpkl,'wb'),pickle.HIGHEST_PROTOCOL)
    return dat
