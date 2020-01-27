import numpy as np
import matplotlib.pyplot as plt

import astropy.units as au
import astropy.constants as ac
# from radps_sp.mass_to_lum import _mass_to_lum
# import hii

class Cloud(object):
    """
    Simple class for spherical clouds

    Initialize by giving two of M, R, or Sigma (in units of M_sun, pc)
    """
    mH = ac.u*1.008          # Mass of a hydrogen atom
    # Particle mass per hydrogen atom (in fully atomic gas)
    muH = 1.4*mH.to(au.gram)

    def __init__(self, M=None, R=None, Sigma=None, alpha_vir=2.0):
        if M is not None and Sigma is not None and R is not None:
            raise ValueError('Exactly two of M, R, Sigma must be defined')
        # Check if input is dimensionless
        if M is not None and not isinstance(M, au.quantity.Quantity):
            M = (M*ac.M_sun).to(au.M_sun)
        if R is not None and not isinstance(R, au.quantity.Quantity):
            R = (R*ac.pc).to(au.pc)
        if Sigma is not None and not isinstance(Sigma, au.quantity.Quantity):
            Sigma = (Sigma*ac.M_sun/ac.pc**2).to(ac.M_sun/ac.pc**2)
        if M is not None and Sigma is not None:  # M and Sigma are given
            self.M = M
            self.Sigma = Sigma
            self.R = np.sqrt(M/(np.pi*Sigma))
        elif M is not None and R is not None:  # M and R are given
            self.M = M
            self.R = R
            self.Sigma = M/(np.pi*R**2)

        self.M = self.M.to(au.Msun)
        self.R = self.R.to(au.pc)
        self.Sigma = self.Sigma.to(au.Msun/au.pc**2)

        self.rho = Cloud.calc_rho(self.M, self.R).to(au.g/au.cm**3)
        self.nH = (self.rho/Cloud.muH).to(au.cm**(-3))
        self.tff = Cloud.calc_tff(self.rho)

        self.alpha_vir = alpha_vir
        self.vesc = (np.sqrt((2.0*ac.G*self.M)/self.R)).to(au.km/au.s)
        self.sigma1d = np.sqrt(self.alpha_vir/10.0)*self.vesc

    def __str__(self):
        if not self.M.shape:
            return 'Cloud object: M={0:<5g}, R={1:<5g},'\
                   'Sigma={2:<5g}, nH={3:<5g}, tff={4:<5g}'.format(
                       self.M, self.R, self.Sigma, self.nH, self.tff)
        else:
            return 'Cloud objects with shape {0:s}'.format(self.M.shape)

    @staticmethod
    def calc_rho(M, R):
        return M/(4.0*np.pi/3.0*R**3)

    @staticmethod
    def calc_tff(rho):
        return np.sqrt((3.0*np.pi)/(32.0*ac.G*rho)).to(au.Myr)

    @staticmethod
    def calc_Sigma_from_M_vesc(M, vesc):
        return ((vesc**2/(2.0*ac.G))**2/(np.pi*M)).to(au.M_sun/au.pc**2)

    @staticmethod
    def calc_Sigma_from_M_tff0(M, tff0):
        return ((np.pi*M/64.0)**(1.0/3.0)/(tff0**2*ac.G)**(2.0/3.0)).to(au.M_sun/au.pc**2)


# class Cloud_HII(Cloud):
#     def __init__(self, M=None, R=None, Sigma=None, alpha_vir=2.0,
#                  SFE=None, Tion=8.0e3, sigmad=1e-21):
#         Cloud.__init__(self, M=M, R=R, Sigma=Sigma, alpha_vir=alpha_vir)
#         if SFE is not None:
#             self.SFE = self.set_SFE(SFE)

#         self.Tion = Tion*au.K
#         self.cion = (np.sqrt(2.1*ac.k_B*self.Tion/Cloud.muH)).cgs
#         self.alphaB = hii.alphaB(self.Tion.value)*au.cm**3/au.s
#         self.sigmad = sigmad*au.cm**2

#     def set_SFE(self, SFE, iPsi=0):
#         self.SFE = SFE
#         self.Mstar = self.SFE*self.M
#         self.set_L_Qi(SFE, iPsi=iPsi)
#         self.calc_F_thm()
#         self.calc_F_rad()

#     def set_L_Qi(self, SFE, iPsi=0):
#         self.L, self.Qi = _mass_to_lum(self.Mstar.value, iPsi=iPsi)
#         self.L *= au.L_sun
#         self.Qi /= au.s
#         self.Psi = self.L/self.Mstar
#         self.Xi = self.Qi/self.Mstar

#     def get_nion_rms(self, Rst=None, fion=1.0):
#         if Rst is None:
#             Rst = self.R
#         self.nion_rms = hii.nion_rms(self.Qi.cgs.value, self.alphaB.cgs.value,
#                                      Rst.cgs.value, fion=fion)*au.cm**(-3)
#         return self.nion_rms

#     def get_Rst(self, nion_rms=None, fion=1.0):
#         if nion_rms is None:
#             self.get_nion_rms(self.R)
#         else:
#             self.nion_rms = nion_rms

#         self.Rst = (hii.rst(self.Qi.cgs.value, self.alphaB.cgs.value,
#                             nion_rms.cgs.value, fion=fion)*au.cm).to(au.pc)
#         # print '[get_Rst]:Qi,nion_rms,Rst ',self.Qi,self.nion_rms,self.Rst
#         return self.Rst

#     def Mdot_phot_blister(self):
#         Area = 2.0*np.pi*self.R**2
#         self.get_nion_rms()
#         rho = self.nion_rms*Cloud.muH
#         return rho*Area*self.cion

#     def Mphot_blister(self, SFE=None, time=None):
#         if SFE is not None:
#             self.set_SFE(SFE)
#         if time is None:
#             time = self.tff
#         return (self.Mdot_phot_blister()*time).to(au.Msun)

#     def Mphot_rocket_blister(self, SFE=None, time=None):
#         if SFE is not None:
#             self.set_SFE(SFE)
#         if time is None:
#             time = self.tff

#         Mphot = (self.Mdot_phot_blister()*time).to(au.Msun)
#         Mdyn = Mphot*self.cion/self.vesc

#         return Mphot+Mdyn

#     def calc_F_thm(self, fion=0.5):
#         T = (8.0*np.pi*ac.k_B*self.Tion) * \
#             np.sqrt(3.0*fion*self.Xi/(4.0*np.pi*self.alphaB))
#         self.Fthm = (T*self.M**0.75/(np.pi**0.25*self.Sigma**0.25)).cgs

#     def calc_F_rad(self):
#         self.Frad = (self.SFE*self.Psi*self.M/ac.c).cgs

if __name__ == '__main__':
    Ms_ = np.logspace(3, 7, num=100)*ac.M_sun
    Ss_ = np.logspace(1, 3, num=100)*ac.M_sun/ac.pc**2
    Ms, Ss = np.meshgrid(Ms_, Ss_)

    ## Input is array
    clouds = Cloud(M=Ms, Sigma=Ss)
    # Array of objects
    #clouds=np.array([[Cloud(M=M,Sigma=S) for S in Ss] for M in Ms])
    # print clouds.M,clouds.Sigma,clouds.tff

    levels = np.arange(-2, 2, 0.5)
    CS = plt.contour(clouds.M, clouds.Sigma, np.log10(clouds.tff.value),
                     levels=levels, linestyles='dashed', colors='r', alpha=0.2)
    plt.xscale('log')
    plt.yscale('log')
    manual_locations = [(-1, -1.4), (-0.62, -0.7), (-2, 0.5),
                        (1.7, 1.2), (2.0, 1.4), (2.4, 1.7)]
    plt.savefig('cloud.png')
