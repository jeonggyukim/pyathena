import numpy as np
from scipy.special import erf

# See Federrath & Klessen (2012)

def epsff_KM05(avir, Mach, b=0.5, phi=1, eps=1, beta=np.inf, phix=1.12, multiff=False):
    """
    Function to calculate eps_ff predicted by the Krumholz & McKee 2005 theory
    """

    sigma = np.sqrt(np.log(1.0 + b**2*Mach**2/(1.0 + 1.0/beta)))
    scrit = np.log(np.pi**2/5.0*phix**2*avir*Mach**2/(1.0+1.0/beta))

    if multiff:
        epsff = eps/(2.0*phi)*np.exp(3.0/8.0*sigma**2)* \
                (1.0 + erf((sigma**2 - scrit)/np.sqrt(2.0*sigma**2)))
    else:
        epsff = eps/(2.0*phi)*(1.0 + erf((sigma**2 - 2*scrit)/np.sqrt(8.0*sigma**2)))

    return epsff

def epsff_HC11(avir, Mach, b=0.5, phi=1, eps=1, beta=np.inf, ycut=0.1, multiff=True):
    """
    Function to calculate eps_ff predicted by the Hennebelle & Chabrier 2011 theory
    """

    sigma = np.sqrt(np.log(1.0 + b**2*Mach**2/(1.0 + 1.0/beta)))
    scrit_th = np.pi**2/5.0/ycut**2*avir/Mach**2*(1.0 + 1.0/beta)
    if multiff:
        scrit_turb = 0.0
    else:
        scrit_turb = np.pi**2/15.0/ycut*avir
    scrit = np.log(scrit_th + scrit_turb)

    return eps/(2.0*phi)*np.exp(3.0/8.0*sigma**2)*\
        (1.0 + erf((sigma**2 - scrit)/np.sqrt(2.0*sigma**2)))

def epsff_PN11(avir, Mach, b=0.5, phi=1, eps=1, beta=np.inf, theta=0.35, multiff=False):
    """
    Function to calculate eps_ff predicted by the Padoan & Nordlund 2011 theory
    """

    sigma = np.sqrt(np.log(1.0 + b**2*Mach**2/(1.0 + 1.0/beta)))
    fbeta = ((1.0 + 0.925*beta**-1.5)**(2.0/3.0))/(1.0 + 1/beta)**2
    scrit = np.log(0.067/theta**2*avir*Mach**2*fbeta)

    if multiff:
        epsff = eps/(2.0*phi)*np.exp(3.0/8.0*sigma**2)* \
                (1.0 + erf((sigma**2 - scrit)/np.sqrt(2.0*sigma**2)))
    else:
        epsff = eps/(2.0*phi)*np.exp(0.5*scrit)* \
                (1.0 + erf((sigma**2 - 2.0*scrit)/np.sqrt(8.0*sigma**2)))

    return epsff

def plt_epsff_theory(ax, avir, tratio, Mach0, Mach1, b=0.5,
                     beta=np.inf, KM=True, HC=True, PN=False):

    if KM:
        ax.fill_between(tratio,
                        epsff_KM05(avir, Mach0, beta=beta, b=b),
                        epsff_KM05(avir, Mach1, beta=beta, b=b),
                        color='gold', alpha=0.2)
        l1, = ax.plot(tratio, epsff_KM05(avir, Mach0, beta=beta, b=b),
                      color='gold', alpha=0.5, label=r'$10$')
        l2, = ax.plot(tratio, epsff_KM05(avir, Mach1, beta=beta, b=b),
                      color='gold', alpha=0.5, label=r'$\mathcal{M}=20$')

    if HC:
        ax.fill_between(tratio,
                        epsff_HC11(avir, Mach0, beta=beta, b=b),
                        epsff_HC11(avir, Mach1, beta=beta, b=b),
                        color='seagreen', alpha=0.2)
        l1, = ax.plot(tratio, epsff_HC11(avir, Mach0, beta=beta, b=b),
                      color='seagreen', alpha=0.5, label='10')
        l2, = ax.plot(tratio, epsff_HC11(avir, Mach1, beta=beta, b=b),
                      color='seagreen', alpha=0.5, label='20')

    if PN:
        ax.fill_between(tratio,
                        epsff_PN11(avir, Mach0, beta=beta, b=b),
                        epsff_PN11(avir, Mach1, beta=beta, b=b),
                        color='darkcyan', alpha=0.2)
        l1, = ax.plot(tratio, epsff_PN11(avir, Mach0, beta=beta, b=b),
                      color='darkcyan', alpha=0.5, label='10')
        l2, = ax.plot(tratio, epsff_PN11(avir, Mach1, beta=beta, b=b),
                      color='darkcyan', alpha=0.5, label='20')

    return l1, l2
