import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize,SymLogNorm
from .shiftedColorMap import *
import astropy.units as au
import astropy.constants as ac

def set_aux(model='solar',verbose=False):
    """
    Set auxiliary information about fields
    """
    
    aux = {}

    # Somewhat confusing, but let's take following convention
    # Density = nH [cm^-3]
    aux['density']=dict(label=r'$n_{\rm H}\;[{\rm cm}^{-3}]$', \
        unit='cm**(-3)', limits=(1.e-6,1.e6), \
        cmap=plt.cm.Spectral_r,clim=(2.e-5,2.e2), \
        cticks=(1.e-4,1.e-2,1,1.e2), \
        n_bins=128, norm=LogNorm())
    
    aux['nH']=dict(label=r'$n_H\;[{\rm cm}^{-3}]$', \
        unit='cm**(-3)', limits=(1.e-6,1.e6), \
        cmap=plt.cm.Spectral_r,clim=(2.e-5,2.e2), \
        cticks=(1.e-4,1.e-2,1,1.e2), \
        n_bins=128, norm=LogNorm())

    # Legacy from the previous pyathena convention
    aux['surface_density']=dict( \
        label=r'$\Sigma\;[{\rm M}_{\odot} {\rm pc}^{-2}]$', \
        cmap=plt.cm.pink_r,clim=(0.1,100),norm=LogNorm())
    
    # Gas density rho = nH*muH [g/cm^3]
    aux['rho']=dict(label=r'$\rho\;[{\rm g}\,{\rm cm}^{-3}]$', \
        unit='g*cm**(-3)', limits=(1.e-30,1.e18), \
        cmap=plt.cm.Spectral_r,clim=(2.0e-29,2.0e-22), \
        cticks=(1.e-28,1.e-26,1,1.e-24,1e-22), \
        proj_mul=1.0/((1.0*ac.M_sun/au.pc**2).to('g/cm**2')).value, \
        n_bins=128, norm=LogNorm())

    # Total gas surface density [Msun/pc^2]
    aux['rho_proj']=dict(
        label=r'$\Sigma\;[M_{\odot} {\rm pc}^{-2}]$', \
        cmap=plt.cm.pink_r,clim=(0.1,300),
        cticks=(1e0,1e1,1e2), \
        norm=LogNorm())

    # Pthm/kB [K*cm^-3]
    aux['pok']=dict(label=r'$P/k_B\;[{\rm K}\,{\rm cm}^{-3}]$', \
        unit='K*cm**(-3)', limits=(1.e-2,1.e8), \
        cmap=plt.cm.gnuplot2,clim=(10,5.e5), \
        cticks=(1.e2,1.e3,1.e4,1.e5), \
        n_bins=128, norm=LogNorm())

    # Temprature [K]
    aux['temperature']=dict(label=r'$T\;[{\rm K}]$', \
        unit='K', limits=(1.e0,1.e9), \
        cmap=shiftedColorMap(plt.cm.RdYlBu_r,midpoint=3/7.), \
        clim=(10,1.e7), \
        cticks=(1.e2,1.e4,1.e6), \
        n_bins=128, norm=LogNorm())

    # v_z [km/s]
    aux['velocity_z']=dict(label=r'$v_z\;[{\rm km/s}]$', \
        unit='km/s', limits=(-1500,1500), \
        cmap=plt.cm.RdBu_r,clim=(-200,200), \
        cticks=(-100,0,100), \
        n_bins=256, norm=Normalize())

    # |B| [muG]
    aux['magnetic_field_strength']=dict(label=r'$B\;[\mu{\rm G}]$', \
        unit='uG', \
        cmap=plt.cm.viridis,clim=(0.01,10),factor=1, \
        n_bins=128, norm=LogNorm())

    # Pmag/kB
    aux['mag_pok']=dict(label=r'$P_{\rm mag}/k_B\;[{\rm K}{\rm cm}^{-3}]$',\
        unit='K*cm**(-3)', limits=(1.e-2,1.e8), \
        cmap=plt.cm.gnuplot2, clim=(10,5.e5), \
        n_bins=128, norm=LogNorm())

    # Ram pressure in the z direction?
    aux['ram_pok_z']=dict(\
        label=r'$P_{\rm ram,z}/k_B\;[{\rm K}{\rm cm}^{-3}]$', \
        unit='K*cm**(-3)', limits=(1.e-2,1.e8), \
        cmap=plt.cm.gnuplot2, clim=(10,5.e5), \
        n_bins=128, norm=LogNorm())

    # Plasma beta
    aux['plasma_beta']=dict(
        label=r'$\beta$', limits=(1.e-4,1.e16), \
        n_bins=256, norm=LogNorm())

    # H neutral fraction
    aux['xn']=dict(label=r'$x_{\rm n}$', \
        unit='dimensionless', limits=(0,1), \
        cmap=plt.cm.YlGn,clim=(0,1), \
        n_bins=128, norm=Normalize())

    # Volume averaged xn
    aux['xn_proj']=dict(
        label=r'$\langle x_{\rm n}\rangle$', \
        unit='dimensionless', limits=(0,1), \
        cmap=plt.cm.YlGn,clim=(0,1), \
        cticks=(0, 0.5, 1), \
        weight_field='cell_volume', \
        n_bins=128, norm=Normalize())

    # ne = Electron number density [cm^-3]
    aux['ne']=dict(label=r'$n_{\rm e}\;[{\rm cm}^{-3}]$', \
        unit='cm**(-3)', limits=(1e-6,1e6), \
        cmap=plt.cm.plasma,clim=(2e-5,2e2), \
        cticks=(1e-4,1e-2,1e0,1e2), \
        n_bins=128, norm=LogNorm())

    # ne^2 [cm^-6]
    aux['nesq']=dict(label=r'$n_{\rm e}^2\;[{\rm cm}^{-6}]$', \
        unit='cm**(-6)', limits=(1e-12,1e5), \
        cmap=plt.cm.plasma,clim=(1e-12,1e4), \
        cticks=(1.e-8,1e-4,1.e0,1.e4), \
        proj_mul=1.0/((1.0*au.cm**(-6)*au.pc).to('cm**(-6)*pc')).cgs.value, \
        n_bins=128, norm=LogNorm())

    # int ne^2 dl = Emission Measure [cm^-6 pc]
    aux['nesq_proj']=dict(label=r'${\rm EM} \;[{\rm cm}^{-6}\,{\rm pc}]$', \
        unit='pc*cm**(-6)', limits=(1e-4,1e6), \
        cmap=plt.cm.plasma,clim=(1e-2,1e5), \
        cticks=(1e-1,1.e1,1e3,1e5), \
        n_bins=128, norm=LogNorm())

    # Radiation energy density0
    aux['Erad0']=dict(label=r'$\mathcal{E}_{\rm i}\,[{\rm erg\,cm}^{-3}]$', \
        unit='erg*cm**(-3)*s**(-1)', limits=(1e-8,1e-18), \
        cmap=plt.cm.viridis,clim=(1e-11,1e-16), \
        cticks=(1e-15,1e-13,1e-11), \
        n_bins=128, norm=LogNorm())
    
    # Radiation energy density1
    aux['Erad1']=dict(label=r'$\mathcal{E}_{\rm n}\,[{\rm erg\,cm}^{-3}]$', \
        unit='erg*cm**(-3)*s**(-1)', limits=(1e-8,1e-18), \
        cmap=plt.cm.viridis,clim=(1e-10,1e-14), \
        cticks=(1e-13,1e-11), \
        n_bins=128, norm=LogNorm())

    aux['star_particles']=dict(label=r'${\rm age}\,[{\rm Myr}]}$', \
        unit='Myr', limits=(0,40), \
        cmap=plt.cm.cool_r,clim=(0,40), \
        cticks=(0,20,40), \
        n_bins=256, norm=LogNorm())

    # Adjust limits
    if model.startswith('R4'):
        if verbose: print('auxilary information is set for R4')
        aux['nH']['clim']=(2.e-4,2.e3)
        aux['nH']['cticks']=(1.e-2,1,1.e2)
        aux['pok']['clim']=(10,5.e6)
        aux['pok']['cticks']=(1.e2,1.e4,1.e6)
        aux['surface_density']['clim']=(1,1000)
        aux['velocity_z']['clim']=(-300,300)
        aux['velocity_z']['cticks']=(-200,0,200)
        aux['magnetic_field_strength']['clim']=(0.01,100)

    elif model.startswith('R2'):
        if verbose: print('auxilary information is set for R2')
        aux['nH']['clim']=(2.e-4,2.e3)
        aux['nH']['cticks']=(1.e-2,1,1.e2)
        aux['pok']['clim']=(10,5.e6)
        aux['pok']['cticks']=(1.e2,1.e4,1.e6)
        aux['velocity_z']['clim']=(-300,300)
        aux['velocity_z']['cticks']=(-200,0,200)
        aux['surface_density']['clim']=(1,1000)
        aux['magnetic_field_strength']['clim']=(0.01,100)

    elif model is 'multi_SN':
        aux['nH']['clim']=(2.e-5,2.e2)
        aux['pok']['clim']=(50,1.e5)
    else:
        if verbose: print('auxilary information is set for Solar nbhd.')

    return aux
