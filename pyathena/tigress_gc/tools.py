from pathlib import Path
from pygc.util import add_derived_fields
from pygc.pot import MHubble, Plummer
import numpy as np
import xarray as xr
from pyathena.tigress_gc import config
from pyathena.util import transform

def calculate_azimuthal_averages(s, num, warmcold=False):
    """
    Calculate azimuthal averages and write to file

    Parameters
    ----------
    s : pa.LoadSim object
    num : integer index of simulation output
    warmcold : if True, calculate quantities only for T < 2e4 gas
    
    Description
    -----------
    Calculates following Fields (function of (z,R))
    density            : <rho>
    pressure           : <P>
    turbulent_pressure : <rho v_z^2>
    velocity1          : <v_R>
    velocity2          : <v_phi>
    velocity3          : <v_z>
    helicity           : <v*w>
    mass_flux1         : <rho v_R>
    mass_flux2         : <rho v_phi>
    mass_flux3         : <rho v_z>
    B1                 : <B_R>
    B2                 : <B_phi>
    B3                 : <B_z>
    B_squared1         : <B_R^2>
    B_squared2         : <B_phi^2>
    B_squared3         : <B_z^2>
    Reynolds           : <rho v_R (v_phi - v_circ)>
    Maxwell            : <B_R B_phi>
    is_wc              : <Theta>
    """


    # Step 1. Load data
    ds = s.load_vtk(num, id0=False)
    field_list = [f for f in ds.field_list if 'scalar' not in f]
    dat = ds.get_field(field_list)
    dat = add_derived_fields(dat, ['R', 'gz_sg'])
    dx, dy, dz = s.domain['dx']
    if warmcold:
        dat = add_derived_fields(dat, 'T')
        # Set switch for warm-cold medium (Theta = 1 for T<2e4; 0 for T>2e4)
        is_wc = xr.where(dat.T < config.Twarm, 1, 0)
        # Apply switch
        dat = dat.where(dat.T < config.Twarm, other=0)
        # Save switch
        dat['is_wc'] = is_wc
        dat = dat.drop('T')

    # helicity
    vx, vy, vz = dat.velocity1, dat.velocity2, dat.velocity3
    wx = vz.differentiate('y') - vy.differentiate('z')
    wy = vx.differentiate('z') - vz.differentiate('x')
    wz = vy.differentiate('x') - vx.differentiate('y')
    dat['helicity'] = vx*wx + vy*wy + vz*wz

    # midplane thermal pressure
    dat['pressure'] = dat.pressure.sel(z=0, method='nearest')

    # midplane turbulent pressure
    rho = dat.density.sel(z=0, method='nearest')
    vz = dat.velocity3.sel(z=0, method='nearest')
    dat['turbulent_pressure'] = rho*vz**2

    # midplane magnetic stress
    if 'cell_centered_B1' in dat:
        bx = dat.cell_centered_B1.sel(z=0, method='nearest')
        by = dat.cell_centered_B2.sel(z=0, method='nearest')
        bz = dat.cell_centered_B3.sel(z=0, method='nearest')
        dat['magnetic_stress'] = 0.5*(bx**2 + by**2 - bz**2)

    # transform vector quantities to cylindrical coordinates
    vec = (dat['velocity1'], dat['velocity2'], dat['velocity3'])
    R, vec_cyl = transform.to_cylindrical(vec, (0, 0, 0))
    for i, axis in enumerate([1,2,3]):
        dat[f'velocity{axis}'] = vec_cyl[i]
    
    if 'cell_centered_B1' in dat:
        vec = (dat['cell_centered_B1'], dat['cell_centered_B2'], dat['cell_centered_B3'])
        R, vec_cyl = transform.to_cylindrical(vec, (0, 0, 0))
        for i, axis in enumerate([1,2,3]):
            dat[f'cell_centered_B{axis}'] = vec_cyl[i]
        dat = dat.rename({'cell_centered_B1':'B1',
                          'cell_centered_B2':'B2',
                          'cell_centered_B3':'B3'})

    # calculate circular velocity
    vcirc = get_circular_velocity(s, dat.x, dat.y)
    
    # Calculate derived quantities
    dat['Reynolds'] = dat.density*dat.velocity1*(dat.velocity2 - vcirc)
    dat['mass_flux1'] = dat.density*dat.velocity1
    dat['mass_flux2'] = dat.density*dat.velocity2
    dat['mass_flux3'] = dat.density*dat.velocity3
    if 'B1' in dat:
        dat['Maxwell'] = -(dat.B1*dat.B2)
        dat['B_squared1'] = dat.B1**2
        dat['B_squared2'] = dat.B2**2
        dat['B_squared3'] = dat.B3**2

    # weights
    bul = MHubble(rb=s.par['problem']['R_b'], rhob=s.par['problem']['rho_b'])
    bh = Plummer(Mc=s.par['problem']['M_c'], Rc=s.par['problem']['R_c'])
    dat['gz_ext'] = bul.gz(dat.x, dat.y, dat.z) + bh.gz(dat.x, dat.y, dat.z)
    dat['gz_ext'] = dat.gz_ext.transpose('z','y','x')
    dat['Wself'] = -(dat.density*dat.gz_sg*dz).sel(z=slice(0, s.domain['re'][2])).sum(dim='z')
    dat['Wext'] = -(dat.density*dat.gz_ext*dz).sel(z=slice(0, s.domain['re'][2])).sum(dim='z')
    
    # Radial binning
    nbin = 64
    edges = np.linspace(0, 1000, nbin+1)
    dat = dat.groupby_bins('R', edges).mean()
    dat.attrs = {'time':ds.domain['time']}
    dat = dat.rename({'R_bins':'R'})
    # Calculate midpoint of each bin and use it as coordinate
    dat.update({'R':list(map(lambda x: x.mid, dat.R.data))})
    dat = dat.transpose('z','R')

    if warmcold:
        area = (is_wc.sel(z=0, method='nearest')*dx*dy).groupby_bins('R', edges).sum()
        area = area.rename({'R_bins':'R'})
        area.coords['R'] = list(map(lambda x: x.mid, area.R.data))
        dat['area_warmcold'] = area

    return dat

def get_circular_velocity(s, x, y=0, z=0):
    bul = MHubble(rb=s.par['problem']['R_b'], rhob=s.par['problem']['rho_b'])
    bh = Plummer(Mc=s.par['problem']['M_c'], Rc=s.par['problem']['R_c'])
    if 'Omega_p' in s.par['problem']:
        Omega_p = s.par['problem']['Omega_p']
    elif 'Omega_0' in s.par['problem']:
        Omega_p = s.par['problem']['Omega_0']
    else:
        raise ValueError("No pattern speed information in the input file")
    vbul = bul.vcirc(x, y, z)
    vbh = bh.vcirc(x, y, z)
    R = np.sqrt(x**2 + y**2)
    vcirc = np.sqrt(vbul**2 + vbh**2) - R*Omega_p
    return vcirc
