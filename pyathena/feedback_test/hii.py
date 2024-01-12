from ..util.units import Units
from ..microphysics.cool import coeff_alpha_rr_H
import numpy as np
import astropy.constants as ac
import astropy.units as au

def get_fion_P72(Qn, T, sigmad):
    """Function to compute the fraction of ionizing photons absorbed by dust (Petrosian 1972)
    """

    alpha = coeff_alpha_rr_H(T)
    taud = (3.0/(4.0*np.pi*alpha))**(1.0/3.0)*Qn**(1.0/3.0)*sigmad
    taud = np.atleast_1d(taud)
    from scipy.optimize import brentq
    def func_P72(x, a):
        return np.exp(-a*x) + 6.0/a**3.0*(np.exp(-a*x) - 1.0 + a*x - 0.5*a**2*x**2)

    # See Eq 8 in P72
    y0 = np.zeros_like(taud)
    taud0 = np.zeros_like(taud)
    for i, taud_ in enumerate(taud):
        if taud_ == 0:
            y0[i] = 1.0
        else:
            y0[i] = brentq(func_P72, 0.0, 1.0, args=(taud_))
        taud0[i] = taud_*y0[i]

    fion_P72 = 1.0/3.0*taud0**3*np.exp(-taud0) / \
         (taud0**2 - 2.0*taud0 + 2.0*(1.0 - np.exp(-taud0)))

    # Alternative
    # fion_P72_alt = y0**3
    if len(fion_P72) == 1:
        fion_P72 = fion_P72[0]

    return fion_P72

def get_rIF0(Q, n, T, sigmad, fion=None):
    """Function to compute initial Stromgren radius for a dusty HII region.
    If fion is not provided, use P72 formula.
    """

    Qn = Q*n
    if fion is None:
        fion = get_fion_P72(Qn, T, sigmad)

    print('fion','fion^1/3', fion, fion**(1.0/3.0))

    return ((3.0*fion*Q)/(4.0*np.pi*coeff_alpha_rr_H(T)*n**2))**(1.0/3.0)

def get_rIF_D(t, rIF0, T, solution='Hosokawa'):
    """Function to compute Hosokawa & Inutsuka or Spitzer solution

    time in Myr
    rIF0 in pc
    T in Kel
    """

    u = Units()
    cs_ion = (np.sqrt(2.1*ac.k_B*T*au.K/(u.muH*1.008*ac.u))).to('km/s').value
    # print(cs_ion)
    if solution == 'Hosokawa':
        return rIF0*(1.0 + 7.0/(2.0*3.0**0.5)*cs_ion*t*u.Myr/rIF0)**(4.0/7.0) # Hosokawa
    elif solution == 'Spitzer':
        return rIF0*(1.0 + 7.0/4.0*cs_ion*t*u.Myr/rIF0)**(4.0/7.0) # Spitzer
    else:
        raise ValueError('Unrecognized solution {0:s}'.format(solution))

def get_Mion(t, rIF0, T, solution='Hosokawa'):
    u = Units()
    cs_ion = (np.sqrt(2.1*ac.k_B*T*au.K/(u.muH*1.008*ac.u))).to('km/s').value
    if solution == 'Hosokawa':
        rIF = rIF0*(1.0 + 7.0/(2.0*3.0**0.5)*cs_ion*t*u.Myr/rIF0)**(4.0/7.0) # Hosokawa
    elif solution == 'Spitzer':
        rIF = rIF0*(1.0 + 7.0/4.0*cs_ion*t*u.Myr/rIF0)**(4.0/7.0) # Spitzer
    else:
        raise ValueError('Unrecognized solution {0:s}'.format(solution))

def get_Rsh(s, num):
    """Function to calculate shell radius using density peak
    """

    r = s.get_profile1d(num, fields_y=['nH'], field_x='r', bins=200, statistic=['mean','median'],
                        force_override=False)
    # print(r.keys())
    from scipy.interpolate import interp1d
    x = r['r']['binc']
    y1 = r['nH']['mean']
    y2 = r['nH']['mean']
    f1 = interp1d(x, y1, kind='quadratic') #, bounds_error=False, fill_value='extrapolate')
    f2 = interp1d(x, y2, kind='quadratic')
    x = np.linspace(x.min(), x.max(), 1000)
    idx1 = np.where(f1(x) == f1(x).max())[0][0]
    idx2 = np.where(f2(x) == f2(x).max())[0][0]
    Rsh1 = x[idx1]
    Rsh2 = x[idx2]
    time = r['time'] # time in Myr

    return Rsh1, Rsh2, time

def get_Rsh_all(s):
    """Function to calculate Rsh of all snapshots
    """
    res = dict()
    res['Rsh1'] = []
    res['Rsh2'] = []
    res['time'] = []
    nums = s.nums
    for num in nums[1::1]:
        print(num, end=' ')
        Rsh1, Rsh2, time = get_Rsh(s, num)
        res['Rsh1'].append(Rsh1)
        res['Rsh2'].append(Rsh2)
        res['time'].append(time)

    return res
