import numpy as np

def calc_xH2eq(nH, xi_H=2.0e-16, k_gr=3.0e-17, zeta_LW=0.0):
    """Calculate equilibrium H2 fraction xH2_eq (Gong et al. 2018)

    Parameters
    ----------
    nH : floats
        number density of hydrogen [cm^-3]
    xi_H : floats
        primary cosmic ray ionization rate [H^-1 s^-1]
    k_gr : floats
        H2 formation rate on grain surface [cm^3 s^-1]
    zeta_LW : floats
        photodissociation rate [s^-1]
        For Draine's ISRF in the solar neighborhood: 5.7e-11 s^-1

    Returns
    -------
    Equilibrium H2 fraction
    """

    a = 2.31*xi_H
    b = -2.0*k_gr*nH - 4.95*xi_H - zeta_LW
    c = nH*k_gr
    #print(a[0],b[0],c[0],np.sqrt(b[0]**2-4.0*a[0]*c[0]))
    #return (-b - np.sqrt(b**2 - 4.0*a*c))/(2.0*a)
    one_minus_sqrt = np.where((4.0*a*c)/b**2 < 1e-6,
                              (2.0*a*c)/b**2,
                              1.0 - np.sqrt(1 - (4.0*a*c)/b**2))

    return -b/(2.0*a)*one_minus_sqrt
