from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel
import numpy as np

def deriv_kernel(axis=0, dim=1, fft=False, sobel=True, gauss=False, stddev=3.0):
    if sobel:
        smooth = np.array([1, 2, 1])
    else:
        smooth = np.array([1, 1, 1])
    if fft:
        deriv = np.array([1, 0, -1])
    else:
        deriv = np.array([-1, 0, 1])

    if dim == 1:
        if gauss:
            return gaussian_deriv_kernel(axis=axis, stddev=stddev, oned=True)
        else:
            return deriv
    elif dim == 2:
        if gauss:
            return gaussian_deriv_kernel(axis=axis, stddev=stddev)
        if axis == 0:
            return np.einsum('i,j', deriv, smooth)
        elif axis == 1:
            return np.einsum('j,i', deriv, smooth)
    elif dim == 3:
        if axis == 0:
            return np.einsum('i,j,k', deriv, smooth, smooth)
        elif axis == 1:
            return np.einsum('j,k,i', deriv, smooth, smooth)
        elif axis == 2:
            return np.einsum('k,i,j', deriv, smooth, smooth)


def gaussian_deriv_kernel(axis=0, stddev=3.0, oned=False):
    if oned:
        gauss = Gaussian1DKernel(stddev)
    else:
        gauss = Gaussian2DKernel(stddev)
    x = np.linspace(0, gauss.shape[axis]-1, gauss.shape[axis])
    dkernel = deriv_central(gauss.array, x, axis=axis)
    if not oned:
        if axis == 0:
            dkernel = dkernel[:, 1:-1]
        elif axis == 1:
            dkernel = dkernel[1:-1, :]
        return dkernel/abs(dkernel).sum()
    else:
        return dkernel


def deriv_direct(yarr, xarr, axis=0):
    dyarr = np.diff(yarr, axis=axis)
    dxarr = np.diff(xarr)
    if yarr.ndim == 1:
        return dyarr/dxarr
    if yarr.ndim == 2:
        if axis == 0:
            dxarr = dxarr[:, np.newaxis]
        if axis == 1:
            dxarr = dxarr[np.newaxis, :]
        return dyarr/dxarr
    if yarr.ndim == 3:
        if axis == 0:
            dxarr = dxarr[:, np.newaxis, np.newaxis]
        if axis == 1:
            dxarr = dxarr[np.newaxis, :, np.newaxis]
        if axis == 2:
            dxarr = dxarr[np.newaxis, np.newaxis, :]
        return dyarr/dxarr


def deriv_central(yarr, xarr, axis=0):
    dx = xarr[2:]-xarr[:-2]
    if yarr.ndim == 1:
        dy = yarr[2:]-yarr[:-2]
        return dy/dx
    elif yarr.ndim == 2:
        yswap = yarr.swapaxes(axis, -1)
        dy = yswap[:, 2:]-yswap[:, :-2]
        dx = dx[np.newaxis, :]
    elif yarr.ndim == 3:
        yswap = yarr.swapaxes(axis, -1)
        dy = yswap[:, :, 2:]-yswap[:, :, :-2]
        dx = dx[np.newaxis, np.newaxis, :]

    dydx = dy/dx
    return dydx.swapaxes(axis, -1)


def deriv_convolve(yarr, xarr, axis=0, fft=False, gauss=True, stddev=3.0):
    kernel = deriv_kernel(axis=axis, dim=yarr.ndim,
                          fft=fft, gauss=gauss, stddev=stddev)
    norm = abs(kernel).sum()

    # print norm,kernel.shape
    if fft:
        dy = convolve_fft(yarr, kernel/float(norm), boundary='wrap')
    else:
        dy = convolve(yarr, kernel/float(norm), normalize_kernel=False,
                      boundary='extend')
    dx = xarr[1]-xarr[0]
    #print (dy/dx).max()
    return dy/dx


def gradient(scal, x, y, z, deriv=deriv_convolve):

    dsdx = deriv(scal, x, axis=2)
    dsdy = deriv(scal, y, axis=1)
    dsdz = deriv(scal, z, axis=0)

    return dsdx, dsdy, dsdz


def divergence(vx, vy, vz, x, y, z, deriv=deriv_convolve):

    dvxdx = deriv(vx, x, axis=2)
    dvydy = deriv(vy, y, axis=1)
    dvzdz = deriv(vz, z, axis=0)

    return dvxdx, dvydy, dvzdz


def curl(vx, vy, vz, x, y, z, deriv=deriv_convolve):

    dvxdy = deriv(vx, y, axis=1)
    dvxdz = deriv(vx, z, axis=0)
    dvydx = deriv(vy, x, axis=2)
    dvydz = deriv(vy, z, axis=0)
    dvzdx = deriv(vz, x, axis=2)
    dvzdy = deriv(vz, y, axis=1)
    xcomp = dvzdy - dvydz
    ycomp = dvxdz - dvzdx
    zcomp = dvydx - dvxdy

    return xcomp, ycomp, zcomp


def helicity(vx, vy, vz, x, y, z):

    curlx, curly, curlz = curl(vx, vy, vz, x, y, z, deriv=deriv_convolve)
    helicity = vx*curlx + vy*curly + vz*curlz
    print((helicity.mean()))

    return helicity
