from __future__ import print_function

import numpy as np

def rebin_xyz(arr, bin_factor, fill_value=None):
    """
    Function to rebin masked 3d array.

    Parameters
    ----------
    arr : ndarray
        Masked or unmasked 3d numpy array. Shape is assumed to be (nz, ny, nx).
    bin_factor : int
        binning factor
    fill_value: float
        If arr is a masked array, fill masked elements with fill_value.
        If *None*, masked elements will be neglected in calculating average.
        Default value is *None*.

    Return
    ------
    arr_rebin: ndarray
        Smaller size, (averaged) 3d array. Shape is assumed to be
        (nz//bin_factor, ny//bin_factor, nx//bin_factor)
    """

    if bin_factor == 1:
        return arr

    # number of cells in the z-direction and xy-direction
    nz0 = arr.shape[0]
    ny0 = arr.shape[1]
    nx0 = arr.shape[2]

    # size of binned array
    nz1 = nz0 // bin_factor
    ny1 = ny0 // bin_factor
    nx1 = nx0 // bin_factor

    if np.ma.is_masked(arr) and fill_value is not None:
        np.ma.set_fill_value(arr, fill_value)
        arr = arr.filled()

    # See
    # https://stackoverflow.com/questions/4624112/grouping-2d-numpy-array-in-average/4624923#4624923
    return arr.reshape([nz1, nz0//nz1, ny1, ny0//ny1, nx1, nx0//nx1]).mean(axis=-1).mean(axis=3).mean(axis=1)


def rebin_xy(arr, bin_factor, fill_value=None):
    """
    Function to rebin masked 3d array in the x-y dimension.

    Parameters
    ----------
    arr : ndarray
        Masked or unmasked 3d numpy array. Shape is assumed to be (nz, ny, nx).
    bin_factor : int
        binning factor
    fill_value: float
        If arr is a masked array, fill masked elements with fill_value.
        If *None*, masked elements will be neglected in calculating average.
        Default value is *None*.

    Return
    ------
    arr_rebin: ndarray
        Smaller size, (averaged) 3d array. Shape is assumed to be
        (nz, ny//bin_factor, nx//bin_factor)
    """

    if bin_factor == 1:
        return arr

    # number of cells in the z-direction and xy-direction
    nz = arr.shape[0]
    ny0 = arr.shape[1]
    nx0 = arr.shape[2]

    # size of binned array
    ny1 = ny0 // bin_factor
    nx1 = nx0 // bin_factor

    if np.ma.is_masked(arr) and fill_value is not None:
        np.ma.set_fill_value(arr, fill_value)
        arr = arr.filled()

    # See
    # https://stackoverflow.com/questions/4624112/grouping-2d-numpy-array-in-average/4624923#4624923
    return arr.reshape([nz, ny1, ny0//ny1, nx1, nx0//nx1]).mean(axis=-1).mean(axis=2)


if __name__ == '__main__':

    # Test of rebin_xy
    mask = True
    # Define test data
    big = np.ma.array([[5, 5, 1, 2],
                       [5, 5, 2, 1],
                       [2, 1, 1, 1],
                       [2, 1, 1, 1]])
    if mask:
        big.mask = [[1, 1, 0, 0],
                    [0, 1, 1, 1],
                    [1, 0, 1, 0],
                    [1, 1, 1, 0]]

    big = np.tile(big, (1, 1, 1))

    small1 = rebin_xy_masked(big, 2, fill_value=0.0)
    small2 = rebin_xy_masked(big, 2, fill_value=None)

    print('Original array\n', big)
    print('With fill value 0.0\n', small1)
    print('Without fill value\n', small2)
