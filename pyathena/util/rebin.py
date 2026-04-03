
import numpy as np

def rebin_xyz(arr, bin_factor, fill_value=None):
    """Rebin a 3D array by averaging over blocks of size ``bin_factor`` in all dimensions.

    Parameters
    ----------
    arr : ndarray
        Masked or unmasked 3D numpy array with shape ``(nz, ny, nx)``.
    bin_factor : int
        Number of cells to bin in each dimension. Must evenly divide each axis.
    fill_value : float, optional
        If ``arr`` is a masked array, replace masked elements with this value
        before averaging. If ``None``, masked elements are excluded from the
        average. Default is ``None``.

    Returns
    -------
    arr_rebin : ndarray
        Rebinned array with shape ``(nz//bin_factor, ny//bin_factor, nx//bin_factor)``.
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
    """Rebin a 3D array by averaging over blocks of size ``bin_factor`` in the x-y plane.

    The z-axis is left unchanged.

    Parameters
    ----------
    arr : ndarray
        Masked or unmasked 3D numpy array with shape ``(nz, ny, nx)``.
    bin_factor : int
        Number of cells to bin along x and y. Must evenly divide both axes.
    fill_value : float, optional
        If ``arr`` is a masked array, replace masked elements with this value
        before averaging. If ``None``, masked elements are excluded from the
        average. Default is ``None``.

    Returns
    -------
    arr_rebin : ndarray
        Rebinned array with shape ``(nz, ny//bin_factor, nx//bin_factor)``.
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
