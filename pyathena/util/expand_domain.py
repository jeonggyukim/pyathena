import xarray as xr
import numpy as np
from scipy.ndimage import shift


def expand_x(sim, data):
    """function to expand original data from -Lx/2 < x < Lx/2 to -3*Lx/2 < x < 3*Lx/2

    Parameters
    ----------
    sim : pyathena.LoadSim class (must called load_vtk to set `ds`)

    data : xarray.Dataset in the original domain returned by ds.get_field()

    Returns
    -------
    xarray.Dataset with extended x-domain (tripled)
    """
    # retrive the AthenaDataset class
    ds = sim.ds

    # is this a shearing-box simulation?
    try:
        shear = sim.par["configure"]["ShearingBox"] == "yes"
    except KeyError:
        shear = False

    # get domain information
    Lx, Ly, Lz = ds.domain["Lx"]
    dx, dy, dz = ds.domain["dx"]

    # exapnd in x assuming periodic BC
    data_left = data.copy(deep=True).assign_coords(x=data.coords["x"] - Lx)
    data_right = data.copy(deep=True).assign_coords(x=data.coords["x"] + Lx)

    # exapnd in x assuming shear-periodic BC
    if shear:
        # get shear related parameters
        qOmL = sim.par["problem"]["qshear"] * sim.par["problem"]["Omega"] * Lx
        time = ds.domain["time"]
        qOmLt = qOmL * time
        dims = data.to_array("variable").dims
        ndims = len(dims) - 1
        try:
            yidx = dims.index("y") - 1
        except ValueError:
            raise (
                "input data must have both x and y in dims to apply shear-periodic BCs"
            )
        # apply y-shift due to shear for each variable
        for var in data:
            # shift for the left
            shifts = np.zeros(ndims)
            shifts[yidx] = qOmLt / dy
            shifted_L = shift(data_left[var].data, shifts, mode="grid-wrap", order=1)
            # shift for the right
            shifts = np.zeros(ndims)
            shifts[yidx] = -qOmLt / dy
            shifted_R = shift(data_right[var].data, shifts, mode="grid-wrap", order=1)
            # add shear velocity
            if var == "vy":
                shifted_L += qOmL
                shifted_R -= qOmL
            # update L/R data
            data_left[var] = (dims[1:], shifted_L)
            data_right[var] = (dims[1:], shifted_R)
    return xr.concat([data_left, data, data_right], dim="x")


def expand_y(sim, data):
    """function to expand original data from -Ly/2 < y < Ly/2 to -3*Ly/2 < y < 3*Ly/2

    Parameters
    ----------
    sim : pyathena.LoadSim class (must called load_vtk to set `ds`)

    data : xarray.Dataset in the original domain returned by ds.get_field()

    Returns
    -------
    xarray.Dataset with extended y-domain (tripled)
    """
    # retrive the AthenaDataset class
    ds = sim.ds

    # get domain information
    Lx, Ly, Lz = ds.domain["Lx"]

    # exapnd in x assuming periodic BC
    data_bot = data.copy(deep=True).assign_coords(y=data.coords["y"] - Ly)
    data_top = data.copy(deep=True).assign_coords(y=data.coords["y"] + Ly)

    return xr.concat([data_bot, data, data_top], dim="y")


def expand_xy(s, data):
    """Triple XY domain

    Parameters
    ----------
    sim : pyathena.LoadSim class (must called load_vtk to set `ds`)

    data : xarray.Dataset in the original domain returned by ds.get_field()

    Returns
    -------
    xarray.Dataset with extended xy-domain (tripled)

    Example
    -------
    >>> from pyathena.tigress_ncr.load_sim_tigress_ncr import LoadSimTIGRESSNCR
    >>> s = LoadSimTIGRESSNCR("/tigress/changgoo/TIGRESS-NCR/R8_8pc_NCR.full.xy2048.eps0.0",verbose=True)
    >>> ds = s.load_vtk(290)
    >>> data = ds.get_field(["nH","T","vy"])
    >>> data_exp = expand_xy(s,data.sel(z=0,method="nearest")) # expanded 2D slices
    >>> data_exp["vy"].plot(**s.dfi["vy"]["imshow_args"])
    """
    return expand_y(s, expand_x(s, data))
