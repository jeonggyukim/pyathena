#!/usr/bin/env python

# import os
# import os.path as osp
import gc
import time
from mpi4py import MPI

# import matplotlib.pyplot as plt
import pprint
import argparse
import sys

# import pickle
import numpy as np
import xarray as xr

import pyathena as pa
from pyathena.util.split_container import split_container
# from pyathena.plt_tools.make_movie import make_movie

from pathlib import Path


def grid_msp(s, num, agemin, agemax):
    """read starpar_vtk and remap starpar mass onto a grid"""
    # domain information
    le1, le2 = s.domain["le"][0], s.domain["le"][1]
    re1, re2 = s.domain["re"][0], s.domain["re"][1]
    # dx1, dx2 = s.domain['dx'][0], s.domain['dx'][1]
    Nx1, Nx2 = s.domain["Nx"][0], s.domain["Nx"][1]
    # edge coordinates
    xfc = np.linspace(le1, re1, Nx1 + 1)
    yfc = np.linspace(le2, re2, Nx1 + 1)
    # center coordinates
    xcc = 0.5 * (xfc[1:] + xfc[:-1])
    ycc = 0.5 * (yfc[1:] + yfc[:-1])
    # load starpar vtk
    sp = s.load_starpar_vtk(num, force_override=True)
    if len(sp.columns) > 0:
        # apply age cut
        if "mage" in sp:
            sp = sp[(sp.mage * s.u.Myr < agemax) & (sp.mage * s.u.Myr >= agemin)]
        elif "age" in sp:
            sp = sp[(sp.age * s.u.Myr < agemax) & (sp.age * s.u.Myr >= agemin)]
        # remap the starpar onto a grid using histogram2d
        msp, _, _ = np.histogram2d(sp.x2, sp.x1, bins=[yfc, xfc], weights=sp.mass)
    else:
        msp = np.zeros((Nx2, Nx1))
    msp = xr.DataArray(msp, dims=["y", "x"], coords=[ycc, xcc])
    return msp


def add_weights(s, dat):
    dz = s.domain["dx"][2]
    z = dat["z"].data

    if "weight_ext" not in dat:
        from pyathena.util.tigress_ext_pot import TigressExtPot

        phiext = TigressExtPot(s.par["problem"]).phiext
        gext = (
            -(phiext((z + dz / 2)) - phiext((z - dz / 2))).to("km**2/s**2").value / dz
        )
        dWext = dat["density"] * (gext * dat["z"] / dat["z"])
        dat["weight_ext"] = dWext
    if "weight_self" not in dat:
        phir = dat.gravitational_potential.shift(z=-1, fill_value=np.nan)
        phil = dat.gravitational_potential.shift(z=1, fill_value=np.nan)
        phir.loc[{"z": phir.z[-1]}] = (
            3 * phir.isel(z=-2) - 3 * phir.isel(z=-3) + phir.isel(z=-4)
        )
        phil.loc[{"z": phir.z[0]}] = (
            3 * phil.isel(z=1) - 3 * phil.isel(z=2) + phil.isel(z=3)
        )
        gz = (phil - phir) / (2 * dz)
        dWsg = dat["density"] * gz
        dat["weight_self"] = dWsg

    return dat


def get_weights(s, dat, direction="upper"):
    dz = s.domain["dx"][2]
    dat2d = xr.Dataset()
    if direction == "upper":
        dat2d["wself"] = (
            -(dat["weight_self"] * dz).sel(z=slice(0, s.domain["re"][2])).sum(dim="z")
        )
        dat2d["wext"] = (
            -(dat["weight_ext"] * dz).sel(z=slice(0, s.domain["re"][2])).sum(dim="z")
        )
    elif direction == "lower":
        dat2d["wself"] = (
            (dat["weight_self"] * dz).sel(z=slice(s.domain["le"][2], 0)).sum(dim="z")
        )
        dat2d["wext"] = (
            (dat["weight_ext"] * dz).sel(z=slice(s.domain["le"][2], 0)).sum(dim="z")
        )
    dat2d["wtot"] = dat2d["wself"] + dat2d["wext"]
    return dat2d * s.u.pok


def get_pressures(s, dat, dz=None):
    if dz is None:
        dz = s.domain["dx"][2]
    dat2d = xr.Dataset()
    slc = dat.sel(z=slice(-dz, dz))
    twop = slc["T"] < 2.0e4
    Pth = slc["pressure"]
    Pturb = slc["density"] * slc["velocity3"] ** 2
    Pimag = 0.5 * (
        slc["cell_centered_B1"] ** 2
        + slc["cell_centered_B2"] ** 2
        - slc["cell_centered_B3"] ** 2
    )

    Ptot = Pth + Pturb + Pimag

    dat2d["pth"] = (Pth * twop).sum(dim="z") / twop.sum(dim="z")
    dat2d["pturb"] = (Pturb * twop).sum(dim="z") / twop.sum(dim="z")
    dat2d["pimag"] = (Pimag * twop).sum(dim="z") / twop.sum(dim="z")
    dat2d["ptot"] = (Ptot * twop).sum(dim="z") / twop.sum(dim="z")
    return dat2d * s.u.pok


def get_sfrmap(s, num, tbin=40):
    dx = s.domain["dx"][0]
    dy = s.domain["dy"][1]
    msp = grid_msp(s, num, 0, tbin)
    sfr = msp * s.u.Msun / tbin / dx / dy
    return sfr.to_dataset(name="sigma_sfr")


def get_prfm_quantities(s, num, overwrite=False):
    """Calculate Q(x, y) of the warm-cold gas.

    Parameters
    ----------
    s : pyathena.LoadSimTIGRESSGC
        LoadSim instance.
    num : int
        Snapshot number
    overwrite : bool, optional
        Flag to overwrite
    """
    fname = Path(s.basedir, "prfm_quantities", "prfm.{:04}.nc".format(num))
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        print("File {} already exists; skipping...".format(fname))
        return

    msg = "[prfm_quantities] processing model {} num {}"
    print(msg.format(s.basename, num))

    ds = s.load_vtk(num, id0=False)
    dat = ds.get_field(
        [
            "density",
            "velocity",
            "pressure",
            "gravitational_potential",
            "cell_centered_B",
            "T",
        ]
    )
    dat = add_weights(s, dat)
    prfm = get_weights(s, dat)
    prfm.update(get_pressures(s, dat))
    prfm.update(get_sfrmap(s, num))
    prfm["sigma"] = dat["density"].sum(dim="z") * s.domain["dx"][2] * s.u.Msun
    prfm.to_netcdf(fname)
    ds.close()


if __name__ == "__main__":
    COMM = MPI.COMM_WORLD

    basedir = "/tigress/changgoo/TIGRESS-NCR/R8_4pc_NCR"

    # savdir = '/tigress/jk11/tmp4/'
    # savdir_pkl = '/tigress/jk11/tmp3/'
    savdir = None
    savdir_pkl = None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", type=str, default=basedir, help="Name of the basedir."
    )
    args = vars(parser.parse_args())
    locals().update(args)

    s = pa.LoadSimTIGRESSNCR(basedir, verbose=False)

    nums = s.nums

    if COMM.rank == 0:
        print("basedir, nums", s.basedir, nums)
        nums = split_container(nums, COMM.size)
    else:
        nums = None

    mynums = COMM.scatter(nums, root=0)
    print("[rank, mynums]:", COMM.rank, mynums)

    time0 = time.time()
    for num in mynums:
        get_prfm_quantities(s, num, overwrite=True)

        n = gc.collect()
        print("Unreachable objects:", n, end=" ")
        print("Remaining Garbage:", end=" ")
        pprint.pprint(gc.garbage)
        sys.stdout.flush()

    # Make movies
    COMM.barrier()
