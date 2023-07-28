#!/usr/bin/env python

import os
import os.path as osp
import gc
import time
from mpi4py import MPI
import matplotlib.pyplot as plt
import pprint
import argparse
import sys
import pickle

import pyathena as pa
from pyathena.util.split_container import split_container
from pyathena.plt_tools.make_movie import make_movie
from pyathena.tigress_ncr.phase import *
from pyathena.tigress_ncr.cooling_breakdown import *
from pyathena.tigress_ncr.do_volume import *

units = dict()
units.update(dict(velocity_z='km/s'))
units.update(dict(vzin='km/s'))
units.update(dict(vzout='km/s'))
units.update(dict(sound_speed='km/s'))
units.update({("gas","temperature"):"K"})
units.update(dict(CII_luminosity="erg/s"))
units.update({"xray_luminosity_0.5_7.0_keV":"erg/s"})
units.update({"H_alpha_luminosity":"erg/s"})
units.update({"HI_21cm_luminosity":"erg/s"})
units.update({("gas","x"):"pc"})
units.update({("gas","y"):"pc"})

extrema=dict()
extrema.update(dict(velocity_z=(-1000,1000)))
extrema.update(dict(vzout=(0.1,2.e3)))
extrema.update(dict(vzin=(0.1,2.e3)))
extrema.update(dict(sound_speed=(0.1,2.e3)))
extrema.update({('gas','temperature'):(1,1.e9)})
extrema.update(dict(CII_luminosity=(1.e20,1.e40)))
extrema.update({"xray_luminosity_0.5_7.0_keV":(1.e20,1.e40)})
extrema.update({"H_alpha_luminosity":(1.e20,1.e40)})
extrema.update({"HI_21cm_luminosity":(1.e15,1.e35)})

def convert_profile_to_dataset(profile):
    # convert profiles to xarray dataset
    dset = xr.Dataset()
    for (g,k), v in profile.items():
        x = 0.5*(profile.x_bins[:-1]+profile.x_bins[1:])
        g, xf = profile.x_field
        coords = [x]
        dims = [xf]
        if hasattr(profile,'y_field'):
            y = 0.5*(profile.y_bins[:-1]+profile.y_bins[1:])
            g, yf = profile.y_field
            coords.append(y)
            dims.append(yf)
        if hasattr(profile,'z_field'):
            z = 0.5*(profile.z_bins[:-1]+profile.z_bins[1:])
            g, zf = profile.z_field
            coords.append(z)
            dims.append(zf)
        da = xr.DataArray(v,coords=coords,dims=dims)
        dset[k] = da
    return dset

def get_inout_pdfs(box):
    out_profile = yt.create_profile(
        data_source=box,
        bin_fields=[("gas", "sound_speed"),("gas", "vzout")],
        fields=[
            ("gas", "cell_volume"),
            ("gas", "cell_mass"),
            ("gas", "HI_21cm_luminosity"),
            ("gas", "H_alpha_luminosity"),
            ("gas", "CII_luminosity"),
            ("gas", "xray_luminosity_0.5_7.0_keV"),
            ("gas", "specific_total_energy"),
            ("gas", "momentum_density_z"),
            ("gas", "total_energy_flux_z"),
            ("gas", "total_energy_density"),
        ],
        n_bins=(256, 256),
        weight_field=None,
        units=dict(vzout='km/s', sound_speed='km/s'),#, temperature='K'),
        extrema=dict(vzout=(0.1,2.e3), sound_speed=(0.1,2.e3)),#, temperature=(10,1.e9)),
    )

    in_profile = yt.create_profile(
        data_source=box,
        bin_fields=[("gas", "sound_speed"),("gas", "vzin")],
        fields=[
            ("gas", "cell_volume"),
            ("gas", "cell_mass"),
            ("gas", "HI_21cm_luminosity"),
            ("gas", "H_alpha_luminosity"),
            ("gas", "CII_luminosity"),
            ("gas", "xray_luminosity_0.5_7.0_keV"),
            ("gas", "specific_total_energy"),
            ("gas", "momentum_density_z"),
            ("gas", "total_energy_flux_z"),
            ("gas", "total_energy_density"),
        ],
        n_bins=(256, 256),
        weight_field=None,
        units=dict(vzin='km/s', sound_speed='km/s'),#, temperature='K'),
        extrema=dict(vzin=(0.1,2.e3), sound_speed=(0.1,2.e3)),# temperature=(10,1.e9)),
    )

    pdfin = convert_profile_to_dataset(in_profile)
    pdfout = convert_profile_to_dataset(out_profile)

    return pdfin, pdfout

def get_pdfs(box,xf,yf):
    bxf = ("gas", xf) if not type(xf) == tuple else xf
    byf = ("gas", yf) if not type(yf) == tuple else yf
    if xf.startswith('xray'):
        extrema['velocity_z']=(-1024,1024)
    else:
        extrema['velocity_z']=(-256,256)
    profile = yt.create_profile(
        data_source=box,
        bin_fields=[bxf,byf],
        fields=[
            ("gas", "cell_volume"),
            ("gas", "cell_mass"),
            ("gas", "H_p0_number_density"),
            ("gas", "H_p1_number_density"),
            ("gas", "H2_number_density"),
            ("gas", "specific_total_energy"),
            ("gas", "momentum_density_z"),
            ("gas", "total_energy_flux_z"),
            ("gas", "total_energy_density"),
        ],
        n_bins=(512, 512),
        units = {f:units[f] for f in [xf,yf]},
        extrema = {f:extrema[f] for f in [xf,yf]},
        weight_field=None,
    )


    pdf = convert_profile_to_dataset(profile)

    return pdf

def get_ppv(box,lf):
    ds = box.ds
    Nx, Ny, Nz = ds.domain_dimensions
    le = ds.domain_left_edge.v
    re = ds.domain_right_edge.v

    if lf.startswith('xray'):
        extrema['velocity_z']=(-1024,1024)
    else:
        extrema['velocity_z']=(-256,256)
    extrema.update({('gas','x'):(le[0],re[0])})
    extrema.update({('gas','y'):(le[1],re[1])})

    bfs = [("gas","x"),("gas","y"),"velocity_z"]
    profile = yt.create_profile(
        data_source=box,
        bin_fields=[("gas","x"),("gas","y"),("gas","velocity_z")],
        fields=[
            ("gas", "cell_volume"),
            ("gas", "cell_mass"),
            ("gas", "HI_21cm_luminosity"),
            ("gas", "H_alpha_luminosity"),
            ("gas", "CII_luminosity"),
            ("gas", "xray_luminosity_0.5_7.0_keV"),
            ("gas", "specific_total_energy"),
            ("gas", "momentum_density_z"),
            ("gas", "total_energy_flux_z"),
            ("gas", "total_energy_density"),
        ],
        n_bins=(int(Nx/2),int(Ny/2),128),
        units = {f:units[f] for f in bfs},
        extrema = {f:extrema[f] for f in bfs},
        weight_field=None,
    )

    return convert_profile_to_dataset(profile)

def do_pdf(s,num):
    ds = s.ytload(num)
    ds = add_fields(s,ds,xray=True)

    foutdir = os.path.join(s.savdir,"wind_pdf")
    os.makedirs(foutdir,exist_ok=True)

    Lz = ds.domain_width[2].v
    zmin = ds.domain_left_edge[2].v
    zmax = ds.domain_right_edge[2].v
    dz = Lz/8
    Nx, Ny, Nz = ds.domain_dimensions
    le = ds.domain_left_edge.v
    re = ds.domain_right_edge.v


    lumfields = ["HI_21cm_luminosity", "H_alpha_luminosity", "CII_luminosity", "xray_luminosity_0.5_7.0_keV"]

    for lf in lumfields:
        print(lf)
        pdflist = []
        ppvlist = []
        for z1, z2 in zip(np.arange(zmin,zmax,dz),np.arange(zmin+dz,zmax+dz,dz)):
            print(z1,z2)
            # pdfs
            box = ds.r[:,:,z1:z2]
            pdf = get_pdfs(box,lf,"velocity_z")
            pdf = pdf.assign_coords(zmin = z1)
            pdflist.append(pdf)

            # ppvs
            if num % 10 == 0:
                if z1<0:
                    box = ds.r[:,:,:z2]
                else:
                    box = ds.r[:,:,z1:]
                ppv = get_ppv(box,lf)
                ppv = ppv.assign_coords(zmin = z1)
                ppvlist.append(ppv)
        pdf = xr.concat(pdflist,dim='zmin')

        for f in list(pdf.keys()):
            pdf.attrs[f] = list()
        for z1, z2 in zip(np.arange(zmin,zmax,dz),np.arange(zmin+dz,zmax+dz,dz)):
            box = ds.r[:,:,z1:z2]
            for f in list(pdf.keys()):
                total = np.abs(box.quantities.total_quantity(f))
                pdf.attrs[f].append(total)

        fout = os.path.join(foutdir,f"{lf}_pdf_{num:04d}.nc")
        pdf.to_netcdf(fout)

        pdf.close()

        # save ppv
        if num % 10 == 0:
            ppv = xr.concat(ppvlist,dim='zmin')
            fout = os.path.join(foutdir,f"{lf}_ppv_{num:04d}.nc")
            ppv.to_netcdf(fout)
            ppv.close()

    # in/out pdfs
    ipdflist = []
    opdflist = []

    for z1, z2 in zip(np.arange(zmin,zmax,dz),np.arange(zmin+dz,zmax+dz,dz)):
        print(z1,z2)
        box = ds.r[:,:,z1:z2]
        pdfin, pdfout = get_inout_pdfs(box)
        pdfin = pdfin.assign_coords(zmin = z1)
        pdfout = pdfout.assign_coords(zmin = z1)
        ipdflist.append(pdfin)
        opdflist.append(pdfout)

    pdfin = xr.concat(ipdflist,dim='zmin')
    pdfout = xr.concat(opdflist,dim='zmin')

    fout = os.path.join(foutdir,f"pdfin_{num:04d}.nc")
    pdfin.to_netcdf(fout)
    pdfin.close()

    fout = os.path.join(foutdir,f"pdfout_{num:04d}.nc")
    pdfout.to_netcdf(fout)
    pdfout.close()

if __name__ == "__main__":

    COMM = MPI.COMM_WORLD

    basedir_def = "/tigress/changgoo/TIGRESS-NCR/R8_4pc_NCR"

    # savdir = '/tigress/jk11/tmp4/'
    # savdir_pkl = '/tigress/jk11/tmp3/'
    savdir = None
    savdir_pkl = None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", type=str, default=basedir_def, help="Name of the basedir."
    )
    args = vars(parser.parse_args())
    locals().update(args)

    s = pa.LoadSimTIGRESSNCR(basedir, verbose=False)
    # tar vtk files
    if s.nums_rawtar is not None:
        nums = s.nums_rawtar
        if COMM.rank == 0:
            print("basedir, nums", s.basedir, nums)
            nums = split_container(nums, COMM.size)
        else:
            nums = None

        mynums = COMM.scatter(nums, root=0)
        for num in mynums:
            s.create_tar(num=num,kind='vtk',remove_original=True,overwrite=True)
        COMM.barrier()

        # reading it again
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
        print(num, end=" ")
        # f1 = draw_Tpdf(s,num)
        # allslc = s.read_allslc(num)
        do_pdf(s,num)

        n = gc.collect()
        print("Unreachable objects:", n, end=" ")
        print("Remaining Garbage:", end=" ")
        pprint.pprint(gc.garbage)
        sys.stdout.flush()

    # Make movies
    COMM.barrier()