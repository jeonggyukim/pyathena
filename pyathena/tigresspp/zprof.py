import os
import os.path as osp

import xarray as xr
import pandas as pd

from ..load_sim import LoadSim


class Zprof:
    @LoadSim.Decorators.check_netcdf
    def load_zprof(
        self,
        prefix="merged_zprof",
        filebase=None,
        savdir=None,
        force_override=False,
        dryrun=False,
    ):
        if dryrun:
            mtime = -1
            for f in self.files["zprof"]:
                mtime = max(osp.getmtime(f), mtime)
            return max(mtime, osp.getmtime(__file__))

        if self.ff.zprof_separate_vz:
            dlist_pvz = dict()
            dlist_nvz = dict()
        else:
            dlist = dict()
        for fname in self.files["zprof"]:
            with open(fname, "r") as f:
                header = f.readline()
            data = pd.read_csv(fname, skiprows=1)
            tmp = data.pop("k")
            data.index = data.x3v
            time = eval(
                header[header.find("time") : header.find("cycle")]
                .split("=")[-1]
                .strip()
            )
            if self.ff.zprof_separate_vz:
                phase = (
                    header[header.find("phase") : header.find("vz_dir")]
                    .split("=")[-1]
                    .strip()
                )
                vz = eval(
                    header[header.find("vz_dir") : header.find("variable")]
                    .split("=")[-1]
                    .strip()
                )
                if vz > 0:
                    dlist = dlist_pvz
                else:
                    dlist = dlist_nvz
            else:
                phase = (
                    header[header.find("phase") : header.find("variable")]
                    .split("=")[-1]
                    .strip()
                )
            for ph in self.phase:
                if ph in fname:
                    if phase not in dlist:
                        dlist[phase] = []
                    dlist[phase].append(
                        data.to_xarray().assign_coords(time=time).rename(x3v="z")
                    )

        if self.ff.zprof_separate_vz:
            dset_vz = []
            for vz_dir, dlist in zip([1, -1], [dlist_pvz, dlist_nvz]):
                dset = []
                for phase in dlist:
                    dset.append(
                        xr.concat(dlist[phase], dim="time").assign_coords(phase=phase)
                    )
                dset_vz.append(
                    xr.concat(dset, dim="phase").assign_coords(vz_dir=vz_dir)
                )
            dset = xr.concat(dset_vz, dim="vz_dir")
        else:
            dset = []
            for phase in dlist:
                dset.append(
                    xr.concat(dlist[phase], dim="time").assign_coords(phase=phase)
                )
            dset = xr.concat(dset, dim="phase")

        return dset

    def load_zprof_postproc_one(self, num, force_override=False, zpoutdir=None):
        if zpoutdir is None:
            zpoutdir = osp.join(self.savdir, "zprof_postproc")
        os.makedirs(zpoutdir, exist_ok=True)
        zpoutfile = osp.join(zpoutdir, f"{self.problem_id}.{num:05d}.zprof.nc")
        if not os.path.isfile(zpoutfile) or force_override:
            self.logger.info(f'creating: {zpoutfile}')
            ds = self.get_data(num)
            ds.load()
            zprof = self.construct_zprof(ds)
            zprof = zprof.assign_coords(time=ds.attrs["Time"])
            zprof.to_netcdf(zpoutfile)
        else:
            self.logger.info(f'reading: {zpoutfile}')
            with xr.open_dataset(zpoutfile) as zprof:
                zprof.load()
        return zprof

    def load_zprof_postproc(self, zpoutdir=None, force_override=False):
        if zpoutdir is None:
            zpoutdir = osp.join(self.savdir, "zprof_postproc")
            self.logger.info(f'save folder is set to {zpoutdir}')
        os.makedirs(zpoutdir, exist_ok=True)
        zplist = []
        if force_override:
            self.logger.info(f'forced recreation of postproc zprof')
        for num in self.nums:
            zprof = self.load_zprof_postproc_one(num, zpoutdir=zpoutdir,
                                                 force_override=force_override)
            zplist.append(zprof)
        zp_pp = xr.concat(zplist, dim="time")

        # load original zprof and synchronize phase and time
        if not hasattr(self, "zprof"):
            self.zprof = self.load_zprof()

        zp_pp = zp_pp.assign_coords(phase=self.zprof.phase[:-1])
        zp_pp["area"] = (
            self.zprof["area"].sel(phase=zp_pp.phase).interp(time=zp_pp["time"])
        )

        self.zp_pp_ph = xr.concat(
            [
                zp_pp.sel(phase=["CNM", "UNM", "WNM"])
                .sum(dim="phase")
                .assign_coords(phase="wc"),
                zp_pp.sel(phase=["WHIM", "HIM"])
                .sum(dim="phase")
                .assign_coords(phase="hot"),
            ],
            dim="phase",
        )

        self.zp_pp = zp_pp
        return zp_pp
