
# import os
import os.path as osp

import xarray as xr
import pandas as pd

from ..load_sim import LoadSim

class Zprof:
    @LoadSim.Decorators.check_netcdf
    def load_zprof(self, prefix="merged_zprof", filebase=None,
                   savdir=None, force_override=False, dryrun=False):
        if dryrun:
            mtime = -1
            for f in self.files["zprof"]:
                mtime = max(osp.getmtime(f),mtime)
            return max(mtime,osp.getmtime(__file__))

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
                if vz>0:
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
            for vz_dir,dlist in zip([1,-1],[dlist_pvz,dlist_nvz]):
                dset = []
                for phase in dlist:
                    dset.append(xr.concat(dlist[phase], dim="time").assign_coords(phase=phase))
                dset_vz.append(xr.concat(dset, dim="phase").assign_coords(vz_dir=vz_dir))
            dset = xr.concat(dset_vz, dim="vz_dir")
        else:
            dset = []
            for phase in dlist:
                dset.append(xr.concat(dlist[phase], dim="time").assign_coords(phase=phase))
            dset = xr.concat(dset, dim="phase")

        return dset

    def calculate_zprof(self, num):
        data = self.get_data(num)