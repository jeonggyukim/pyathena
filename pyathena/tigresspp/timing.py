import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ..load_sim import LoadSim


class Timing:
    @LoadSim.Decorators.check_pickle
    def load_task_time(
        self, prefix="timing", filebase="tasktime", savdir=None, force_override=False
    ):
        """Read .task_time.txt file

        Parameters
        ----------
        groups : list, e.g., ['Hydro','Primitives','UserWork']
                 if provided, group tasks that have the same string in the list
                 everything else will be summed and stored in 'Others'

        Returns
        -------
        pandas.DataFrame

        The breakdown of time taken by each task of the time integrator
        """

        def from_block(block):
            info = dict()
            h = block[0].split(",")
            info["ncycle"] = int(h[0].split("=")[1])
            info["TaskList"] = h[1].split("=")[1]
            info["time"] = float(h[2].split("=")[1])
            for line in block[1:]:
                sp = line.split(",")
                name = sp[0].replace(" ", "")
                time = sp[1].split("=")[1]
                info[name] = float(time)
            return info

        with open(self.files["task_time"]) as fp:
            lines = fp.readlines()

        block_idx = []
        for i, line in enumerate(lines):
            if line.startswith("#"):
                block_idx.append(i)
        timing = dict()

        # read all
        for i, j in zip(block_idx[:-1], block_idx[1:]):
            info = from_block(lines[i:j])
            if info["TaskList"] not in timing:
                timing[info["TaskList"]] = dict()
            t_ = timing[info["TaskList"]]
            for k, v in info.items():
                if k == "TaskList":
                    continue
                if k in t_:
                    t_[k].append(v)
                else:
                    t_[k] = [v]
        for tn in timing:
            for k in timing[tn]:
                timing[tn][k] = np.array(timing[tn][k])
        for tn in timing:
            timing[tn] = pd.DataFrame(timing[tn])

        return timing

    def group_task_time(self, groups=None):
        """Read .task_time.txt file

        Parameters
        ----------
        groups : list, e.g., ['Hydro','Primitives','UserWork']
                 if provided, group tasks that have the same string in the list
                 everything else will be summed and stored in 'Others'

        Returns
        -------
        pandas.DataFrame

        The breakdown of time taken by each task of the time integrator
        """

        timing = self.load_task_time()
        # need to add grouping options
        if groups is None:
            return timing

        grouped = dict()
        for tl in timing:
            ttl = timing[tl]
            keys = set(ttl.keys())
            for k in ["ncycle", "time"]:
                grouped[k] = ttl[k]
                keys = keys - {k}
            for g in groups:
                for k in ttl:
                    if g in k:
                        gid = "_".join([tl, g])
                        if gid not in grouped:
                            grouped[gid] = ttl[k]
                        else:
                            grouped[gid] += ttl[k]
                        keys = keys - {k}
            g = "Others"
            for k in keys:
                gid = "_".join([tl, g])
                if gid not in grouped:
                    grouped[gid] = ttl[k]
                else:
                    grouped[gid] += ttl[k]

        return pd.DataFrame(grouped)

    @LoadSim.Decorators.check_pickle
    def load_loop_time(
        self, prefix="timing", filebase="looptime", savdir=None, force_override=False
    ):
        def from_one_line(line):
            info = dict()
            for sp in line.split(","):
                name, value = sp.split("=")
                if name in ["ncycle"]:
                    info[name.replace(" ", "")] = int(value)
                else:
                    info[name.replace(" ", "")] = float(value)
            return info

        with open(self.files["loop_time"]) as fp:
            lines = fp.readlines()

        timing = dict()
        info = from_one_line(lines[0])
        for k in info:
            timing[k] = []

        for line in lines:
            info = from_one_line(line)
            for k, v in info.items():
                timing[k].append(v)
        return pd.DataFrame(timing)

    def plt_loop_timing(self):
        lt = self.load_loop_time()
        ncells = np.prod(self.domain["Nx"])

        keys = list(set(lt.keys()) - {"ncycle", "time", "Nblocks"})
        keys = list(lt[keys].median().sort_values(ascending=True).keys())

        fig, axes = plt.subplots(1, 2, figsize=(8, 3))

        plt.sca(axes[0])
        lt[keys].boxplot(vert=False, showfliers=False)
        plt.xlabel("time/step [s]")

        plt.sca(axes[1])
        (ncells / lt[keys]).boxplot(vert=False, showfliers=False)
        plt.xlim(1.0e4, 1.0e8)
        plt.axvline(1.0e5)
        plt.xscale("log")
        plt.xlabel("zcs")

        plt.tight_layout()
        # plt.savefig("timing.png", bbox_inches="tight")
        return fig

    def plt_timing(self, plot=True):
        if "ncycle_out_timing" in self.par["time"]:
            dncycle = self.par["time"]["ncycle_out_timing"]
        else:
            dncycle = 1.0
        tt = self.load_task_time()
        lt = self.load_loop_time()
        tasklist = list(tt["TimeIntegrator"].keys())
        optasklist = list(tt["OperatorSplitTaskList"].keys())
        dummy = tasklist.pop(0)
        dummy = tasklist.pop(0)
        dummy = optasklist.pop(0)
        dummy = optasklist.pop(0)

        timing = dict()

        timing["All"] = lt["All"]
        timing["SelfGravity"] = lt["SelfGravity"]

        mhdlist = [k for k in tasklist if ("Hydro" in k or "Field" in k or "EMF" in k)]
        crlist = [k for k in set(tasklist) - set(mhdlist) if "CR" in k]
        scalarlist = [
            k for k in set(tasklist) - set(mhdlist) - set(crlist) if "Scalar" in k
        ]
        particlelist = [
            k
            for k in set(tasklist) - set(mhdlist) - set(crlist) - set(scalarlist)
            if "Particle" in k
        ]
        otherlist = (
            set(tasklist)
            - set(mhdlist)
            - set(crlist)
            - set(scalarlist)
            - set(particlelist)
            - set(["Primitives"])
        )
        for name, sel in zip(
            ["MHD", "CR", "Scalar", "Particle", "Primitives", "Others"],
            [mhdlist, crlist, scalarlist, particlelist, ["Primitives"], otherlist],
        ):
            # print(name, tt["TimeIntegrator"][list(sel)].mean().sum())
            timing[name] = tt["TimeIntegrator"][list(sel)].sum(axis=1)/dncycle

        op_mhdlist = [
            k for k in optasklist if ("Hydro" in k or "Field" in k or "EMF" in k)
        ]
        op_crlist = [k for k in set(optasklist) - set(op_mhdlist) if "CR" in k]
        op_scalarlist = [
            k
            for k in set(optasklist) - set(op_mhdlist) - set(op_crlist)
            if "Scalar" in k
        ]
        op_particlelist = [
            k
            for k in set(optasklist)
            - set(op_mhdlist)
            - set(op_crlist)
            - set(op_scalarlist)
            if "Particle" in k
        ]
        op_coolinglist = [            k
            for k in set(optasklist)
            - set(op_mhdlist)
            - set(op_crlist)
            - set(op_scalarlist)
            - set(op_particlelist)
            if ("Cooling" in k or "Photochemistry" in k)
        ]

        op_otherlist = (
            set(optasklist)
            - set(op_mhdlist)
            - set(op_crlist)
            - set(op_scalarlist)
            - set(op_particlelist)
            - set(op_coolinglist)
            - set(["Primitives"])
        )

        for name, sel in zip(
            ["MHD", "CR", "Scalar", "Particle", "Primitives", "Others"],
            [
                op_mhdlist,
                op_crlist,
                op_scalarlist,
                op_particlelist,
                ["Primitives"],
                op_otherlist,
            ],
        ):
            # print(name, tt["OperatorSplitTaskList"][list(sel)].mean().sum())
            timing[name] += tt["OperatorSplitTaskList"][list(sel)].sum(axis=1)/dncycle
        timing["Cooling"] = tt["OperatorSplitTaskList"][list(op_coolinglist)].sum(axis=1)/dncycle

        if not plot:
            return timing


        fig, axes = plt.subplots(1, 2, figsize=(8, 3))

        plt.sca(axes[0])
        plt.boxplot(
            [v.dropna() for v in timing.values()],
            tick_labels=timing.keys(),
            vert=False,
            whis=(5, 95),
            flierprops=dict(marker="+"),
        )
        plt.xscale("log")
        plt.xlabel("time/step [s]")

        plt.sca(axes[1])
        ncells = np.prod(self.domain["Nx"])
        plt.boxplot(
            [ncells/v.dropna() for v in timing.values()],
            tick_labels=timing.keys(),
            vert=False,
            whis=(5, 95),
            flierprops=dict(marker="+"),
        )
        plt.xlim(1.0e4, 1.0e7)
        plt.grid(True)
        plt.xscale("log")
        plt.xlabel("zcs")
        plt.tight_layout()

        return timing
