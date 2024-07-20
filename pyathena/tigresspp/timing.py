import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Timing:
    def load_task_time(self, groups=None):
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

        # need to add grouping options
        if groups is None:
            # default groups
            groups = ["Hydro"]

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

    def load_loop_time(self):
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

    def plt_timing(self):
        lt = self.load_loop_time()
        ncells = np.prod(self.domain["Nx"])

        keys = list(set(lt.keys()) - {"ncycle", "time", "Nblocks"})
        keys = list(lt[keys].median().sort_values(ascending=True).keys())

        fig, axes = plt.subplots(1, 2, figsize=(8, 3), num=1)

        plt.sca(axes[0])
        lt[keys].boxplot(vert=False, showfliers=False)
        plt.xlabel("time/step [s]")

        plt.sca(axes[1])
        (ncells / lt[keys]).boxplot(vert=False, showfliers=False)
        plt.xlim(1.e4,1.e8)
        plt.axvline(1.e5)
        plt.xscale("log")
        plt.xlabel("zcs")

        plt.tight_layout()
        # plt.savefig("timing.png", bbox_inches="tight")
        return fig