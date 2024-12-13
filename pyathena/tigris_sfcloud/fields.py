import matplotlib as mpl
import numpy as np
# import astropy.constants as ac
# import astropy.units as au

from matplotlib.colors import Normalize, LogNorm

from ..plt_tools.cmap import cmap_shift


class Fields:
    def temperature_dfi(self):
        mydfi = dict()
        mydfi["field_dep"] = ["density", "pressure"]

        def _T(d, u):
            T1 = (d["pressure"] / d["density"] * u.temperature_mu).value
            logT1 = np.log10(T1)
            mu = self.coolftn["mu"](logT1)
            return mu * T1

        mydfi["func"] = _T
        mydfi["label"] = r"$T\;[{\rm K}]$"
        mydfi["cmap"] = cmap_shift(
            mpl.cm.RdYlBu_r, midpoint=3.0 / 7.0, name="cmap_pyathena_T"
        )
        mydfi["vminmax"] = (1e1, 1e7)
        mydfi["take_log"] = True

        if mydfi["take_log"]:
            mydfi["norm"] = LogNorm(*mydfi["vminmax"])
            mydfi["scale"] = "log"
        else:
            mydfi["norm"] = Normalize(*mydfi["vminmax"])
            mydfi["scale"] = "linear"
        mydfi["imshow_args"] = dict(
            norm=mydfi["norm"],
            cmap=mydfi["cmap"],
            cbar_kwargs=dict(label=mydfi["label"]),
        )

        return mydfi
