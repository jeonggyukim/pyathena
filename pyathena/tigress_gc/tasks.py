"""Module containing functions that are not generally reusable"""

# python modules
from pathlib import Path

# pyathena modules
from pyathena.tigress_gc import tools

def save_azimuthal_averages(s, overwrite=False):
    """Calculates azimuthal averages and save to file

    Args:
        s: pyathena.LoadSim instance
    """
    for num in s.nums:
        fname = Path(s.basedir, 'azimuthal_averages_warmcold',
                     'gc_azimuthal_average.{:04}.nc'.format(num))
        fname.parent.mkdir(exist_ok=True)
        if fname.exists() and not overwrite:
            print('File {} already exists; skipping...'.format(fname))
            continue
        print('processing model {} num {}'.format(s.basename, num))
        rprf = tools.calculate_azimuthal_averages(s, num, warmcold=True)
        rprf.to_netcdf(fname)
