# profile_1d.py

import numpy as np
from scipy import stats
from inspect import getsource

from ..load_sim import LoadSim

class Profile1D:

    @LoadSim.Decorators.check_pickle
    def get_profile1d(self, num, fields_y, field_x='r', bins=None, statistic='mean',
                      prefix='profile1d', savdir=None, force_override=False):
        """
        Function to calculate 1D profile(s) and pickle using
        scipy.stats.binned_statistics

        Parameters
        ----------
        num : int
            vtk output number
        fields_y : (list of) str
            Fields to be profiled
        fields_x : str
            Field for binning
        bins : int or sequence of scalars, optional
            If bins is an int, it defines the number of equal-width bins in the
            given range. If bins is a sequence, it defines the bin edges,
            including the rightmost edge, allowing for non-uniform bin widths.
            Values in x that are smaller than lowest bin edge are assigned to
            bin number 0, values beyond the highest bin are assigned to
            bins[-1]. If the bin edges are specified, the number of bins will
            be, (nx = len(bins)-1). The default value is np.linspace(x.min(),
            x.max(), 50)
        statistic : (list of) string or callable
            The statistic to compute (default is ‘mean’). The following
            statistics are available:
            ‘mean’ : compute the mean of values for points within each bin.
            Empty bins will be represented by NaN.
            ‘std’ : compute the standard deviation within each bin. This is
            implicitly calculated with ddof=0.
            ‘median’ : compute the median of values for points within each bin.
            Empty bins will be represented by NaN.
            ‘count’ : compute the count of points within each bin. This is
            identical to an unweighted histogram. values array is not
            referenced.
            ‘sum’ : compute the sum of values for points within each bin. This
            is identical to a weighted histogram.
            ‘min’ : compute the minimum of values for points within each bin.
            Empty bins will be represented by NaN.
            ‘max’ : compute the maximum of values for point within each bin.
            Empty bins will be represented by NaN.
            function : a user-defined function which takes a 1D array of values,
            and outputs a single numerical statistic. This function will be
            called on the values in each bin. Empty bins will be represented by
            function([]), or NaN if this returns an error.
        savdir : str, optional
            Directory to pickle results
        prefix : str
            Prefix for python pickle file
        force_override : bool
            Flag to force read of starpar_vtk file even when pickle exists
        """

        fields_y = np.atleast_1d(fields_y)
        statistic = np.atleast_1d(statistic)

        ds = self.load_vtk(num)
        ddy = ds.get_field(fields_y)
        ddx = ds.get_field(field_x)
        x1d = ddx[field_x].data.flatten()
        if bins is None:
            bins = np.linspace(x1d.min(), x1d.max(), 50)

        res = dict()
        res[field_x] = dict()
        for y in fields_y:
            res[y] = dict()

        get_lambda_name = lambda l: getsource(l).split('=')[0].strip()

        # Compute statistics
        for y in fields_y:
            y1d = ddy[y].data.flatten()
            for st in statistic:
                # Get name of statistic
                if callable(st):
                    if st.__name__ == "<lambda>":
                        name = get_lambda_name(st)
                    else:
                        name = st.__name__
                else:
                    name = st

                st, bine, _ = stats.binned_statistic(x1d, y1d, st, bins=bins)
                # Store result
                res[y][name] = st

        # bin edges
        res[field_x]['bine'] = bine
        # bin centers
        res[field_x]['binc'] = 0.5*(bine[1:] + bine[:-1])

        # Time of the snapshot
        res['time_code'] = ds.domain['time']
        res['time'] = ds.domain['time']*self.u.Myr

        return res
