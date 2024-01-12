from ..load_sim import LoadSim

import numpy as np
import os.path as osp
import xarray as xr

bins_def={
    'nH': np.logspace(-4, 4, 81),
    'xH2': np.logspace(-6, 0, 81),
    'T': np.logspace(0.0, 8.5, 141),
    'pok': np.logspace(1, 7, 71),
    'fshld_H2': np.logspace(-6,0,61),
    'cool_rate': np.logspace(-30, -18, 81),
    'net_cool_rate': np.append(np.append(-np.logspace(-18,-25,51),
                                         np.linspace(-1e-25,1e-25,51)[1:-1]),
                               np.logspace(-25,-18,51)),
    'xe': np.linspace(0, 1.3, 51),
}

class Hist2d:

    @LoadSim.Decorators.check_pickle
    def read_hist2d_all(self, nums=None, hist2d_kwargs=dict(),
                        savdir=None,
                        prefix='hist2d_all',
                        force_override=False):
        """Function to calculate sum of hist2d output from multiple snapshots

        Parameters
        ----------
        nums : list of integer
            vtk snapshot numbers
        hist2d_args : dict
            dictionary that contains keyword arguments for read_hist2d
        savdir : str
            Directory to save pickle file
        prefix : str
            Prefix to be used for pickle file name
        force_override : bool
            Recalculate histograms. Do not read from pickle file even if it
            exists.
        """

        if nums is None:
            nums = self.nums

        rr = dict()
        print('[read_hist2d_all]:', end=' ')
        for i,num in enumerate(nums):
            print(num, end=' ')
            r = self.read_hist2d(num, **hist2d_kwargs)
            if i == 0:
                for k in r.keys():
                    if k == 'time_code':
                        rr[k] = []
                    else:
                        rr[k] = dict()
                        rr[k]['x'] = r[k]['x']
                        rr[k]['y'] = r[k]['y']
                        rr[k]['H'] = np.zeros_like(r[k]['H'])

            for k in r.keys():
                if k == 'time_code':
                    rr[k].append(r[k])
                else:
                    rr[k]['H'] += r[k]['H']


        print('')

        return rr

    @LoadSim.Decorators.check_pickle
    def read_hist2d(self, num,
                    bin_fields=[['nH', 'pok'], ['nH','xH2'], ['nH','fshld_H2'],
                                ['T','xH2'], ['T','fshld_H2']],
                    weights=[None,None,None,None,None],
                    bins=None,
                    sel=None, # Selection function
                    prefix='hist2d',
                    savdir=None, force_override=False):
        """
        Parameters
        ----------
        num : int
            vtk snapshot number
        bin_fields : list of list of str
            List of a pair of field names to make 2d histograms
        weights : list of str
            name of fields names to use as weights. Volume-weighted for uniform
            grid if None.
        bins : dict
            binning of fields (used for argument of np.histogram2d)
        sel : lambda function
            Function that specifies selection criteria
            For example, to select cells with T < 3e4 and |z| < 300, use
            sel = lambda d: (d['T'] < 3e4) & (d['z'] < 300) & (d['z] > -300)
        prefix : str
            Prefix to be used for pickle file name
        savdir : str
            Directory to save pickle file
        force_override : bool
            Recalculate histograms. Do not read from pickle file even if it
            exists.

        Returns
        -------
        Dictionary containing bins, histograms, and time_code.
        """

        # Check if bin_fields is not nested
        if not any(isinstance(i, tuple) or isinstance(i, list) for i in bin_fields):
            bin_fields = [bin_fields]
        # Check if weights is not a sequence
        if not (isinstance(weights, tuple) or isinstance(weights, list)):
            weights = [weights]

        if bins is None:
            bins = bins_def

        # Determine fields to read
        bf_set = set.union(*map(set, bin_fields))
        w_set = set(weights) - set([None])
        fields = list(set.union(bf_set, w_set))
        if sel is not None:
            import inspect
            import re
            sel_str = str(inspect.getsourcelines(sel)[0])
            sel_str = sel_str.split(':')[1].replace("\\", "")
            sel_set = set(re.findall(r'["\'](.*?)["\']',sel_str))
            sel_set -= set(['x','y','z'])
            fields = list(set.union(sel_set, set(fields)))

        if (bf_set - set(bins.keys())) != set():
            print('bins not defined:', bf_set - set(bins.keys()))

        ds = self.load_vtk(num)

        try:
            self.load_chunk(num)
            dd = self.get_field_from_chunk(fields)
        except IOError:
            print('Reading fields from vtk:', fields)
            dd = ds.get_field(fields)

        if sel is not None:
            dd = dd.where(sel, drop=True)

        r = dict()
        for bf,w in zip(bin_fields, weights):
            k = '-'.join(bf)
            if w is not None:
                k += '-{0:s}'.format(w)

            r[k] = dict()

            xdat = dd[bf[0]].data.flatten()
            ydat = dd[bf[1]].data.flatten()
            if w is not None:
                wdat = dd[w].data.flatten()
            else:
                wdat = None
            H,x,y = np.histogram2d(xdat, ydat, bins=(bins[bf[0]], bins[bf[1]]),
                                   weights=wdat)
            r[k]['H'] = H
            r[k]['x'] = x
            r[k]['y'] = y

        r['time_code'] = ds.domain['time']

        return r

    def load_chunk(self,num,scratch_dir='/scratch/gpfs/changgoo/TIGRESS-NCR/'):
        """Read in temporary outputs in scartch directory
        """
        scratch_dir += osp.join(self.basename,'midplane_chunk')
        chunk_file = osp.join(scratch_dir,'{:s}.{:04d}.hLx.nc'.format(self.problem_id,num))
        if not osp.isfile(chunk_file):
            raise IOError("File does not exist: {}".format(chunk_file))
        with xr.open_dataset(chunk_file) as chunk:
            self.data_chunk = chunk

    def get_field_from_chunk(self,fields):
        """Get fields using temporary outputs in scartch directory
        """
        dd = xr.Dataset()
        for f in fields:
            if f in self.data_chunk:
                dd[f] = self.data_chunk[f]
            elif f in self.dfi:
                dd[f] = self.dfi[f]['func'](self.data_chunk,self.u)
            else:
                raise IOError("{} is not available".format(f))
        return dd
