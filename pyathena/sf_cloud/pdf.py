# pdf.py

import numpy as np
import astropy.units as au
import astropy.constants as ac

from ..load_sim import LoadSim

class PDF:

    bins=dict(nH=np.logspace(-2,5,100),
              T=np.logspace(0,5,100),
              pok=np.logspace(0,7,100)
    )
    
    @LoadSim.Decorators.check_pickle
    def read_pdf2d(self, num,
                   bin_fields=None, bins=None, prefix='pdf2d',
                   savdir=None, force_override=False):

        bin_fields_def = [['nH', 'pok'], ['nH', 'T']]
        if bin_fields is None:
            bin_fields = bin_fields_def

        ds = self.load_vtk(num=num)
        res = dict()
        
        for bf in bin_fields:
            k = '-'.join(bf)
            res[k] = dict()
            dd = ds.get_field(bf)
            xdat = dd[bf[0]].data.flatten()
            ydat = dd[bf[1]].data.flatten()
            # Volume weighted hist
            weights = None
            H, xe, ye = np.histogram2d(xdat, ydat, (self.bins[bf[0]], self.bins[bf[1]]),
                                       weights=weights)
            res[k]['H'] = H
            res[k]['xe'] = xe
            res[k]['ye'] = ye
            
            # Density weighted hist
            weights = (ds.get_field('nH'))['nH'].data.flatten()
            Hw, xe, ye = np.histogram2d(xdat, ydat, (self.bins[bf[0]], self.bins[bf[1]]),
                                        weights=weights)
            res[k]['Hw'] = Hw

        return res

    
