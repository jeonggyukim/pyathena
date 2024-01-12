# xray.py

import pandas as pd

from ..fields.xray_emissivity import get_xray_emissivity
from ..load_sim import LoadSim

class Xray:

    @LoadSim.Decorators.check_pickle
    def read_xray_all(self, nums=None, prefix='xray_all',
                      savdir=None, force_override=False):
        rr = dict()
        if nums is None:
            nums = self.nums


        print('num:', end=' ')
        for i,num in enumerate(nums):
            print(num, end=' ')
            r = self.read_xray(num=num, savdir=savdir, force_override=False)
            if i == 0:
                for k in r.keys():
                    rr[k] = []

            for k in r.keys():
                try:
                    rr[k].append(r[k].value.item())
                except:
                    rr[k].append(r[k])

        rr = pd.DataFrame(rr)
        return rr

    @LoadSim.Decorators.check_pickle
    def read_xray(self, num, Z_gas=1.0, emin_keV=0.5, emax_keV=7.0, prefix='L_X',
                  savdir=None, force_override=False):
        """
        Function to calculate x-ray luminosity of the snapshot
        """

        ds = self.load_vtk(num)
        dV = ds.domain['dx'].prod()*(self.u.length.cgs.value)**3
        d = ds.get_field(['density','temperature'])
        em = get_xray_emissivity(d['temperature'].data, Z_gas,
                                 emin_keV, emax_keV, energy=True)
        d['j_X'] = d['density']**2*em
        #dd['I_X'] = d['j_X'].sum(dim='z')*d.domain['dx'][2]*self.u.length.cgs.value

        res = dict()
        res['time'] = ds.domain['time']
        res['L_X'] = float(d['j_X'].sum()*dV)

        return res
