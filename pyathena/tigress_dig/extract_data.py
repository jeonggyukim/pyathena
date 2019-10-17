# extract_data.py

import numpy as np
import matplotlib.pyplot as plt

import pyathena as pa
from ..load_sim import LoadSim


class ExtractData:

    @LoadSim.Decorators.check_pickle
    def read_EM_pdf(self, num, savdir=None, force_override=False):

        ds = self.load_vtk(num)
        nH = ds.get_field(field='density')
        xn = ds.get_field(field='specific_scalar[0]')
        nesq = ((1.0 - xn)*nH)**2

        z2 = 200.0
        
        bins = np.linspace(-8, 5, 100)
        dz = ds.domain['dx'][0]
        id0 = 0
        id1 = ds.domain['Nx'][2] // 2
        
        # Calculate EM integrated from z = 200pc
        id2 = id1 + int(z2/dz)

        EM0 = nesq[id0:,:,:].sum(axis=0)*dz
        EM1 = nesq[id1:,:,:].sum(axis=0)*dz
        EM2 = nesq[id2:,:,:].sum(axis=0)*dz

        h0, b0, _ = plt.hist(np.log10(EM0.flatten()), bins=bins, histtype='step', color='C0');
        h1, b1, _ = plt.hist(np.log10(EM1.flatten()), bins=bins, histtype='step', color='C1');
        h2, b2, _ = plt.hist(np.log10(EM2.flatten()), bins=bins, histtype='step', color='C2');

        return dict(EM0=EM0, EM1=EM1, EM2=EM2, bins=bins, h0=h0, h1=h1, h2=h2)
