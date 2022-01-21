import os
import os.path as osp
import pandas as pd
import numpy as np
import astropy.constants as ac
import astropy.units as au

from ..load_sim import LoadSim
from ..util.units import Units
from ..io.read_hst import read_hst
from ..classic.cooling import coolftn
from .pdf import PDF
from .h2 import H2
from .hst import Hst
from .zprof import Zprof
from .slc_prj import SliceProj
from .starpar import StarPar
from .snapshot_HIH2EM import Snapshot_HIH2EM
from .profile_1d import Profile1D

class LoadSimTIGRESSNCR(LoadSim, Hst, Zprof, SliceProj,
                        StarPar, PDF, H2, Profile1D, Snapshot_HIH2EM):
    """LoadSim class for analyzing TIGRESS-RT simulations.
    """

    def __init__(self, basedir, savdir=None, load_method='pyathena',
                 muH = 1.4271, verbose=False):
        """The constructor for LoadSimTIGRESSNCR class

        Parameters
        ----------
        basedir : str
            Name of the directory where all data is stored
        savdir : str
            Name of the directory where pickled data and figures will be saved.
            Default value is basedir.
        load_method : str
            Load vtk using 'pyathena' or 'yt'. Default value is 'pyathena'.
            If None, savdir=basedir. Default value is None.
        verbose : bool or str or int
            Print verbose messages using logger. If True/False, set logger
            level to 'DEBUG'/'WARNING'. If string, it should be one of the string
            representation of python logging package:
            ('NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            Numerical values from 0 ('NOTSET') to 50 ('CRITICAL') are also
            accepted.
        """

        super(LoadSimTIGRESSNCR,self).__init__(basedir, savdir=savdir,
                                               load_method=load_method, verbose=verbose)

        # Set unit and domain
        try:
            muH = self.par['problem']['muH']
        except KeyError:
            pass
        self.muH = muH
        self.u = Units(muH=muH)
        self.domain = self._get_domain_from_par(self.par)

    def test_newcool(self):
        try:
            if self.par['configure']['new_cooling'] == 'ON':
                newcool = True
            else:
                newcool = False
        except KeyError:
            newcool = False
        return newcool

    def test_spiralarm(self):
        try:
            if self.par['configure']['SpiralArm'] == 'yes':
                arm = True
            else:
                arm = False
        except KeyError:
            arm = False
        return arm

    def show_timeit(self):
        import matplotlib.pyplot as plt
        try:
            time = pd.read_csv(self.files['timeit'],delim_whitespace=True)

            tfields = [k.split('_')[0] for k in time.keys() if k.endswith('tot')]

            for tf in tfields:
                if tf == 'rayt': continue
                plt.plot(time['time'],time[tf].cumsum()/time['all'].cumsum(),label=tf)
            plt.legend()
        except KeyError:
            print("No timeit plot is available")

    def get_timeit_mean(self):
        try:
            time = pd.read_csv(self.files['timeit'],delim_whitespace=True)

            tfields = [k.split('_')[0] for k in time.keys() if k.endswith('tot')]

            return time[tfields].mean()
        except:
            print("No timeit file is available")

    def get_classic_cooling_rate(self, ds):
        if (not hasattr(self,'heat_ratio')):
            hst = read_hst(self.files['hst'])
            self.heat_ratio = hst['heat_ratio']
            self.heat_ratio.index = hst['time']
        dd = ds.get_field(['density','pressure'])
        nH = dd['density']
        heat_ratio = np.interp(ds.domain['time'],self.heat_ratio.index,self.heat_ratio)
        T1 = dd['pressure']/dd['density']
        T1 *= (self.u.velocity**2*ac.m_p/ac.k_B).cgs.value
        T1data = T1.data
        temp = nH/nH*coolftn().get_temp(T1data)
        cool = nH*nH*coolftn().get_cool(T1data)
        heat = heat_ratio*nH*coolftn().get_heat(T1data)
        net_cool = cool-heat
        dd['T'] = temp
        dd['cool_rate'] = cool
        dd['heat_rate'] = heat
        dd['net_cool_rate'] = net_cool

        return dd




class LoadSimTIGRESSNCRAll(object):
    """Class to load multiple simulations"""
    def __init__(self, models=None, muH=None):

        # Default models
        if models is None:
            models = dict()
        if muH is None:
            muH = dict()
            for mdl in models:
                muH[mdl] = 1.4271
        self.models = []
        self.basedirs = dict()
        self.muH = dict()
        self.simdict = dict()

        for mdl, basedir in models.items():
            if not osp.exists(basedir):
                print('[LoadSimTIGRESSNCRAll]: Model {0:s} doesn\'t exist: {1:s}'.format(
                    mdl,basedir))
            else:
                self.models.append(mdl)
                self.basedirs[mdl] = basedir
                if mdl in muH:
                    self.muH[mdl] = muH[mdl]
                else:
                    print('[LoadSimTIGRESSNCRAll]: muH for {0:s} has to be set'.format(
                          mdl))

    def set_model(self, model, savdir=None, load_method='pyathena', verbose=False):
        self.model = model
        try:
            self.sim = self.simdict[model]
        except KeyError:
            self.sim = LoadSimTIGRESSNCR(self.basedirs[model], savdir=savdir,
                                         muH=self.muH[model],
                                         load_method=load_method, verbose=verbose)
            self.simdict[model] = self.sim

        return self.sim

    # adding two objects
    def __add__(self, o):
        for mdl in o.models:
            if not (mdl in self.models):
                self.models += [mdl]
                self.basedirs[mdl] = o.basedirs[mdl]
                self.muH[mdl] = o.muH[mdl]
                if mdl in o.simdict: self.simdict[mdl] = o.simdict[mdl]

        return self

    # get self class with only one key
    def __getitem__(self, key):
        return self.set_model(key)

    def __setitem__(self, key, value):
        if (type(value) == LoadSimTIGRESSNCR):
            self.models.append(key)
            self.simdict[key] = value
            self.basedirs[key] = value.basedir
            self.muH[key] = value.muH
        else:
            print("Assigment only accepts LoadSimTIGRESSNCR")
