import os
import os.path as osp
import pandas as pd
import xarray as xr
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
        if self.test_newcool(): self.test_newcool_params()

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

    def test_newcool_params(self):
        s = self
        try:
            s.iCoolH2colldiss = s.par['cooling']['iCoolH2colldiss']
        except KeyError:
            s.iCoolH2colldiss = 0

        try:
            s.iCoolH2rovib = s.par['cooling']['iCoolH2rovib']
        except KeyError:
            s.iCoolH2rovib = 0

        try:
            s.ikgr_H2 = s.par['cooling']['ikgr_H2']
        except KeyError:
            s.ikgr_H2 = 0

        s.config_time = pd.to_datetime(s.par['configure']['config_date'])
        if 'PDT' in s.par['configure']['config_date']:
            config_time = config_time.tz_localize('US/Pacific')
        if s.config_time < pd.to_datetime('2021-06-30 20:29:36 -04:00'):
            s.iCoolHIcollion = 0
        else:
            s.iCoolHIcollion = 1

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
        T1data = np.clip(T1.data,10,None)
        temp = nH/nH*coolftn().get_temp(T1data)
        cool = nH*nH*coolftn().get_cool(T1data)
        heat = heat_ratio*nH*np.clip(coolftn().get_heat(T1data),0.0,None)
        net_cool = cool-heat
        dd['T'] = temp
        dd['cool_rate'] = cool
        dd['heat_rate'] = heat
        dd['net_cool_rate'] = net_cool

        return dd

    def get_savdir_pdf(self,zrange=None):
        '''return joint pdf savdir
        '''
        if zrange is None:
            zmin,zmax = 0,self.domain['re'][2]
        else:
            zmin,zmax = zrange.start, zrange.stop
            if zmin < 0: zmin = 0
        savdir = '{}/jointpdf_z{:02d}-{:02d}/cooling_heating/'.format(self.savdir,int(zmin/100),int(zmax/100))
        return savdir

    def get_coolheat_pdf(self,num,zrange=None,xHI=False):
        '''return pdf from netcdf file

        ==========
        Parameters
        ==========

        xHI : bool
            return T-xHI pdfs if true else nH-T pdfs by default
        '''
        savdir = self.get_savdir_pdf(zrange=zrange)
        if not os.path.isdir(savdir): os.makedirs(savdir)
        fcool=os.path.join(savdir,'{}.{:04d}.cool.pdf.nc'.format(self.problem_id,num))
        fheat=os.path.join(savdir,'{}.{:04d}.heat.pdf.nc'.format(self.problem_id,num))
        if xHI:
            fcool=os.path.join(savdir,'{}.{:04d}.cool.xHI.pdf.nc'.format(self.problem_id,num))
            fheat=os.path.join(savdir,'{}.{:04d}.heat.xHI.pdf.nc'.format(self.problem_id,num))
        if not (os.path.isfile(fcool) and os.path.isfile(fheat)):
            return

        with xr.open_dataset(fcool) as pdf_cool:
            pdf_cool.load()
        with xr.open_dataset(fheat) as pdf_heat:
            pdf_heat.load()
        return pdf_cool, pdf_heat

    def get_merge_jointpdfs(self,zrange=None,force_override=False):
        savdir = self.get_savdir_pdf(zrange=zrange)
        merged_fname = os.path.join(savdir,'jointpdf_all.nc')
        if os.path.isfile(merged_fname) and (not force_override):
            with xr.open_dataset(merged_fname) as pdf:
                pdf.load()
            return pdf

        pdf = []
        for num in self.nums:
            pdfs = self.get_coolheat_pdf(num,zrange=zrange)
            if pdfs is not None:
                print(num, end=' ')
                pdf_cool, pdf_heat = pdfs
                if 'OIold' in pdf_cool:
                    pdf_cool = pdf_cool.drop_vars('OIold')
                pdf_cool = pdf_cool.rename(total='total_cooling')*pdf_cool.attrs['total_cooling']
                pdf_heat = pdf_heat.rename(total='total_heating')*pdf_heat.attrs['total_heating']
                pdf_cool.update(pdf_heat)
                if not ('time' in pdf_cool):
                    ds = s.load_vtk(num)
                    pdf_cool = pdf_cool.assign_coords(time=ds.domain['time'])
                pdf_cool = pdf_cool.assign_coords(cool = pdf_cool.attrs['total_cooling'],
                                                heat = pdf_cool.attrs['total_heating'],
                                                netcool = pdf_cool.attrs['total_netcool'])
                pdf.append(pdf_cool)
        pdf = xr.concat(pdf,dim='time')
        pdf.to_netcdf(merged_fname)
        pdf.close()

        return pdf

    @staticmethod
    def get_phase_Tlist():
        return [500,6000,15000,35000,5.e5]

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
