from .load_sim_tigress_ncr import LoadSimTIGRESSNCR,LoadSimTIGRESSNCRAll
import pyathena as pa
import numpy as np
import pandas as pd
import xarray as xr
import astropy.constants as ac
import astropy.units as au
import scipy
import os
import cmasher as cmr
import matplotlib.pyplot as plt

from .phase import get_phcolor_dict

class PaperData(object):
    def __init__(self):
        self.outdir='/tigress/changgoo/public_html/TIGRESS-NCR/figures/'
        basedir = '/tigress/changgoo/TIGRESS-NCR/'
        base_lowZ = '/tigress/changgoo/TIGRESS-NCR-lowZ/'
        models_full = dict(R8_8pc=basedir+'R8_8pc_NCR.full.B1',
                           LGR4_8pc=basedir+'LGR4_8pc_NCR.full',
                           R8_4pc=basedir+'R8_4pc_NCR.full',
                           LGR4_4pc=basedir+'LGR4_4pc_NCR.full',
                           LGR4_2pc=basedir+'LGR4_2pc_NCR.full',
                           )
        models_fb = dict(R8_UV=basedir+'R8_8pc_NCR.UV/',
                      R8_SN=basedir+'R8_8pc_NCR.SN/',
                      R8_classic=basedir+'R8_8pc_NCR.classic/',
                      LGR4_UV=basedir+'LGR4_4pc_NCR.UV/',
                      LGR4_SN=basedir+'LGR4_4pc_NCR.SN/',
                      LGR4_classic=basedir+'LGR4_4pc_NCR.classic/',
                    )
        models_full2 = dict(R8_8pc2=basedir+'R8_8pc_NCR.full.xy2048.eps0.0',
              # R8_4pc2=basedir+'R8_4pc_NCR.full',
              LGR4_4pc2=basedir+'LGR4_4pc_NCR.full.xy1024.eps1.e-8',
              # LGR4_2pc2=basedir+'LGR4_2pc_NCR.full',
             )
        models_fb2 = dict(R8_UV2=basedir+'R8_8pc_NCR.UV.v2/',
                      R8_SN2=basedir+'R8_8pc_NCR.SN.v2/',
                      R8_classic2=basedir+'R8_8pc_NCR.classic.v2/',
                      # LGR4_UV2=basedir+'LGR4_4pc_NCR.UV.v2/',
                      # LGR4_SN2=basedir+'LGR4_4pc_NCR.SN.v2/',
                      # LGR4_classic2=basedir+'LGR4_4pc_NCR.classic.v2/',
                    )
        models_mag = dict(R8_b10=basedir+'R8_8pc_NCR.full/',
                      R8_UVb10=basedir+'R8_8pc_NCR.UV.b10/',
                      R8_UV=basedir+'R8_8pc_NCR.UV/',
                      R8_norot=basedir+'R8_8pc_NCR.full.norot.b10/',
                      R8_nosh=basedir+'R8_8pc_NCR.full.q0.b10/',
                      R8_hydro=basedir+'R8_8pc_NCR.full.hd/',
                    )
        models_lowZ = dict(R8_Z03=base_lowZ+'R8_8pc_NCR.full.Z0.3.xy4096.eps0.0/',
                           R8_Z01=base_lowZ+'R8_8pc_NCR.full.Z0.1.xy4096.eps0.0/')
        models_norun = dict(R8_norun=basedir+'R8_8pc_NCR.full.norun/',
                            LGR4_norun=basedir+'LGR4_4pc_NCR.full.norun/',
                            )
        models=dict()
        models.update(models_full)
        models.update(models_fb)
        models.update(models_mag)
        models.update(models_full2)
        models.update(models_fb2)
        models.update(models_lowZ)
        models.update(models_norun)
        self.sa = LoadSimTIGRESSNCRAll(models)
        self.models = dict(full = models_full, fb = models_fb, mag = models_mag,
                           full2 = models_full2, fb2 = models_fb2, lowZ = models_lowZ,
                           norun = models_norun
                           )
        self._set_colors()
        self._set_plot_kwargs()
        self.set_model_list()

    def _set_colors(self):
        import cmasher as cmr

        cmap1 = cmr.get_sub_cmap('cmr.pride', 0.1, 0.45, N=3)
        cmap2 = cmr.get_sub_cmap('cmr.pride_r', 0.1, 0.45, N=3)

        colors=dict()

        for pre,cmap in zip(['R8','LGR4'],[cmap1,cmap2]):
            for m,c in zip(['classic','UV','SN'],cmap.colors):
                colors['_'.join([pre,m])] = c

        for m,c in zip(['Z03','Z01'],cmap2.colors[::-1]):
            colors['_'.join(['R8',m])] = c

        colors['R8_8pc']='tab:cyan'
        colors['R8_4pc']='tab:blue'
        colors['LGR4_8pc']='tab:pink'
        colors['LGR4_4pc']='tab:red'
        colors['LGR4_2pc']='tab:purple'
        colors['R8_full']='tab:cyan'
        colors['R8_b10']='gold'
        colors['R8_UVb10']='tab:olive'
        colors['R8_nosh']='black'
        colors['R8_norot']='tab:gray'
        colors['R8_hydro']='tab:orange'

        for m in self.models['full2']:
            colors[m] = colors[m[:-1]]
        for m in self.models['fb2']:
            colors[m] = colors[m[:-1]]
        self.colors = colors

    def _set_plot_kwargs(self):
        plt_kwargs=dict()
        #initialize
        for m in self.sa.models:
            if not m in self.colors:
                plt_kwargs[m]=dict()
                continue
            plt_kwargs[m]=dict(color=self.colors[m])
            if m.endswith('2'):
                plt_kwargs[m].update(dict(lw=3,alpha=0.5))
        plt_kwargs['LGR4_8pc'].update(dict(lw=1,alpha=0.5))
        self.plt_kwargs = plt_kwargs

    def set_model_list(self):
        mlist = dict(res = ['R8_4pc','R8_8pc','LGR4_2pc','LGR4_4pc','LGR4_8pc'],
                     std = ['R8_4pc','R8_8pc','LGR4_2pc','LGR4_4pc'],
                     fbR8 = ['R8_8pc','R8_classic','R8_UV','R8_SN'],
                     fbLGR4 = ['LGR4_4pc','LGR4_classic','LGR4_UV','LGR4_SN'],
                     mag = list(self.models['mag'].keys()),
                     oldnew = ['R8_8pc','R8_8pc2','LGR4_4pc','LGR4_4pc2'],
                     fbR8new = ['R8_8pc2','R8_classic2','R8_UV2','R8_SN2'],
                     lowZ = ['R8_8pc2','R8_Z03','R8_Z01'],
                     norun = ['R8_norun','LGR4_norun']
                     )
        trlist = dict(res = [slice(250,450)]*2+[slice(250,350)]*3,
                      std = [slice(250,450)]*2+[slice(250,350)]*2,
                      fbR8 = [slice(400,600)]*len(mlist['fbR8']),
                      fbLGR4 = [slice(350,450)]*len(mlist['fbLGR4']),
                      mag = [slice(400,600)]*len(mlist['mag']),
                      oldnew = [slice(250,450)]*2+[slice(250,350)]*2,
                      fbR8new = [slice(400,600)]*len(mlist['fbR8new']),
                      lowZ = [slice(300,600)]*len(mlist['lowZ']),
                      norun = [slice(250,450),slice(250,350)]
        )

        mlist['fb'] = mlist['fbR8'] + mlist['fbLGR4']
        trlist['fb'] = trlist['fbR8'] + trlist['fbLGR4']
        restart_from = dict(R8_4pc='R8_8pc',LGR4_2pc='LGR4_4pc',LGR4_4pc='LGR4_8pc')
        restart_from = dict(R8_norun='R8_8pc',LGR4_norun=]
        restart_from.update({m:'R8_8pc' for m in mlist['fbR8'][1:]})
        restart_from.update({m:'LGR4_4pc' for m in mlist['fbLGR4'][1:]})
        restart_from.update({m:'R8_8pc2' for m in mlist['fbR8new'][1:]})
        self.mlist = mlist
        self.trlist = trlist
        self.restart_from = restart_from

    def show_global_history(self,m):
        h=self.set_global_history(m,recal=False)
        if len(plt.gcf().axes) != 12:
            axes = None
        else:
            axes = plt.gcf().axes
        axes = self.plot_basics(h,name=m,axes=axes)

    def set_global_history(self,m,recal=False):
        s = self.sa.set_model(m)
        if hasattr(s,'h') and not recal: return s.h
        h = pa.read_hst(s.files['hst'])
        if not hasattr(s,'zpw'):
            s.zpw = s.read_zprof(phase='whole')

        if s.test_phase_sep_hst():
            hw = s.read_hst_phase(iph=0)

        vol = np.prod(s.domain['Lx'])
        area = vol/s.domain['Lx'][2]
        nscal = s.par['configure']['nscalars']

        Sigma_gas = h['mass']*s.u.Msun*vol/area
        Sigma_star = h['msp']*s.u.Msun*vol/area

        time = h['time']
        if s.test_phase_sep_hst():
            mass_out = (hw["Fzm_upper_dt"] - hw["Fzm_lower_dt"])
        else:
            mass_out = scipy.integrate.cumtrapz(h['F3_upper'] - h['F3_lower'], h['time'], initial=0.0)
        Sigma_out = (mass_out)/(s.domain['Lx'][2])*vol*s.u.Msun/area
        Sigma_H2 = 2.0*h['scalar{}'.format(nscal-2)]*s.u.Msun*vol/area
        Sigma_HI = h['scalar{}'.format(nscal-3)]*s.u.Msun*vol/area
        Sigma_HII = Sigma_gas-Sigma_HI-Sigma_H2

        sfr10 = h['sfr10']
        sfr40 = h['sfr40']
        sfr100 = h['sfr100']
        sfr = sfr10
        if s.test_phase_sep_hst():
            H = np.sqrt(hw['H2']/hw['mass'])
            P = hw['P']
        else:
            H = np.sqrt(h['H2']/h['mass'])
            P = h['P']
        Pimag = 0.
        if 'x1ME' in h:
            Pimag = h['x1ME']+h['x2ME']-2.0*h['x3ME']
            vA = np.sqrt(2.0*(h['x1ME']+h['x2ME']+h['x3ME'])/h['mass'])
        szeff = np.sqrt((2.0*h['x3KE'] + P + Pimag)/h['mass'])
        vz = np.sqrt((2.0*h['x3KE'])/h['mass'])
        tver = H/vz*s.u.Myr
        tMyr = h['time']*s.u.Myr
        torb = 2*np.pi/s.par['problem']['Omega']*s.u.Myr
        tdep = Sigma_gas/sfr/1.e3
        tdep10 = Sigma_gas/sfr10/1.e3
        tdep40 = Sigma_gas/sfr40/1.e3
        tdep100 = Sigma_gas/sfr100/1.e3
        # basic quantities
        h1 = dict(Sigma_gas=Sigma_gas,Sigma_star=Sigma_star,Sigma_out=Sigma_out,
                     Sigma_H2=Sigma_H2,Sigma_HI=Sigma_HI,Sigma_HII=Sigma_HII,
                     sfr=sfr, sfr10=sfr10, sfr40=sfr40, sfr100=sfr100,
                     H=H, szeff=szeff, vz=vz,
                     tdep=tdep, tdep10=tdep10, tdep40=tdep40, tdep100=tdep100,
                     tver=tver, tMyr=tMyr, torb=time/torb, time=time)
        if 'x1ME' in h: h1.update(dict(vA=vA))

        if not s.test_newcool():
            s.h=pd.DataFrame(h1)
            return s.h

        # energy gain/loss
        ifreq = dict()
        for f in ('PH','LW','PE'): #,'PE_unatt'):
            try:
                ifreq[f] = s.par['radps']['ifreq_{0:s}'.format(f)]
            except KeyError:
                pass
        for i in range(s.par['radps']['nfreq']):
            for k, v in ifreq.items():
                if i == v:
                    factor = s.u.Lsun if s.test_phase_sep_hst() else vol*s.u.Lsun
                    h[f'Ltot_{k}'] = h[f'Ltot{i}']*factor
        # injected radiation
        SLyC = h[f'Ltot_PH']/area
        SPE = h[f'Ltot_PE']/area
        SLW = h[f'Ltot_LW']/area
        Srad=SLyC+SPE+SLW

        # injected SN energy
        sn = pa.read_hst(s.files['sn'])
        Nsn,tbin = np.histogram(sn['time']*s.u.Myr,bins=tMyr)
        dt = np.diff(tbin)
        SSN = np.concatenate([[0],(1.e51*au.erg/au.Myr).to('Lsun').value*Nsn/dt/area])
        SSN = pd.Series(SSN,h.index)

        # total radiative heating rate
        zp = s.zpw
        factor = (s.u.energy/s.u.time).to('Lsun') if s.test_phase_sep_hst() else (au.pc**3*au.erg/au.cm**3/au.s).to('Lsun')
        Sheat=np.interp(tMyr, zp.time, zp['heat'].sum(dim='z')*(s.domain['dx'][2]*factor))
        Scool=np.interp(tMyr, zp.time, zp['cool'].sum(dim='z')*(s.domain['dx'][2]*factor))

        # take rolling means
        h2 = dict(SLyC=SLyC, SPE=SPE, SLW=SLW, Srad=Srad,
                  SSN=SSN, Sheat=Sheat, Scool=Scool, Snet=Scool-Sheat)
        h1.update(h2)

        s.h=pd.DataFrame(h1)
        return s.h

    @staticmethod
    def zprof_rename(s):
        if hasattr(s,'newzp'): return s.newzp
        rename_dict = dict()
        kind = 'new' if s.test_phase_sep_hst() else 'old'
        shorthands = s.get_phase_shorthand()
        for i,pname in enumerate(shorthands):
            rename_dict['phase{}'.format(i+1)] = pname
        if not hasattr(s,'zp'):
            zp = s.read_zprof_new()
        else:
            if not 'phase' in s.zp:
                zp = s.read_zprof_new()
        zp = s.zp
        zp = zp.to_array().to_dataset('phase').rename(rename_dict)
        zp = zp.to_array('phase').to_dataset('variable')

        newzp = xr.Dataset()
        if not s.test_newcool():
            newzp['CNM'] = zp.sel(phase='c').squeeze().to_array()
            newzp['UNM'] = zp.sel(phase='u').squeeze().to_array()
            newzp['WNM'] = zp.sel(phase='w').squeeze().to_array()
            newzp['WHIM'] = zp.sel(phase='h1').squeeze().to_array()
            newzp['HIM'] = zp.sel(phase='h2').squeeze().to_array()
        else:
            if kind == 'old':
                newzp['CMM'] = zp.sel(phase='mol').squeeze().to_array()
                newzp['WIM'] = zp.sel(phase='pi').squeeze().to_array()
                newzp['CNM'] = zp.sel(phase=['HIcc','HIc']).sum(dim='phase').to_array()
                newzp['UNM'] = zp.sel(phase=['HIu','HIuc']).sum(dim='phase').to_array()
                newzp['WNM'] = zp.sel(phase=['HIw','HIwh']).sum(dim='phase').to_array()
                newzp['WHIM'] = zp.sel(phase='h1').squeeze().to_array()
                newzp['HIM'] = zp.sel(phase='h2').squeeze().to_array()
            elif kind == 'new':
                newzp['CMM'] = zp.sel(phase='cmm').squeeze().to_array()
        #         newzp['UIM'] = zp.sel(phase='uim').squeeze().to_array()
                newzp['WIM'] = zp.sel(phase=['uim','wpim','wcim']).sum(dim='phase').to_array()
                newzp['CNM'] = zp.sel(phase='cnm').squeeze().to_array()
                newzp['UNM'] = zp.sel(phase='unm').squeeze().to_array()
                newzp['WNM'] = zp.sel(phase='wnm').squeeze().to_array()
                newzp['WHIM'] = zp.sel(phase='h1').squeeze().to_array()
                newzp['HIM'] = zp.sel(phase='h2').squeeze().to_array()
    #         newzp['Others'] = (zp.sel(phase=['hotnothers']).squeeze()-zp.sel(phase=['h1','h2']).sum(dim='phase')).to_array()
        s.newzp = newzp.to_array('phase').to_dataset('variable')
        return s.newzp

    def Pmid_time_series(self,m,sfr=None,dt=0,zrange=slice(-10,10),
                         from_files=True,recal=False):
        s = self.sa.set_model(m)
        if from_files:
            fzpmid=os.path.join(s.basedir,'zprof','{}.zpmid.nc'.format(s.problem_id))
            fzpw=os.path.join(s.basedir,'zprof','{}.zpwmid.nc'.format(s.problem_id))
            if os.path.isfile(fzpmid):
                with xr.open_dataset(fzpmid) as zpmid:
                    s.zpmid = self.set_zprof_attr(zpmid)
            if os.path.isfile(fzpw):
                with xr.open_dataset(fzpw) as zpwmid: s.zpwmid = zpwmid

        if hasattr(s,'zpmid') and hasattr(s,'zpwmid') and (not recal):
            # smoothing
            if dt>0:
                window = int(dt/s.zpmid.time.diff(dim='time').median())
                s.zpmid = s.zpmid.rolling(time=window,center=True,min_periods=1).mean()
            return s.zpmid, s.zpwmid

        if hasattr(s,'newzp'):
            zp = s.newzp
        else:
            zp = self.zprof_rename(s)

        zpmid = xr.Dataset()
        dm = s.domain
        dz = dm['dx'][2]
        # calculate weight first
        uWext=(zp['dWext'].sel(z=slice(0,dm['re'][2]))[:,::-1].cumsum(dim='z')[:,::-1]*dz)
        lWext=(-zp['dWext'].sel(z=slice(dm['le'][2],0)).cumsum(dim='z')*dz)
        Wext=xr.concat([lWext,uWext],dim='z')
        if 'dWsg' in zp:
            uWsg=(zp['dWsg'].sel(z=slice(0,dm['re'][2]))[:,::-1].cumsum(dim='z')[:,::-1]*dz)
            lWsg=(-zp['dWsg'].sel(z=slice(dm['le'][2],0)).cumsum(dim='z')*dz)
            Wsg=xr.concat([lWsg,uWsg],dim='z')
        W=Wext+Wsg
        zpmid['Wext']=Wext
        zpmid['Wsg']=Wsg
        zpmid['W']=W

        # caculate radiation pressure
        if 'frad_z0' in zp:
            frad_list=['frad_z0','frad_z1','frad_z2']
            for frad in frad_list:
                uPrad=zp[frad].sel(z=slice(0,dm['re'][2]))[:,::-1].cumsum(dim='z')[:,::-1]*dz
                lPrad=-zp[frad].sel(z=slice(dm['le'][2],0)).cumsum(dim='z')*dz
                zpmid[frad]=xr.concat([lPrad,uPrad],dim='z')
            zpmid['Prad']=zpmid[frad_list].to_array().sum(dim='variable')

        # Pressures/Stresses
        zpmid['Pth'] = zp['P']
        zpmid['Pturb'] = 2.0*zp['Ek3']
        zpmid['Ptot'] = zpmid['Pth']+zpmid['Pturb']
        if 'PB1' in zp:
            zpmid['Pmag'] = zp['PB1']+zp['PB2']+zp['PB3']
            zpmid['Pimag'] = zpmid['Pmag'] - 2.0*zp['PB3']
            zpmid['dPmag'] = zp['dPB1']+zp['dPB2']+zp['dPB3']
            zpmid['dPimag'] = zpmid['dPmag'] - 2.0*zp['dPB3']
            zpmid['oPmag'] = zpmid['Pmag']-zpmid['dPmag']
            zpmid['oPimag'] = zpmid['Pimag']-zpmid['dPimag']
            zpmid['Ptot'] += zpmid['Pimag']

        # density, area
        zpmid['nH'] = zp['d']
        zpmid['A'] = zp['A']

        # heat and cool
        if 'cool' in zp:
            zpmid['heat']= zp['heat']
            zpmid['cool']= zp['cool']
            if 'netcool' in zp:
                zpmid['net_cool']= zp['net_cool']
            else:
                zpmid['net_cool']= zp['cool'] - zp['heat']

        # Erad
        if 'Erad0' in zp: zpmid['Erad0']= zp['Erad0']
        if 'Erad1' in zp: zpmid['Erad1']= zp['Erad1']
        if 'Erad2' in zp: zpmid['Erad2']= zp['Erad2']

        # select midplane
        zpmid = zpmid.sel(z=zrange).mean(dim='z')
        zpwmid = zpmid.sum(dim='phase')

        # rearrange phases
        twop=zpmid.sel(phase=['CMM','CNM','UNM','WNM']).sum(dim='phase').assign_coords(phase='2p')
        hot=zpmid.sel(phase=['WHIM','HIM']).sum(dim='phase').assign_coords(phase='hot')
        if 'WIM' in zpmid.phase:
            wim=zpmid.sel(phase=['WIM']).sum(dim='phase').assign_coords(phase='WIM')
            zpmid = xr.concat([twop,wim,hot],dim='phase')
        else:
            zpmid = xr.concat([twop,hot],dim='phase')
    #     zpw=zpmid.sel(phase=['whole']).squeeze()

        # SFR from history
        vol = np.prod(s.domain['Lx'])
        area = vol/s.domain['Lx'][2]

        h=pa.read_hst(s.files['hst'])
        if sfr is None:
            zpmid['sfr'] = xr.DataArray(np.interp(zpmid.time_code,h['time'],h['sfr10']),coords=[zpmid.time])
        else:
            zpmid['sfr'] = sfr

        # sz from history
        if 'x1ME' in h:
            szmag = h['x1ME']+h['x2ME']-2.0*h['x3ME']
        else:
            szmag = 0.0
        if s.test_phase_sep_hst():
            hw = s.read_hst_phase()
            P = hw['P']
        else:
            P = h['P']
        szeff = np.sqrt((2.0*h['x3KE']+P+szmag)/h['mass'])
        zpmid['szeff'] = xr.DataArray(np.interp(zpmid.time_code,h['time'],szeff),coords=[zpmid.time])

        # PDE
        zpmid['sigma_eff'] = np.sqrt(zpmid['Ptot']/zpmid['nH'])
        zpmid['Sigma_gas'] = xr.DataArray(np.interp(zp.time_code,h['time'],h['mass']*s.u.Msun*vol/area),coords=[zpmid.time])
        rhosd=0.5*s.par['problem']['SurfS']/s.par['problem']['zstar']+s.par['problem']['rhodm']
        zpmid['PDE1'] = np.pi*zpmid['Sigma_gas']**2/2.0*(ac.G*(ac.M_sun/ac.pc**2)**2/ac.k_B).cgs.value
        zpmid['PDE2_2p'] = zpmid['Sigma_gas']*np.sqrt(2*rhosd)*zpmid['sigma_eff'].sel(phase='2p')*(np.sqrt(ac.G*ac.M_sun/ac.pc**3)*(ac.M_sun/ac.pc**2)*au.km/au.s/ac.k_B).cgs.value
        zpmid['PDE2'] = zpmid['Sigma_gas']*np.sqrt(2*rhosd)*zpmid['szeff']*(np.sqrt(ac.G*ac.M_sun/ac.pc**3)*(ac.M_sun/ac.pc**2)*au.km/au.s/ac.k_B).cgs.value
        zpmid['PDE_2p'] = zpmid['PDE1']+zpmid['PDE2_2p']
        zpmid['PDE'] = zpmid['PDE1']+zpmid['PDE2']

        fzpmid=os.path.join(s.basedir,'zprof','{}.zpmid.nc'.format(s.problem_id))
        fzpw=os.path.join(s.basedir,'zprof','{}.zpwmid.nc'.format(s.problem_id))
        if os.path.isfile(fzpmid): os.remove(fzpmid)
        if os.path.isfile(fzpw): os.remove(fzpw)

        zpmid.to_netcdf(fzpmid)
        zpwmid.to_netcdf(fzpw)

        # smoothing
        if dt>0:
            window = int(dt/zpmid.time.diff(dim='time').median())
            zpmid = zpmid.rolling(time=window,center=True,min_periods=1).mean()

        s.zpmid = self.set_zprof_attr(zpmid)
        s.zpwmid = zpwmid
        return zpmid,zpwmid

    @staticmethod
    def set_zprof_attr(zpmid):
        # misc.
        phcolors=get_phcolor_dict(cmr.pride,cmin=0.1,cmax=0.8)
        phcolors['WIM']=phcolors['WPIM']
        colors = {'2p':phcolors['CMM'],'WIM':phcolors['WNM'],'hot':phcolors['WHIM']}
        labels = {'2p':'2p','WIM':'WIM','hot':'hot'}
        Plabels = {'Ptot':r'$P_{\rm tot}$','Pth':r'$P_{\rm th}$',
                  'Pturb':r'$P_{\rm turb}$','Pimag':r'$\Pi_{\rm mag}$',
                  'dPimag':r'$\Pi_{\delta B}$','oPimag':r'$\Pi_{\overline{B}}$',
                  'W':r'$W_{\rm tot}$','Wext':r'$W_{\rm ext}$','Wsg':r'$W_{\rm sg}$',
                  'PDE':r'$P_{\rm DE}$','Prad':r'$P_{\rm rad}$'}
        Pcolors = {'Ptot':'k','Pth':'tab:blue','Pturb':'tab:orange','Pimag':'tab:green',
                  'dPimag':'gold','oPimag':'tab:green','W':'k'}
        Ulabels = {'Ptot':r'$\Upsilon_{\rm tot}$','Pth':r'$\Upsilon_{\rm th}$',
                  'Pturb':r'$\Upsilon_{\rm turb}$','Pimag':r'$\Upsilon_{\rm mag}$',
                  'dPimag':r'$\Upsilon_{\delta B}$','oPimag':r'$\Upsilon_{\overline{B}}$'}
        zpmid.attrs['colors']=colors
        zpmid.attrs['labels']=labels
        zpmid.attrs['Plabels']=Plabels
        zpmid.attrs['Ulabels']=Ulabels
        zpmid.attrs['Pcolors']=Pcolors

        return zpmid

    @staticmethod
    def plot_basics(h,name='model',axes=None):
        if axes is None:
            fig,axes = plt.subplots(4,3,figsize=(10,10),sharey='row')
            axes = axes.flatten()
        axes_it = iter(axes)
        fields = ['Sigma_gas','Sigma_star','Sigma_out',
                'Sigma_HI','Sigma_HII','Sigma_H2',
                'sfr10','sfr40','sfr100',
                'szeff','vz','vA']

        for f in fields:
            plt.sca(next(axes_it))
            plt.plot(h['time'],h[f],label=name)
            plt.ylabel(f)
        plt.tight_layout()
        return axes