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
from .ncr_papers import PaperData
from ..plt_tools.utils import texteffect

import cmasher as cmr

class LowZData(PaperData):
    def __init__(self):
        self.outdir='/tigress/changgoo/public_html/TIGRESS-NCR/lowZ-figures/'
        models, mlist_early = self._get_models('/scratch/gpfs/changgoo/TIGRESS-NCR/')
        self.sa = pa.LoadSimTIGRESSNCRAll(models)
        self.mlist_all = list(models)
        self.mlist = list(mlist_early)
        self.mlist_early = mlist_early
        self._set_model_params()
        self._set_model_list()

        self._set_colors()
        self._set_torb()
        os.makedirs(self.outdir,exist_ok=True)

    def _set_model_list(self):
        mgroup = dict()
        for m in self.mlist:
            if 'R8' in m:
                head = 'R8'
            elif 'LGR4' in m:
                head = 'LGR4'
            elif 'LGR8' in m:
                head = 'LGR8'

            if '.b1.' in m:
                head += '-b1'
            elif '.b10.' in m:
                head += '-b10'

            if 'S30' in m:
                head += '-S30'
            if 'S100' in m:
                head += '-S100'
            if 'S05' in m:
                head += '-S05'
            print(head, m)
            if head in mgroup:
                mgroup[head].append(m)
            else:
                mgroup[head] = [m]
        mgroup['R8']=mgroup['R8-b1']
        mgroup['LGR4']=mgroup['LGR4-b1']

        self.mgroup = mgroup

    def _set_torb(self):
        self.torb = dict()
        self.torb_Myr = dict()
        self.torb_code = dict()
        for m in self.mlist:
            s = self.sa.set_model(m)
            torb = 2*np.pi/s.par['problem']['Omega']
            s.torb_code = torb
            s.torb_Myr = torb*s.u.Myr
            self.torb[m] = torb
            if 'R8' in m:
                self.torb_code['R8'] = torb
                self.torb_Myr['R8'] = torb*s.u.Myr
            elif 'LGR4' in m:
                self.torb_code['LGR4'] = torb
                self.torb_Myr['LGR4'] = torb*s.u.Myr


    @staticmethod
    def stitch_hsts(sa,m1,m2):
        s1 = sa.set_model(m1)
        h1 = s1.read_hst()
        s2 = sa.set_model(m2)
        h2 = s2.read_hst()
        t0 = h2.index[0]
        hnew = h1[:t0].append(h2).fillna(0.0)
        return hnew

    @staticmethod
    def get_model_name(s,beta=True,zonly=False):
        p = s.par['problem']
        head = s.basename.split('_')[0]
        if beta: head += f"-b{int(p['beta'])}"
        if p['Z_gas'] == p['Z_dust']:
            ztail = f"Z{p['Z_gas']:3.1f}"
        else:
            ztail = f"Zg{p['Z_gas']:3.1f}Zd{p['Z_dust']:5.3f}"
        if zonly:
            return ztail
        else:
            return '{}-{}'.format(head,ztail)

    def _set_model_params(self):
        sa = self.sa
        for m in sa.models:
            s = sa.set_model(m)
            pp = s.par['problem']
            prp = s.par['radps']
            s.Zgas = pp['Z_gas']
            s.Zdust = pp['Z_dust']
            s.xymax = prp['xymaxPP']
            s.epspp = prp['eps_extinct']
            s.beta = pp['beta']
            s.name = self.get_model_name(s)

    def _get_models(self,basedir):
        dirs = sorted(os.listdir(basedir))[::-1]
        models = dict()
        for d in dirs:
            if 'v3.iCR4' in d:
                print(d)
                models[d] = os.path.join(basedir,d)
        models['R8_8pc_NCR.full.b10.v3']=os.path.join(basedir,'R8_8pc_NCR.full.b10.v3')

        mlist = list(models)
        mlist_early = dict()
        for m in mlist:
            if 'xy' in m:
                mearly = m[:m.rfind('xy')-1]
                for m1 in mlist:
                    if m1 == mearly:
                        mlist_early[m]=m1
        mlist_early['R8_8pc_NCR.full.b10.v3.iCR4.Zg1.Zd1.xy2048.eps0.0'] = 'R8_8pc_NCR.full.b10.v3'
        return models, mlist_early

    def _set_colors(self):
        colors1 = cmr.take_cmap_colors('cmr.pride', 4, cmap_range=(0.05, 0.45))
        colors2 = cmr.take_cmap_colors('cmr.pride_r', 4, cmap_range=(0.05, 0.45))
        self.plt_kwargs = dict()
        for gr in self.mgroup:
            self.plt_kwargs[gr]=dict()
            if 'R8' in gr:
                colors = colors1
            elif 'LGR4' in gr:
                colors = colors2

            if 'b10' in gr:
                ls = '--'
            else:
                ls = '-'
            self.plt_kwargs[gr]['colors'] = colors
            self.plt_kwargs[gr]['ls'] = ls

            for m in self.mgroup[gr]:
                s = self.sa.set_model(m)
                if s.Zdust == 1.0:
                    s.color = colors[0]
                elif s.Zdust == 0.3:
                    s.color = colors[1]
                elif s.Zdust == 0.1:
                    s.color = colors[2]
                elif s.Zdust == 0.025:
                    s.color = colors[3]
                else:
                    print("{} cannot find matching color".format(m))

                s.ls = ls

    def read_hst(self):
        for m in self.mlist:
            mearly = self.mlist_early[m]
            h = self.stitch_hsts(self.sa,mearly,m)
            s = self.sa.set_model(m)
            s.h = h
            s.h['tdep40'] = s.h['Sigma_gas']/s.h['sfr40']*1.e-3

    def plot_hst(self,m,y='sfr40',full=True):
        s = self.sa.set_model(m)
        if full:
            h = s.h
        else:
            h = s.hst
        name = self.get_model_name(s)
        torb = 2*np.pi/s.par['problem']['Omega']*s.u.Myr
        plt.plot(h['time']/torb,get_smoothed(h[y],h['time'],5),
                 label=name,lw=1,color=s.color,ls=s.ls)

    def collect_hst_list(self,y,tslice=None,group='R8-b10'):
        hlist = []
        namelist = []
        Zlist = []
        for m in self.mgroup[group]:
            s = self.sa.set_model(m)
            namelist.append(s.name)
            if tslice is not None:
                ysel = s.h[y][tslice]
            else:
                ysel = s.h[y]
            hlist.append(ysel)
            Zlist.append(s.Zdust)
        return namelist,hlist,Zlist

    def collect_zpdata(self,m,trange=None,reduce=True,recal=False,
                       func=np.mean,**func_kwargs):
        zpmid,zpwmid = self.get_PW_time_series(m,recal=recal)
        if m in self.mlist_early:
            zpmid_early,zpwmid_early = self.get_PW_time_series(self.mlist_early[m],recal=recal)
            tmax = zpmid.time.min().data*0.999
            tmin = zpmid_early.time.min().data
            zpmid=xr.concat([zpmid_early.sel(time=slice(tmin,tmax)),zpmid],dim='time')
            zpwmid=xr.concat([zpwmid_early.sel(time=slice(tmin,tmax)),zpwmid],dim='time')
        s = self.sa.set_model(m)

        if trange is None:
            trange = slice(s.torb_Myr*2,s.torb_Myr*5)

        zpmid = zpmid.sel(time=trange)
        zpwmid = zpwmid.sel(time=trange)

        ydata = xr.Dataset()

        yield_conv = ((au.cm**(-3)*au.K*ac.k_B)/(ac.M_sun/ac.kpc**2/au.yr)).to('km/s').value
        A = zpmid['A'].sel(phase='2p')
        for yf in ['Ptot','Pturb','Pth','Pimag','oPimag','dPimag','Prad']:
            y = zpmid[yf].sel(phase='2p')/A*s.u.pok
            ydata[yf] = y
            y = zpmid[yf].sel(phase='2p')/A*s.u.pok/zpmid['sfr40']*yield_conv
            ydata[yf.replace('Pi','Y').replace('P','Y')] = y
        for yf in ['nH']:
            y = zpmid[yf].sel(phase='2p')/A
            ydata[yf] = y
        for yf in ['sigma_eff_mid','sigma_eff','sigma_turb_mid','sigma_turb',
                   'sigma_th_mid','sigma_th']:
            y = zpmid[yf].sel(phase='2p')
            ydata[yf] = y
        for yf in ['PDE','sfr10','sfr40','sfr100','Sigma_gas']:
            ydata[yf] = zpmid[yf]
        ydata['W'] = zpwmid['W']*s.u.pok
        ydata['tdep40'] = zpmid['Sigma_gas']/zpmid['sfr40']
        ydata['Zgas'] = s.Zgas*zpmid['Sigma_gas']/zpmid['Sigma_gas']
        ydata['Zdust'] = s.Zdust*zpmid['Sigma_gas']/zpmid['Sigma_gas']

        if reduce: ydata = ydata.reduce(func,dim='time',**func_kwargs)
        ydata = ydata.to_array().drop('phase')
        ydata = ydata.assign_coords(name=m)

        return ydata

    def add_legend(self,kind='R8',main=True,beta=True,beta_loc=5,**kwargs):
        colors = self.plt_kwargs[kind]['colors']
        from matplotlib.lines import Line2D
        if main:
            custom_lines = []
            for c in colors:
                custom_lines.append(Line2D([0], [0], color=c))
            leg1 = plt.legend(custom_lines,
                              ['(1,1)','(0.3,0.3)','(0.1,0.1)','(0.1,0.025)'],
                              title=r'$(Z_{\rm g}^\prime,Z_{\rm d}^\prime)$',**kwargs)
        if beta:
            if 'loc' in kwargs: kwargs.pop('loc')
            beta_def = 1
            beta_alt = 10
            labels = [r'$\beta_0 = {}$'.format(beta_def),
                      r'$\beta_0 = {}$'.format(beta_alt)]
            custom_lines2 = [Line2D([0], [0], ls = '-', color='k'),
                             Line2D([0], [0], ls = '--', color='k')]
            leg2 = plt.legend(custom_lines2,labels,loc=beta_loc,**kwargs)
            if main: plt.gca().add_artist(leg1)

    def get_PW_time_series(self,m,dt=0,zrange=slice(-10,10),recal=False):
        s = self.sa.set_model(m)

        # test needs for recalculation
        zpfiles = [os.path.join(s.basedir,'zprof','{}.PWzprof.nc'.format(s.problem_id)),
                   os.path.join(s.basedir,'zprof','{}.zpmid.nc'.format(s.problem_id)),
                   os.path.join(s.basedir,'zprof','{}.zpwmid.nc'.format(s.problem_id))]
        print("Getting P, W time series for {}".format(m))

        for f in zpfiles:
            isexist = os.path.isfile(f)
            if isexist:
                isold = os.path.getmtime(f) < os.path.getmtime(s.files['zprof'][-1])
                recal = recal | isold
                if isold:
                    print("  -- {} is old".format(f))
                    break
            else:
                print("  -- {} is not available".format(f))
                recal = recal | (~isexist)
                break

        if not recal:
            print("  -- read from files")
        else:
            print("  -- recalculate from zprof")

        zprof = get_PW_zprof(s, recal=recal)
        zpmid, zpwmid = get_PW_time_series_from_zprof(s,zprof,dt=dt,zrange=zrange,recal=recal)
        return zpmid, zpwmid


def add_torb(ax,torb,ticklabels=True):
    # Define function and its inverse
    f = lambda x: x/torb
    g = lambda x: torb*x

    ax2 = ax.secondary_xaxis("top", functions=(f,g))
    plt.setp(ax2.get_xticklabels(),visible=ticklabels)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='x',which='both',top=False)
    return ax2

def get_smoothed(y,t,dt):
    dt = 5
    window = int(dt/t.diff().median())
    return y.rolling(window=window,center=True,min_periods=1).mean()

def plot_errors(x,y,quantiles=[0.16,0.5,0.84],**kwargs):
    qx = list(x.quantile(quantiles))
    qy = list(y.quantile(quantiles))
    plt.errorbar(qx[1],qy[1],
                 yerr=[[qy[1]-qy[0]],[qy[2]-qy[1]]],
                 xerr=[[qx[1]-qx[0]],[qx[2]-qx[1]]],**kwargs)

def add_boxplot(pdata,group='R8-b1',field='sfr40',offset=0,tslice=None,label=True,beta_label=False):
    if tslice is None:
        torb = pdata.torb_code[group.split('-')[0]]
        tslice = slice(torb*2,torb*5)
    namelist,ylist,Zdlist = pdata.collect_hst_list(field,group=group,tslice=tslice)
    Zd_to_idx = {1:0,0.3:1,0.1:2,0.025:3}
    pos = np.array([Zd_to_idx[Zd] for Zd in Zdlist])
    width = 0.4

    colors = [pdata.plt_kwargs[group]['colors'][idx] for idx in pos]
    ls = pdata.plt_kwargs[group]['ls']

    if 'b10' in group: pos = pos.astype('float')+width*1.1
    box = plt.boxplot(ylist,positions=pos+offset,widths=width,
                      showfliers=False,patch_artist=True,
                      medianprops=dict(color='k',ls=ls))

    for artist,c in zip(box['boxes'],colors):
        plt.setp(artist, color=c, alpha=0.8, ls = ls)
    for element in ['whiskers','caps']:
        for artist1,artist2,c in zip(box[element][::2],box[element][1::2],colors):
            plt.setp(artist1, color=c, ls = ls)
            plt.setp(artist2, color=c, ls = ls)
    if label:
        labels = [r'$Z_{\rm g}^\prime=1$'+'\n'+r'$Z_{\rm d}^\prime =1$',
                    r'$Z_{\rm g}^\prime=0.3$'+'\n'+r'$Z_{\rm d}^\prime =0.3$',
                    r'$Z_{\rm g}^\prime=0.1$'+'\n'+r'$Z_{\rm d}^\prime =0.1$',
                    r'$Z_{\rm g}^\prime=0.1$'+'\n'+r'$Z_{\rm d}^\prime =0.025$']
        for c,name, x, y in zip(colors,labels,pos,box['caps'][1::2]):
            y0,y1=y.get_ydata()
            plt.annotate(name,(x,y0*1.05),xycoords='data',ha='center',va='bottom',fontsize='x-small',color=c)
    if beta_label:
        for c, x, y in zip(colors,pos,box['medians']):
            if 'b10' in group:
                name = r'$\beta_0=10$'
            else:
                name = r'$\beta_0=1$'
            y0,y1=y.get_ydata()
            plt.annotate(name,(x,y0*0.99),xycoords='data',ha='center',va='top',fontsize='x-small',color='k')
    plt.ylabel(r'$\Sigma_{\rm SFR}\,[M_\odot\,{\rm kpc^{-2}\,yr^{-1}}]$')
    ax = plt.gca()
    ax.grid(visible=False)
    ax.xaxis.set_tick_params(which='both',bottom=False,top=False,labelbottom=False)

def get_PW_zprof(s,recal=False):
    fzpmid=os.path.join(s.basedir,'zprof','{}.PWzprof.nc'.format(s.problem_id))
    if not recal:
        if os.path.isfile(fzpmid):
            with xr.open_dataset(fzpmid) as zpmid:
                return zpmid

    if hasattr(s,'newzp'):
        zp = s.newzp
    else:
        zp = PaperData.zprof_rename(s)

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


    # rearrange phases
    twop=zpmid.sel(phase=['CMM','CNM','UNM','WNM']).sum(dim='phase').assign_coords(phase='2p')
    hot=zpmid.sel(phase=['WHIM','HIM']).sum(dim='phase').assign_coords(phase='hot')
    if 'WIM' in zpmid.phase:
        wim=zpmid.sel(phase=['WIM']).sum(dim='phase').assign_coords(phase='WIM')
        zpmid = xr.concat([twop,wim,hot],dim='phase')
    else:
        zpmid = xr.concat([twop,hot],dim='phase')

    if os.path.isfile(fzpmid): os.remove(fzpmid)
    zpmid.to_netcdf(fzpmid)

    return zpmid

def get_PW_time_series_from_zprof(s,zprof,sfr=None,dt=0,zrange=slice(-10,10),recal=False):
    fzpmid=os.path.join(s.basedir,'zprof','{}.zpmid.nc'.format(s.problem_id))
    fzpw=os.path.join(s.basedir,'zprof','{}.zpwmid.nc'.format(s.problem_id))

    if (os.path.isfile(fzpmid) and os.path.isfile(fzpw)) and (not recal):
        zpmid = xr.open_dataset(fzpmid)
        zpmid.close()
        # smoothing
        if dt>0:
            window = int(dt/zpmid.time.diff(dim='time').median())
            zpmid_t = zpmid.rolling(time=window,center=True,min_periods=1).mean()
        else:
            zpmid_t = zpmid
        zpwmid = xr.open_dataset(fzpw)
        zpwmid.close()
        return zpmid_t, zpwmid

    # szeff
    szeff = np.sqrt(zprof['Ptot'].sum(dim='z')/zprof['nH'].sum(dim='z'))
    vzeff = np.sqrt(zprof['Pturb'].sum(dim='z')/zprof['nH'].sum(dim='z'))
    cseff = np.sqrt(zprof['Pth'].sum(dim='z')/zprof['nH'].sum(dim='z'))

    # select midplane
    zpmid = zprof.sel(z=zrange).mean(dim='z')
    zpwmid = zprof.sel(z=zrange).mean(dim='z').sum(dim='phase')

    # SFR from history
    vol = np.prod(s.domain['Lx'])
    area = vol/s.domain['Lx'][2]

    h = pa.read_hst(s.files['hst'])

    zpmid['sfr10'] = xr.DataArray(np.interp(zpmid.time_code,h['time'],h['sfr10']),coords=[zpmid.time])
    zpmid['sfr40'] = xr.DataArray(np.interp(zpmid.time_code,h['time'],h['sfr40']),coords=[zpmid.time])
    zpmid['sfr100'] = xr.DataArray(np.interp(zpmid.time_code,h['time'],h['sfr100']),coords=[zpmid.time])

    if sfr is None:
        zpmid['sfr'] = xr.DataArray(np.interp(zpmid.time_code,h['time'],h['sfr10']),coords=[zpmid.time])
    else:
        zpmid['sfr'] = sfr

    # sz from history
    KE = h['x1KE']+h['x2KE']+h['x3KE']
    if 'x1ME' in h:
        szmag = h['x1ME']+h['x2ME']-2.0*h['x3ME']
        ME = h['x1ME']+h['x2ME']+h['x3ME']
    else:
        szmag = 0.0
        ME = 0.0

    P = (h['totalE'] - KE - ME)*(5/3.-1)
    szeff_mid = np.sqrt((2.0*h['x3KE']+P+szmag)/h['mass'])
    zpmid['szeff'] = xr.DataArray(np.interp(zpmid.time_code,h['time'],szeff_mid),coords=[zpmid.time])

    # PDE

    zpmid['sigma_eff'] = szeff
    zpmid['sigma_turb'] = vzeff
    zpmid['sigma_th'] = cseff
    zpmid['sigma_eff_mid'] = np.sqrt(zpmid['Ptot']/zpmid['nH'])
    zpmid['sigma_turb_mid'] = np.sqrt(zpmid['Pturb']/zpmid['nH'])
    zpmid['sigma_th_mid'] = np.sqrt(zpmid['Pth']/zpmid['nH'])
    zpmid['Sigma_gas'] = xr.DataArray(np.interp(zpmid.time_code,h['time'],h['mass']*s.u.Msun*vol/area),coords=[zpmid.time])
    rhosd=0.5*s.par['problem']['SurfS']/s.par['problem']['zstar']+s.par['problem']['rhodm']
    zpmid['PDE1'] = np.pi*zpmid['Sigma_gas']**2/2.0*(ac.G*(ac.M_sun/ac.pc**2)**2/ac.k_B).cgs.value
    zpmid['PDE2_2p'] = zpmid['Sigma_gas']*np.sqrt(2*rhosd)*zpmid['sigma_eff'].sel(phase='2p')*(np.sqrt(ac.G*ac.M_sun/ac.pc**3)*(ac.M_sun/ac.pc**2)*au.km/au.s/ac.k_B).cgs.value
    zpmid['PDE2_2p_mid'] = zpmid['Sigma_gas']*np.sqrt(2*rhosd)*zpmid['sigma_eff_mid'].sel(phase='2p')*(np.sqrt(ac.G*ac.M_sun/ac.pc**3)*(ac.M_sun/ac.pc**2)*au.km/au.s/ac.k_B).cgs.value
    zpmid['PDE2'] = zpmid['Sigma_gas']*np.sqrt(2*rhosd)*zpmid['szeff']*(np.sqrt(ac.G*ac.M_sun/ac.pc**3)*(ac.M_sun/ac.pc**2)*au.km/au.s/ac.k_B).cgs.value
    zpmid['PDE_2p_mid'] = zpmid['PDE1']+zpmid['PDE2_2p_mid']
    zpmid['PDE_2p'] = zpmid['PDE1']+zpmid['PDE2_2p']
    zpmid['PDE'] = zpmid['PDE1']+zpmid['PDE2']

    if os.path.isfile(fzpmid): os.remove(fzpmid)
    if os.path.isfile(fzpw): os.remove(fzpw)

    zpmid.to_netcdf(fzpmid)
    zpwmid.to_netcdf(fzpw)

    zpmid.attrs = PaperData.set_zprof_attr()

    # smoothing
    if dt>0:
        window = int(dt/zpmid.time.diff(dim='time').median())
        zpmid = zpmid.rolling(time=window,center=True,min_periods=1).mean()

    return zpmid, zpwmid

def plot_DE(pdata,m,tr,xf,yf,label='',ax=None,fit=False,qr=[0.16,0.5,0.84]):
    Punit_label=r'$/k_B\,[{\rm cm^{-3}\,K}]$'
    sfr_unit_label=r'$\,[M_\odot{\rm \,kpc^{-2}\,yr}]$'

    if ax is None: ax = plt.gca()
    plt.sca(ax)
    s = pdata.sa.set_model(m)
    zpmid, zpwmid = pdata.get_PW_time_series(m)

    zpmid = zpmid.sel(time=tr)
    zpwmid = zpwmid.sel(time=tr)

    wpdata=dict(sfr = zpmid['sfr'], W=zpwmid['W']*s.u.pok,PDE=zpmid['PDE_2p'],
                Ptot=zpmid['Ptot'].sel(phase='2p')/zpmid['A'].sel(phase='2p')*s.u.pok)

    x = wpdata[xf]
    y = wpdata[yf]
    plt.plot(x,y,'o',markersize=5,markeredgewidth=0,color=s.color,alpha=0.3)
    qx = x.quantile(qr).data
    qy = y.quantile(qr).data
    plt.errorbar(qx[1],qy[1],
                 xerr=[[qx[1]-qx[0]],[qx[2]-qx[1]]],
                 yerr=[[qy[1]-qy[0]],[qy[2]-qy[1]]],
                 marker='o',markersize=8,ecolor='k',markeredgecolor='k',
                 color=s.color,zorder=10,label=label)
    xl = zpmid.attrs['Plabels'][xf]
    if xf != 'W': xl += r'${}_{\rm ,2p}$'
    xl += Punit_label
    if yf == 'sfr':
        yl = r'$\Sigma_{\rm SFR}$'+sfr_unit_label
    else:
        yl = zpmid.attrs['Plabels'][yf]
        if yf != 'W': yl += r'${}_{\rm ,2p}$'
        yl+=Punit_label
    plt.xlabel(xl)
    plt.ylabel(yl)
    # draw reference line
    Prange = np.logspace(2,7)
    plt.plot(Prange,Prange,ls=':',color='k')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(5.e3,5.e5)
    if yf == 'sfr':
        plt.ylim(5.e-4,5.e-1)
    else:
        plt.ylim(5.e3,5.e5)
        ax.set_aspect('equal')

    if fit:
        if xf == 'W' and yf == 'Ptot':
            plt.plot(Prange,10.**(0.99*np.log10(Prange)+0.083),ls='-',color='k')
        if xf == 'PDE' and yf == 'Ptot':
            plt.plot(Prange,10.**(1.03*np.log10(Prange)-0.199),ls='-',color='k')
        if xf == 'PDE' and yf == 'W':
            plt.plot(Prange,10.**(1.03*np.log10(Prange)-0.276),ls='-',color='k')
        if xf == 'W' and yf == 'sfr':
            plt.plot(Prange,10.**(1.18*np.log10(Prange)-7.32),ls='-',color='k')
        if xf == 'PDE' and yf == 'sfr':
            plt.plot(Prange,10.**(1.18*np.log10(Prange)-7.32),ls='-',color='k')
        if xf == 'Ptot' and yf == 'sfr':
            plt.plot(Prange,10.**(1.17*np.log10(Prange)-7.43),ls='-',color='k')

def plot_Pcomp(pdata,m,tr,yf,xf='PDE',label='',ax=None,fit=False,qr=[0.16,0.5,0.84]):
    s = pdata.sa.set_model(m)
    Punit_label=r'$/k_B\,[{\rm cm^{-3}\,K}]$'

    if ax is None: ax = plt.gca()
    plt.sca(ax)

    zpmid, zpw = pdata.get_PW_time_series(m)

    zpmid = zpmid.sel(time=tr)
    zpw = zpw.sel(time=tr)

    wpdata=dict(W=zpw['W']*s.u.pok,PDE=zpmid['PDE'],
                Ptot=zpmid['Ptot'].sel(phase='2p')/zpmid['A'].sel(phase='2p')*s.u.pok)

    Ptot=zpmid['Ptot'].sel(phase='2p')
    A = zpmid['A'].sel(phase='2p')
    x = wpdata[xf]
    if not yf in zpmid: return
    y = zpmid[yf].sel(phase='2p')/Ptot
    plt.plot(x,y,'o',markersize=5,markeredgewidth=0,color=s.color,alpha=0.3)
    qx = x.quantile(qr).data
    qy = y.quantile(qr).data
    plt.errorbar(qx[1],qy[1],
                 xerr=[[qx[1]-qx[0]],[qx[2]-qx[1]]],
                 yerr=[[qy[1]-qy[0]],[qy[2]-qy[1]]],
                 marker='o',markersize=8,ecolor='k',markeredgecolor='k',
                 color=s.color,zorder=10,label=label)
    xl = zpmid.attrs['Plabels'][xf]
    if xf != 'W': xl += r'${}_{\rm ,2p}$'
    xl += Punit_label
    yl = zpmid.attrs['Plabels'][yf]
    if yf != 'W': yl += r'${}_{\rm ,2p}$'
    yl += r'$/$'+zpmid.attrs['Plabels']['Ptot']+r'${}_{\rm ,2p}$'

    plt.xlabel(xl)
    plt.ylabel(yl)


    # draw reference line
    Prange = np.logspace(2,7)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(1.e-2,1)
    plt.xlim(1.e3,1.e6)
    if fit:
        if yf == 'Pth':
            plt.plot(Prange,10.**(-0.275*np.log10(Prange)+0.517),ls='-',color='k')
        if yf == 'Pturb':
            plt.plot(Prange,10.**(0.129*np.log10(Prange)-0.918),ls='-',color='k')

def plot_Pcomp_sfr(pdata,m,tr,yf,label='',ax=None,fit=False,qr=[0.16,0.5,0.84]):
    s = pdata.sa.set_model(m)
    Punit_label=r'$/k_B\,[{\rm cm^{-3}\,K}]$'
    sfr_unit_label=r'$\,[M_\odot{\rm \,kpc^{-2}\,yr}]$'


    if ax is None: ax = plt.gca()
    plt.sca(ax)

    zpmid, zpw = pdata.get_PW_time_series(m)
    zpmid = zpmid.sel(time=tr)
    zpw = zpw.sel(time=tr)

    x = zpmid['sfr']
    if not yf in zpmid: return
    A = zpmid['A'].sel(phase='2p')
    y = zpmid[yf].sel(phase='2p')/A*s.u.pok
    plt.plot(x,y,'o',markersize=5,markeredgewidth=0,color=s.color,alpha=0.3)
    qx = x.quantile(qr).data
    qy = y.quantile(qr).data
    plt.errorbar(qx[1],qy[1],
                 xerr=[[qx[1]-qx[0]],[qx[2]-qx[1]]],
                 yerr=[[qy[1]-qy[0]],[qy[2]-qy[1]]],
                 marker='o',markersize=8,ecolor='k',markeredgecolor='k',
                 color=s.color,zorder=10,label=label)
    plt.xlabel(r'$\Sigma_{\rm SFR}$'+sfr_unit_label)
    plt.ylabel(zpmid.attrs['Plabels'][yf]+r'${}_{\rm ,2p}$'+Punit_label)

    # draw reference line
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(5.e-4,0.5)
    plt.ylim(1.e3,1.e6)
    sfrrange = np.logspace(-4,2)
    if fit:
        if yf=='Pth':
            plt.plot(sfrrange,10.**(0.603*np.log10(sfrrange)+4.99),ls='-',color='k')
#             plt.plot(sfrrange,10.**(0.86*np.log10(sfrrange)+5.69),ls=':',color='k')
        if yf in ['Pturb']:#,'Pimag','oPimag','dPimag']:
            plt.plot(sfrrange,10.**(0.960*np.log10(sfrrange)+6.17),ls='-',color='k')
#             plt.plot(sfrrange,10.**(0.89*np.log10(sfrrange)+6.3),ls=':',color='k')
        if yf=='Ptot':
            plt.plot(sfrrange,10.**(0.840*np.log10(sfrrange)+6.26),ls='-',color='k')
#             plt.plot(sfrrange,10.**(0.847*np.log10(sfrrange)+6.27),ls=':',color='k')

def plot_Upsilon_sfr(pdata,m,tr,xf,yf,label='',ax=None,fit=False,qr=[0.16,0.5,0.84]):
    s = pdata.sa.set_model(m)
    yield_conv = ((au.cm**(-3)*au.K*ac.k_B)/(ac.M_sun/ac.kpc**2/au.yr)).to('km/s').value
    Uunit_label=r'$\,[{\rm km/s}]$'
    Punit_label=r'$/k_B\,[{\rm cm^{-3}\,K}]$'
    sfr_unit_label=r'$\,[M_\odot{\rm \,kpc^{-2}\,yr}]$'

    if ax is None: ax = plt.gca()
    plt.sca(ax)

    zpmid, zpw = pdata.get_PW_time_series(m)

    zpmid = zpmid.sel(time=tr)
    zpw = zpw.sel(time=tr)


    if not yf in zpmid: return
    A = zpmid['A'].sel(phase='2p')
    y = zpmid[yf].sel(phase='2p')/A*s.u.pok/zpmid['sfr40']*yield_conv
    if xf == 'Zgas':
        x = s.Zgas * zpmid['time']/zpmid['time']
    elif xf == 'Zdust':
        x = s.Zdust * zpmid['time']/zpmid['time']
    else:
        x = zpmid[xf]
    plt.plot(x,y,'o',markersize=5,markeredgewidth=0,color=s.color,alpha=0.3)

    qx = x.quantile(qr).data
    qy = y.quantile(qr).data
    plt.errorbar(qx[1],qy[1],
                 xerr=[[qx[1]-qx[0]],[qx[2]-qx[1]]],
                 yerr=[[qy[1]-qy[0]],[qy[2]-qy[1]]],
                 marker='o',markersize=8,ecolor='k',markeredgecolor='k',
                 color=s.color,zorder=10,label=label)
    if xf == 'sfr': plt.xlabel(r'$\Sigma_{\rm SFR}$'+sfr_unit_label)
    if xf == 'PDE': plt.xlabel(r'$P_{\rm DE}$'+Punit_label)
    if xf == 'PDE_2p': plt.xlabel(r'$P_{\rm DE,2p}$'+Punit_label)
    if xf == 'Zgas': plt.xlabel(r'$Z_{\rm gas}^\prime$')
    if xf == 'Zdust': plt.xlabel(r'$Z_{\rm dust}^\prime$')
    plt.ylabel(r'$\Upsilon_{{\rm {}}}$'.format(yf[2:] if yf.startswith('Pi') else yf[1:])+Uunit_label)


    # draw reference line
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(10,3000)
    if xf == 'sfr':
        plt.xlim(5.e-4,0.5)
        sfrrange = np.logspace(-4,2)
        if fit:
            if yf=='Pth':
#                 plt.plot(sfrrange,200*(sfrrange/1.e-2)**(-0.14),ls=':',color='k')
                plt.plot(sfrrange,110*(sfrrange/1.e-2)**(-0.4),ls='-',color='k')
            if yf in ['Pturb','Pimag','oPimag','dPimag']:
            #                 plt.plot(sfrrange,700*(sfrrange/1.e-2)**(-0.11),ls=':',color='k')
                plt.plot(sfrrange,330*(sfrrange/1.e-2)**(-0.05),ls='-',color='k')
            if yf=='Ptot':
#                 plt.plot(sfrrange,770*(sfrrange/1.e-2)**(-0.15),ls=':',color='k')
                plt.plot(sfrrange,740*(sfrrange/1.e-2)**(-0.2),ls='-',color='k')
    if xf in ['PDE','PDE_2p']:
        plt.xlim(1.e3,1.e6)
        Prange = np.logspace(3,6)
        if fit:
            if yf=='Pth':
                plt.plot(Prange,10.**(-0.506*np.log10(Prange)+4.45),ls='-',color='k')
            if yf in ['Pturb']:#,'Pimag','oPimag','dPimag']:
                plt.plot(Prange,10.**(-0.060*np.log10(Prange)+2.81),ls='-',color='k')
            if yf=='Ptot':
                plt.plot(Prange,10.**(-0.212*np.log10(Prange)+3.86),ls='-',color='k')

def plot_PWYstack(pdata,m,tr,i,plabel=True,Upsilon=False,
                 label='',ax=None,qr=[0.16,0.5,0.84],errorbar_color='k'):
    s = pdata.sa.set_model(m)
    Punit_label=r'$/k_B\,[10^4{\rm cm^{-3}\,K}]$'
    Punit = 1.e-4*s.u.pok
    Pcolors=cmr.get_sub_cmap('cmr.cosmic',0.3,1,N=4).colors
    Wcolors=cmr.get_sub_cmap('cmr.ember',0.3,1,N=4).colors
    yield_conv = ((au.cm**(-3)*au.K*ac.k_B)/(ac.M_sun/ac.kpc**2/au.yr)).to('km/s').value
    Uunit_label=r'$\,[{\rm km/s}]$'
    if ax is None: ax = plt.gca()
    plt.sca(ax)

    zpmid, zpw = pdata.get_PW_time_series(m)
    zpmid = zpmid.sel(time=tr)
    zpw = zpw.sel(time=tr)

    Ptot=zpmid['Ptot'].sel(phase='2p')
    A = zpmid['A'].sel(phase='2p')

    # pressure
    f0 = 0
    for yf,c,cl in zip(['oPimag','dPimag','Pth','Pturb'],Pcolors,['w','w','k','k']):
        if not (yf in zpmid): continue
        if Upsilon:
            y = zpmid[yf].sel(phase='2p')/A*s.u.pok/zpmid['sfr40']*yield_conv
            width=0.8
            offset=0.0
            align='center'
        else:
            y = zpmid[yf].sel(phase='2p')/A*Punit
            width=-0.4
            offset=-0.2
            align='edge'
        qy = y.quantile(qr).data
        plt.bar(label,qy[1],bottom=f0,color=c,
                width=width,align=align)

        if plabel:
            plt.annotate(zpmid.attrs['Ulabels'][yf] if Upsilon else zpmid.attrs['Plabels'][yf],
                         (i+offset,f0+0.5*qy[1]),
                         color=cl, ha='center',va='center')
        f0 = f0+qy[1]
    if 'Prad' in zpw:
        y = zpw['Prad']*Punit
        qy = y.quantile(qr).data
        plt.bar(label,qy[1],bottom=f0,color=Wcolors[-1],
                width=width,align=align)
    if Upsilon:
        y = zpmid['Ptot'].sel(phase='2p')/A*s.u.pok/zpmid['sfr40']*yield_conv
    else:
        y = zpmid['Ptot'].sel(phase='2p')/A*Punit
    qy = y.quantile(qr).data
    if errorbar_color is not None: plt.plot([i+offset,i+offset],[qy[0],qy[2]],color=errorbar_color)
#     plt.plot([i+offset,i+offset],[qy[1],qy[1]],'or')

    # weight
    if not Upsilon:
        f0 = 0
        for yf,c,cl in zip(['Wsg','Wext'],Wcolors[1:],['w','k']):
            if not (yf in zpmid): continue
            y = zpw[yf]*Punit
            qy = y.quantile(qr).data
            plt.bar(label,qy[1],bottom=f0,color=c,
                    width=-width,align=align)
            if plabel:
                plt.annotate(zpmid.attrs['Plabels'][yf],(i-offset,f0+0.5*qy[1]),
                            color=cl, ha='center',va='center')
            f0 = f0+qy[1]
        y = (zpw['W'])*Punit
        qy = y.quantile(qr).data
        if errorbar_color is not None: plt.plot([i-offset,i-offset],[qy[0],qy[2]],color=errorbar_color)
        plt.ylabel(r'$P$'+Punit_label)
    else:
        plt.ylabel(r'$\Upsilon$'+Uunit_label)
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)

def plot_PWYbox(pdata,m,tr,i,nmodels,legend=True,Upsilon=False,
                Pcomps = ['Ptot','Pturb','Pth','dPimag','oPimag'],
                label='',ax=None, edge_color='k'):
    s = pdata.sa.set_model(m)

    Punit_label=r'$/k_B\,[10^4{\rm cm^{-3}\,K}]$'
    Punit = 1.e-4*s.u.pok
    Pcolors=cmr.get_sub_cmap('cmr.cosmic',0.3,1,N=4).colors
    Pcolors=cmr.get_sub_cmap('cmr.neutral',0.3,0.9,N=4).colors
    Wcolors=cmr.get_sub_cmap('cmr.ember',0.3,1,N=4).colors
    yield_conv = ((au.cm**(-3)*au.K*ac.k_B)/(ac.M_sun/ac.kpc**2/au.yr)).to('km/s').value
    Uunit_label=r'$\,[{\rm km/s}]$'
    if ax is None: ax = plt.gca()
    plt.sca(ax)

    zpmid, zpw = pdata.get_PW_time_series(m)
    zpmid = zpmid.sel(time=tr)
    zpw = zpw.sel(time=tr)

    Ptot=zpmid['Ptot'].sel(phase='2p')
    A = zpmid['A'].sel(phase='2p')
    ncomp = len(Pcomps)
    w = 0.8/nmodels

    Ycomp = []
    pos = []
    Ylabels = []
    for j,yf in enumerate(Pcomps):
        if yf == 'sfr':
            y = zpmid['sfr40']
            Ylabels.append(r'$\Sigma_{\rm SFR}$')
        else:
            if Upsilon:
                Ylabels.append(zpmid.attrs['Ulabels'][yf])
            else:
                Ylabels.append(zpmid.attrs['Plabels'][yf])
            if not (yf in zpmid): continue
            if Upsilon:
                y = zpmid[yf].sel(phase='2p')/A*s.u.pok/zpmid['sfr40']*yield_conv
            else:
                y = zpmid[yf].sel(phase='2p')/A*s.u.pok

        Ycomp.append(y.dropna(dim='time'))

        offset = (i)/nmodels*0.8-0.4 + 0.5*w
        pos.append(j+offset)
    box = plt.boxplot(Ycomp,positions=pos,
                      widths=w,whis=[16,84],showfliers=False,
                      patch_artist=True,
#                       showmeans=True,
#                       meanprops=dict(markerfacecolor='tab:orange',markeredgecolor='w',
#                                      markersize=5,markeredgewidth=0.5,marker='*')
                     )
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(box[element], color=edge_color)
    plt.setp(box['boxes'],facecolor=s.color)

    plt.xticks(ticks=np.arange(ncomp),labels=Ylabels,fontsize='medium')
    plt.xlim(-0.5,ncomp-0.5)
    for c, yf, xc in zip(Pcolors, Pcomps[::-1], np.arange(ncomp)[::-1]):
        plt.axvspan(xc-0.5,xc+0.5,color=c,alpha=0.1,lw=0)
    if Upsilon:
        plt.ylabel(r'$\Upsilon$'+Uunit_label)
    else:
        plt.ylabel(r'$P$'+Punit_label)
    plt.yscale('log')
    if legend:
        x0=0.85
        dx=0.02
        y0=0.95
        dy=0.05

        c = s.color
        plt.annotate('        ',(x0,y0-dy*i),xycoords='axes fraction',
                     ha='right',va='top',size='xx-small',
                     backgroundcolor=c,color=c)
        plt.annotate(label,(x0+dx,y0-dy*i),xycoords='axes fraction',
                     size='xx-small',ha='left',va='top')
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)