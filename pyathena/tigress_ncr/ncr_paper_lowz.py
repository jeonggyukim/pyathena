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

            if '.b1.' in m:
                head += '-b1'
            elif '.b10.' in m:
                head += '-b10'
            
            if 'S30' in m:
                head += '-S30'
            if 'S100' in m:
                head += '-S100'
            print(head, m)
            if head in mgroup:
                mgroup[head].append(m)
            else:
                mgroup[head] = [m]
        mgroup['R8']=mgroup['R8-b1']
        mgroup['LGR4']=mgroup['LGR4-b1']

        self.mgroup = mgroup

    def _set_torb(self):
        self.torb_Myr = dict()
        self.torb_code = dict()
        for m in self.mlist:
            s = self.sa.set_model(m)
            torb = 2*np.pi/s.par['problem']['Omega']
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
        colors1 = cmr.take_cmap_colors('cmr.pride', 4, cmap_range=(0.15, 0.45))
        colors2 = cmr.take_cmap_colors('cmr.pride_r', 4, cmap_range=(0.15, 0.45))
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
    box = plt.boxplot(ylist,positions=pos+offset,widths=width,showfliers=False,patch_artist=True,
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