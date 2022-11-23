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

import pyathena as pa

class PaperIData(PaperData):
    def __init__(self):
        self.outdir='/tigress/changgoo/public_html/TIGRESS-NCR/I-figures/'
        basedir = '/tigress/changgoo/TIGRESS-NCR/'
        scrdir = '/scratch/gpfs/changgoo/TIGRESS-NCR/'
        models_full = dict(R8_8pc=scrdir+'R8_8pc_NCR.full.v3',
                           R8_4pc=basedir+'R8_4pc_NCR.full.xy2048.eps0.np768.has',
                           LGR4_4pc=scrdir+'LGR4_4pc_NCR.full.v3.frad.g0',
                        #    LGR4_2pc=basedir+'LGR4_2pc_NCR.full.xy1024.eps2.5e-9.np768')
                           LGR4_2pc=basedir+'LGR4_2pc_NCR.full')
        models_full2 = dict(R8_8pc2=basedir+'R8_8pc_NCR.full.latest2',
                            LGR4_4pc3=basedir+'LGR4_4pc_NCR.full.xy1024.eps1.e-8',
                            LGR4_4pc2=basedir+'LGR4_4pc_NCR.full',
                            LGR4_8pc2=basedir+'LGR4_8pc_NCR.full')

        models=dict()
        models.update(models_full)
        models.update(models_full2)
        self.sa = pa.LoadSimTIGRESSNCRAll(models)
        self.models = dict(full = models_full)
        self._set_colors()
        self._set_plot_kwargs()
        self.set_model_list()

        os.makedirs(self.outdir,exist_ok=True)

    def _set_colors(self):
        colors=dict()
        colors['R8_8pc']='tab:cyan'
        colors['R8_4pc']='tab:blue'
        colors['LGR4_4pc']='tab:red'
        colors['LGR4_2pc']='tab:purple'
        self.colors=colors

    def _set_plot_kwargs(self):
        plt_kwargs=dict()
        #initialize
        for m in self.sa.models:
            if not m in self.colors:
                plt_kwargs[m]=dict()
                continue
            plt_kwargs[m]=dict(color=self.colors[m])
        self.plt_kwargs = plt_kwargs

    def set_model_list(self):
        mlist = dict(std = ['R8_4pc','R8_8pc','LGR4_2pc','LGR4_4pc'])
        trlist = dict(std = [slice(250,450)]*2+[slice(250,350)]*2)
        restart_from = dict(R8_4pc='R8_8pc2',R8_8pc='R8_8pc2',LGR4_2pc='LGR4_4pc2',LGR4_4pc='LGR4_4pc3',LGR4_4pc2='LGR4_8pc2')
        self.mlist = mlist
        self.trlist = trlist
        self.restart_from = restart_from

class RadConvergenceData(PaperData):
    def __init__(self):
        self.outdir='/tigress/changgoo/public_html/TIGRESS-NCR/Conv-figures/'
        basedir = '/tigress/changgoo/TIGRESS-NCR/'
        scrdir = '/scratch/gpfs/changgoo/TIGRESS-NCR/'
        models_radconv = dict(R8_xy1_eps0=scrdir+'R8_8pc_NCR.full.v3.RT.xy1024.eps0.0',
                              R8_xy2_eps0=scrdir+'R8_8pc_NCR.full.v3.frad.g0',
                              R8_xy4_eps0=scrdir+'R8_8pc_NCR.full.v3.RT.xy4096.eps0.0',
                              R8_xy05_eps0=scrdir+'R8_8pc_NCR.full.v3.RT.xy512.eps0.0',
                              R8_xy1_eps1_8=scrdir+'R8_8pc_NCR.full.v3.RT.xy1024.eps1.e-8',
                              R8_xy2_eps1_8=scrdir+'R8_8pc_NCR.full.v3.RT.xy2048.eps1.e-8',
                             )
        self.sa = pa.LoadSimTIGRESSNCRAll(models_radconv)

        os.makedirs(self.outdir,exist_ok=True)

class PaperIFigures():
    @staticmethod
    def get_label(tf,wf,normed=False):
        xlabel = dict(T=r'$T$',
                    T1=r'$T_\mu$',
                    nH=r'$n_H$',
                    pok=r'$P_{\rm th}/k_B$',
                    )

        ylabel = dict(vol=r'$dV/V(|z|<300{\rm\,pc})$',
                    mass=r'$dM/M(|z|<300{\rm\,pc})$',
                    nH2=r'$dM/M(|z|<300{\rm\,pc})$',
                    net_cool_rate=r'$d(\mathcal{L}-\mathcal{G})$',
                    netcool=r'$d(\mathcal{L}-\mathcal{G})$',
                    cool_rate=r'$d\mathcal{L}$',
                    heat_rate=r'$d\mathcal{G}$'
                    )

        units = dict(T=r'$\,[{\rm K}]$',T1=r'$\,[{\rm K}]$',
                    nH=r'$\,[{\rm cm^{-3}}]$',
                    pok=r'$\,[{\rm cm^{-3}\,K}]$',
                    net_cool_rate=r'${\rm erg\,s^{-1}\,cm^{-3}}$',
                    netcool=r'${\rm erg\,s^{-1}\,cm^{-3}}$',
                    cool_rate=r'${\rm erg\,s^{-1}\,cm^{-3}}$',
                    heat_rate=r'${\rm erg\,s^{-1}\,cm^{-3}}$',
                    )
        xl = xlabel[tf] + units[tf]
        yl_u = ylabel[wf][1:-1]
        if wf in ['netcool','net_cool_rate','cool_rate','heat_rate'] and normed: yl_u += r'/\mathcal{L}(|z|<300{\rm\,pc})'
        yl_l = r'd\log\,' + xlabel[tf][1:-1]
        yl = r'$\frac{{{}}}{{{}}}$'.format(yl_u,yl_l)
        if wf in ['vol','mass','nH2'] or normed:
            yl += r'$\,[{\rm dex^{-1}}]$'
        else:
            yl += r'$\,[$' + units[wf] + r'${\rm\,dex^{-1}}]$'

        return xl,yl

    @staticmethod
    def plot_1Dpdf(pdf,phdef,quantiles=[0.16,0.5,0.84],plt_kwargs=None,
                verbose=True, normvalue=1.0, overlay=False,
                xlabel=True,ylabel=True,legend=True,reduced='all'):
        if reduced is 'all':
            phdef_reduced = {ph['name']:[ph['idx']-1] for ph in phdef}
        elif reduced == 'eight':
            phdef_reduced = dict(CMM=[0],CNM=[1],UNM=[2,3],WNM=[4],WPIM=[5],WCIM=[6],WHIM=[7],HIM=[8])
        elif reduced == 'seven':
            phdef_reduced = dict(CNM=[0,1],UNM=[2,3],WNM=[4],WPIM=[5],WCIM=[6],WHIM=[7],HIM=[8])
        elif reduced == 'six':
            phdef_reduced = dict(CNM=[0,1],UNM=[2,3],WNM=[4],WIM=[5,6],WHIM=[7],HIM=[8])
        elif reduced == 'five':
            phdef_reduced = dict(CNM=[0,1],UNM=[2,3],WNM=[4],WIM=[5,6],HIM=[7,8])
        elif reduced == 'one':
            phdef_reduced = dict(all=np.arange(9))

        tf, = set(pdf.coords) - {'phase','time'}
        x = pdf[tf]
        dx=np.log10(x[1])-np.log10(x[0])
        wf = pdf.name
        ytot = pdf.sum(dim=[tf,'phase'])
        if 'time' in pdf.coords:
            ytot = ytot.mean(dim='time')
        autocolor=True if plt_kwargs is None else False
        for phname,idx in phdef_reduced.items():
            phnames=[ph['name'] for ph in np.array(phdef)[idx]]
            if autocolor:
                c=[ph['c'] for ph in np.array(phdef)[idx]][-1]
                plt_kwargs=dict(color=c,label=phname)
            else:
                if 'color' in plt_kwargs: c=plt_kwargs['color']
            y = pdf.sel(phase=[ph['name'] for ph in np.array(phdef)[np.atleast_1d(idx)]]).sum(dim='phase')/dx/normvalue
            if wf in ['vol','mass','nH2']: y /= ytot
            if 'time' in pdf.coords:
                q1,q2,q3=quantiles
                q = y.quantile(quantiles,dim='time')
                if verbose: print(phname,'{:5.3f}'.format((y.mean(dim='time')*dx).sum().data), end=' ')
                if phname == 'UIM': continue
                if overlay:
                    plt.plot(x,q.sel(quantile=q2),color=c,ls=':')
                else:
                    plt.plot(x,q.sel(quantile=q2),**plt_kwargs)
                    plt.fill_between(x,q.sel(quantile=q1),q.sel(quantile=q3),
                                    alpha=0.2,color=c,lw=0)
            else:
                plt.step(x,y,**plt_kwargs)
        xl,yl = PaperIFigures.get_label(tf,wf, normed = normvalue!=1.0)

        ymins = dict(vol=1.e-4,mass=1.e-4,nH2=1.e-4,netcool=1.e-22,net_cool_rate=1.e-22,cool_rate=1.e-22,heat_rate=1.e-22)
        xranges = dict(T=(10,2.e7),T1=(10,1.e8),nH=(1.e-5,1.e3),pok=(10.,1.e7))
        plt.xscale('log')
        plt.yscale('log')
        if xlabel: plt.xlabel(xl)
        if ylabel: plt.ylabel(yl,fontsize='large')
        if legend: plt.legend()
        plt.ylim(bottom=ymins[wf])
        plt.xlim(xranges[tf])
        if verbose: print('')

    @staticmethod
    def get_tseries(pdf,phdef,reduced='all'):
        if reduced is 'all':
            phdef_reduced = {ph['name']:[ph['idx']-1] for ph in phdef}
        elif reduced == 'eight':
            phdef_reduced = dict(CMM=[0],CNM=[1],UNM=[2,3],WNM=[4],WPIM=[5],WCIM=[6],WHIM=[7],HIM=[8])
        elif reduced == 'seven':
            phdef_reduced = dict(CNM=[0,1],UNM=[2,3],WNM=[4],WPIM=[5],WCIM=[6],WHIM=[7],HIM=[8])
        elif reduced == 'mass':
            phdef_reduced = dict(CNM=[0,1],UNM=[2,3],WNM=[4],WPIM=[5])
        elif reduced == 'vol':
            phdef_reduced = dict(UNM=[2,3],WNM=[4],WPIM=[5])
            phdef_reduced['W+HIM']=[7,8]
        elif reduced == 'six':
            phdef_reduced = dict(CNM=[0,1],UNM=[2,3],WNM=[4],WIM=[5,6],WHIM=[7],HIM=[8])
        elif reduced == 'five':
            phdef_reduced = dict(Cold=[0,1],UNM=[2,3],WNM=[4],WIM=[5,6],Hot=[7,8])
        elif reduced == 'one':
            phdef_reduced = dict(all=np.arange(9))

        tf, = set(pdf.coords) - {'phase','time'}
        x = pdf[tf]
        dx=np.log10(x[1])-np.log10(x[0])
        wf = pdf.name
        ytot = pdf.sum(dim=[tf,'phase'])
        tseries = []
        colors = []
        for phname,idx in phdef_reduced.items():
            phnames=[ph['name'] for ph in np.array(phdef)[idx]]
            c=[ph['c'] for ph in np.array(phdef)[idx]][-1]
            colors.append(c)
            y = pdf.sel(phase=[ph['name'] for ph in np.array(phdef)[np.atleast_1d(idx)]]).sum(dim='phase')
            y /= ytot
            tseries.append(y.sum(dim=tf))
        return tseries,colors,list(phdef_reduced.keys())

    @staticmethod
    def print_pdf_table(sa,ms,tr,reduced='seven',kind='percentile'):
        tf = 'nH'
        for wf in ['mass','vol']:
            if wf == 'mass': print('Mass Fraction')
            if wf == 'vol': print('Volume Fraction')
            for i,(m,ts) in enumerate(zip(ms,tr)):
                s = sa.set_model(m)
                data, colors, labels = \
                    PaperIFigures.get_tseries(getattr(s.pdf,tf)[wf].sel(time=ts),
                        s.pdf.phdef, reduced=wf if reduced is None else reduced)
                print_latex_string(data,labels,m,kind=kind)

    @staticmethod
    def plot_box_comparison(sa,ms,ts,wf,mcolors,reduced=None):
        if len(np.atleast_1d(ts)) == 1: ts = [ts]*len(ms)
        tf = 'nH'
        nmodel = len(ms)
        w = 0.8/nmodel
        offset = np.arange(nmodel+1)/nmodel*0.8-0.4 + 0.5*w
        for i,m in enumerate(ms):
            s = sa.set_model(m)
            data, colors, labels = \
                PaperIFigures.get_tseries(getattr(s.pdf,tf)[wf].sel(time=ts[i]),
                    s.pdf.phdef, reduced=wf if reduced is None else reduced)
            box=plt.boxplot(data,positions=np.arange(len(data))+offset[i],widths=w,
                            whis=[16,84],showfliers=False,
                            patch_artist=True,medianprops=dict(color='k'),)
            plt.setp(box['boxes'],'facecolor',mcolors[m])
        plt.xticks(ticks=np.arange(len(labels)),labels=labels)
        for d, c, l, xc in zip(data, colors, labels, np.arange(len(data))):
            plt.axvspan(xc-0.5,xc+0.5,color=c,alpha=0.1,lw=0)
        plt.ylim(0,1)
        plt.xlim(-0.5,len(data)-0.5)

    @staticmethod
    def add_box_legend(ms,mcolors,x0=0.90,dx=0.015,y0=0.95,dy=0.12,
                       fontsize=16,linewidth=3):
        for i,m in enumerate(ms):
            c = mcolors[m]
            plt.annotate('               ',(x0,y0-dy*i),xycoords='axes fraction',
                        ha='center',va='top',size=fontsize,
                        backgroundcolor=c,color=c)
            plt.annotate(m.replace('_','-'),(x0,y0-dy*i),xycoords='axes fraction',
                        ha='center',va='top',**texteffect(fontsize,linewidth=linewidth))

    @staticmethod
    def plot_box_comparison_both(sa,ms,ts,reduced=None):
        fig,axes = plt.subplots(1,2,figsize=(15,5))
        plt.sca(axes[0])
        plot_box_comparison(sa,ms,ts,'mass',reduced=reduced,legend=None)
        plt.ylabel('Mass Fraction')

        plt.sca(axes[1])
        plot_box_comparison(sa,ms,ts,'vol',reduced=reduced)
        plt.ylabel('Volume Fraction')
        return fig

    @staticmethod
    def plot_tseries(sa,ms,tr,wf='mass'):
        nmodel = len(ms)
        fig, axes = plt.subplots(nmodel,2,figsize=(10,3*nmodel),sharey='row',sharex='col')
        for i,m in enumerate(ms):
            s = sa.set_model(m)
            tf = 'nH'
            data, colors, labels = \
                PaperIFigures.get_tseries(getattr(s.pdf,tf)[wf].sel(time=tr),
                                          s.pdf.phdef, reduced='seven')
            plt.sca(axes[i,0])
            for d,c,l in zip(data,colors,labels):
                plt.plot(d['time'],d,color=c,label=l)
            plt.ylabel('{} fraction'.format(wf))
            plt.sca(axes[i,1])
            box=plt.boxplot(data,patch_artist=True,medianprops=dict(color='k'),labels=labels)
            for p,c in zip(box['boxes'],colors):
                p.set_facecolor(c)
            plt.annotate(m,(0.95,0.95),xycoords='axes fraction',ha='right',va='top')
        plt.sca(axes[-1,0])
        plt.xlabel('time')
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_PWYbox(s,tr,i,nmodels,legend=True,Upsilon=False,
                    Pcomps = ['Ptot','Pturb','Pth','dPimag','oPimag'],
                    label='',ax=None, facecolor='tab:blue', edgecolor='k'):
        Punit_label=r'$/k_B\,[10^4{\rm cm^{-3}\,K}]$'
        Punit = 1.e-4*s.u.pok
        Pcolors=cmr.get_sub_cmap('cmr.cosmic',0.3,1,N=4).colors
        Pcolors=cmr.get_sub_cmap('cmr.neutral',0.3,0.9,N=4).colors
        Wcolors=cmr.get_sub_cmap('cmr.ember',0.3,1,N=4).colors
        yield_conv = ((au.cm**(-3)*au.K*ac.k_B)/(ac.M_sun/ac.kpc**2/au.yr)).to('km/s').value
        Uunit_label=r'$\,[{\rm km/s}]$'
        if ax is None: ax = plt.gca()
        plt.sca(ax)

        zpmid,zpw = PaperData.get_Pmid_time_series(s,dt=0)
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
                y = zpmid['sfr']
                Ylabels.append(r'$\Sigma_{\rm SFR}$')
            else:
                if Upsilon:
                    Ylabels.append(zpmid.attrs['Ulabels'][yf])
                else:
                    Ylabels.append(zpmid.attrs['Plabels'][yf])
                if not (yf in zpmid): continue
                if Upsilon:
                    y = zpmid[yf].sel(phase='2p')/A*s.u.pok/zpmid['sfr']*yield_conv
                else:
                    y = zpmid[yf].sel(phase='2p')/A*s.u.pok

            Ycomp.append(y.dropna(dim='time'))

            offset = (i)/nmodels*0.8-0.4 + 0.5*w
            pos.append(j+offset)
        box = plt.boxplot(Ycomp,positions=pos,
                        widths=w,whis=[16,84],showfliers=False,
                        patch_artist=True,
                        )
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(box[element], color=edgecolor)
        plt.setp(box['boxes'],facecolor=facecolor)

        plt.xticks(ticks=np.arange(ncomp),labels=Ylabels,fontsize='medium')
        plt.xlim(-0.5,ncomp-0.5)
        for c, yf, xc in zip(Pcolors, Pcomps[::-1], np.arange(ncomp)[::-1]):
            plt.axvspan(xc-0.5,xc+0.5,color=c,alpha=0.1,lw=0)
        if Upsilon:
            plt.ylabel(r'$\Upsilon$'+Uunit_label)
        else:
            plt.ylabel(r'$P$'+Punit_label)
        plt.yscale('log')


def print_latex_string(data,labels,m,kind='percentile'):
    latex_string1 = '{:10s}'.format(m.replace('_','-'))
    total_mean = 0.0
    total_median = 0.0
    latex_string2 = '{:10s}'.format(m.replace('_','-'))
    for d,l in zip(data,labels):
        if d is None:
            latex_string1 += ' & & '
            latex_string2 += ' & & '
            continue
        darr = np.array(d)
        q = np.quantile(darr,[0.16,0.5,0.84])*100
        latex_string1 += ' & ({:5.3f},{:5.3f})'.format(darr.mean(),darr.std())
        if q[1] < 0.1:
            latex_string2 += ' & {:4.2f}^{{+{:4.2f}}}_{{-{:4.2f}}}'.format(q[1],q[2]-q[1],q[1]-q[0])
        else:
            latex_string2 += ' & {:3.1f}^{{+{:3.1f}}}_{{-{:3.1f}}}'.format(q[1],q[2]-q[1],q[1]-q[0])
        total_mean += darr.mean()
        total_median += q[1]
    if kind == 'mean-std': print(latex_string1 + ' \\\\')
    if kind == 'percentile': print(latex_string2 + ' \\\\')