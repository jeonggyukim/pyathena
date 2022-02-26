import os
import os.path as osp
import pandas as pd
import numpy as np
import astropy.constants as ac
import astropy.units as au
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, SymLogNorm, Normalize
import cmasher.cm as cma
import seaborn as sns

import pyathena as pa
import gc

def draw_jointpdfs(s,num,pdf=None,zrange=None,save=True):
    '''draw joint PDFs of T and nH for cooling and heating processes
    '''
    savdir = s.get_savdir_pdf(zrange=zrange)
    pdf_cool, pdf_heat = s.get_coolheat_pdf(num,zrange=zrange)
    coolkey = set(list(pdf_cool.keys()))-{'total','OIold'}
    heatkey = set(list(pdf_heat.keys()))-{'total'}
    if pdf is not None:
        total_cooling = pdf['cool'].sum().data
        total_heating = pdf['heat'].sum().data
        pdf_cool=pdf[list(coolkey)+['total_cooling']].sum(dim='time').rename(total_cooling='total')
        pdf_heat=pdf[list(heatkey)+['total_heating']].sum(dim='time').rename(total_heating='total')
        pdf_cool/=total_cooling
        pdf_heat/=total_heating
        pdf_cool.attrs['total_cooling']=total_cooling
        pdf_heat.attrs['total_heating']=total_heating

    cmap_cool = sns.color_palette("Blues", as_cmap=True)
    cmap_heat = sns.color_palette("Reds", as_cmap=True)
    fig = plt.figure(figsize=(15,12))
    gs1 = gridspec.GridSpec(5,4)
    gs1.update(left=0.05, right=0.48, wspace=0.05)

    i=0
    axes=[]
    for c in list(coolkey):
        ax = plt.subplot(gs1[i])
        pdf_cool[c].plot(norm=LogNorm(1.e-5,10),ax=ax,add_colorbar=False,cmap=cmap_cool)
        plt.title('')
        ax.annotate(r'$\mathcal{{L}}_{{\rm {}}}$'.format(c.replace('_','-')),(0.95,0.95),xycoords='axes fraction',
                    fontsize='x-small',ha='right',va='top')
        axes.append(ax)
        i += 1

    for h in list(heatkey):
        ax = plt.subplot(gs1[i])
        pdf_heat[h].plot(norm=LogNorm(1.e-5,10),ax=ax,add_colorbar=False,cmap=cmap_heat)
        plt.title('')
        ax.annotate(r'$\mathcal{{G}}_{{\rm {}}}$'.format(h.replace('_','-')),
                    (0.95,0.95),xycoords='axes fraction',
                    fontsize='x-small',ha='right',va='top')
        axes.append(ax)
        i += 1

    for i,ax in enumerate(axes):
        plt.sca(ax)
        if i//4 < 4:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.xlabel('')
        else:
            plt.xlabel(r'$\log\,n_H\,[{\rm cm^{-3}}]$')
        if i%4 != 0 :
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.ylabel('')
        else:
            plt.ylabel(r'$\log\,T\,[{\rm K}]$')
        plt.xlim(-5,3)
        plt.ylim(1,8)

    gs2 = gridspec.GridSpec(3, 1)
    gs2.update(left=0.55, right=0.98, hspace=0.05)
    ax = plt.subplot(gs2[0])
    pdf_cool['total'].plot(norm=LogNorm(1.e-5,10),ax=ax,cmap=cmap_cool,
                           cbar_kwargs=dict(label=r'$\frac{d^2\mathcal{L}/\mathcal{L}_{\rm tot}}'
                                            r'{d\log T\,d\log n_H}[{\rm dex^{-2}}]$'))
    plt.title('')
    plt.xlim(-5,3)
    plt.ylim(1,8)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.xlabel('')
    plt.ylabel(r'$\log\,T\,[{\rm K}]$')

    ax = plt.subplot(gs2[1])
    pdf_heat['total'].plot(norm=LogNorm(1.e-5,10),ax=ax,cmap=cmap_heat,
                           cbar_kwargs=dict(label=r'$\frac{d^2\mathcal{G}/\mathcal{G}_{\rm tot}}'
                                            r'{d\log T\,d\log n_H}[{\rm dex^{-2}}]$'))
    plt.title('')
    plt.xlim(-5,3)
    plt.ylim(1,8)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.xlabel(r'$\log\,n_H\,[{\rm cm^{-3}}]$')
    plt.ylabel(r'$\log\,T\,[{\rm K}]$')

    ax = plt.subplot(gs2[2])
    netcool =pdf_cool['total']*pdf_cool.attrs['total_cooling']-pdf_heat['total']*pdf_heat.attrs['total_heating']
    netcool /= pdf_cool.attrs['total_cooling']
    netcool.plot(norm=LogNorm(1.e-5,10),ax=ax,cmap=cmap_cool,
                 cbar_kwargs=dict(label=r'$\frac{d^2(\mathcal{L}-\mathcal{G})/\mathcal{L}_{\rm tot}}'
                                  r'{d\log T\,d\log n_H}[{\rm dex^{-2}}]$'))
    (-netcool).plot(norm=LogNorm(1.e-5,10),ax=ax,cmap=cmap_heat,add_colorbar=False)
    plt.title('')
    plt.xlim(-5,3)
    plt.ylim(1,8)
    plt.xlabel(r'$\log\,n_H\,[{\rm cm^{-3}}]$')
    plt.ylabel(r'$\log\,T\,[{\rm K}]$')

    if save:
        if pdf is None:
            plt.savefig(os.path.join(savdir,'{}.{:04d}.coolheat.jointpdf.png'.format(s.problem_id,num)))
        else:
            plt.savefig(os.path.join(savdir,'{}.coolheat.jointpdf.png'.format(s.problem_id)))

    return fig

def draw_Tpdf(s,num,pdf=None,zrange=None,save=True):
    '''draw temperature pdfs of cooling/heating (nH-integrated)
    '''
    savdir = s.get_savdir_pdf(zrange=zrange)
    pdf_cool, pdf_heat = s.get_coolheat_pdf(num,zrange=zrange)
    if 'OIold' in pdf_cool: pdf_cool=pdf_cool.drop_vars('OIold')
    coolkey = set(list(pdf_cool.keys()))-{'total'}
    heatkey = set(list(pdf_heat.keys()))-{'total'}
    if pdf is not None:
        total_cooling = pdf['cool'].sum().data
        total_heating = pdf['heat'].sum().data
        pdf_cool=pdf[list(coolkey)+['total_cooling']].sum(dim='time').rename(total_cooling='total')
        pdf_heat=pdf[list(heatkey)+['total_heating']].sum(dim='time').rename(total_heating='total')
        pdf_cool/=total_cooling
        pdf_heat/=total_heating
        pdf_cool.attrs['total_cooling']=total_cooling
        pdf_heat.attrs['total_heating']=total_heating
    fig,axes = plt.subplots(3,2,figsize=(10,12))
    axes=axes.flatten()
    dy = pdf_cool.nH[1]-pdf_cool.nH[0]
    pdf_cool_T = pdf_cool.sum(dim='nH')*dy
    pdf_heat_T = pdf_heat.sum(dim='nH')*dy
    x = pdf_cool_T['T']

    exclude_cool = {'total'}
    if not s.iCoolH2colldiss: exclude_cool = exclude_cool | {'H2_colldiss'}
    if not s.iCoolH2rovib: exclude_cool = exclude_cool | {'H2_rovib'}
    if not s.iCoolHIcollion: exclude_cool = exclude_cool | {'HI_collion'}

    cool_list = list(set(pdf_cool.keys()) - exclude_cool)
    heat_list = list(set(pdf_heat.keys()) - {'total','PH_H2'})
    plt.sca(axes[0])
    plt.step(x,pdf_cool_T['total'],label='total',where='mid',color='k',lw=1)
    for c in ['CI','CII','OI','H2_rovib','H2_colldiss']:
        ls = ':' if c in exclude_cool else '-'
        plt.step(x,pdf_cool_T[c],label=c,where='mid',ls=ls)

    plt.ylabel(r'$\frac{d\mathcal{L}/\mathcal{L}_{\rm tot}}{d\log T}[{\rm dex^{-1}}]$')

    plt.sca(axes[2])
    plt.step(x,pdf_cool_T['total'],label='total',where='mid',color='k',lw=1)
    for c in ['OII','Rec','HI_Lya','HII_rec']:
        ls = ':' if c in exclude_cool else '-'
        plt.step(x,pdf_cool_T[c],label=c,where='mid',ls=ls)
    plt.ylabel(r'$\frac{d\mathcal{L}/\mathcal{L}_{\rm tot}}{d\log T}[{\rm dex^{-1}}]$')

    plt.sca(axes[4])
    plt.step(x,pdf_cool_T['total'],label='total',where='mid',color='k',lw=1)
    for c in ['HI_collion','HII_ff','CIE_metal','CIE_He']:
        ls = ':' if c in exclude_cool else '-'
        plt.step(x,pdf_cool_T[c],label=c,where='mid',ls=ls)
    plt.ylabel(r'$\frac{d\mathcal{L}/\mathcal{L}_{\rm tot}}{d\log T}[{\rm dex^{-1}}]$')


    plt.sca(axes[1])
    plt.step(x,pdf_heat_T['total'],label='total',where='mid',color='k',lw=1)
    for h in ['PE','CR','H2_pump','H2_diss','H2_form']:
        plt.step(x,pdf_heat_T[h],label=h,where='mid')
    plt.ylabel(r'$\frac{d\mathcal{G}/\mathcal{G}_{\rm tot}}{d\log T}[{\rm dex^{-1}}]$')

    plt.sca(axes[3])
    plt.step(x,pdf_heat_T['total'],label='total',where='mid',color='k',lw=1)
    for h in ['PH_HI','PH_H2']:
        if h in pdf_heat_T:
            plt.step(x,pdf_heat_T[h],label=h,where='mid')
    plt.ylabel(r'$\frac{d\mathcal{G}/\mathcal{G}_{\rm tot}}{d\log T}[{\rm dex^{-1}}]$')


    plt.sca(axes[5])
    plt.step(x,pdf_cool_T['total'],label='total cooling',where='mid',color='tab:blue')
    pdf_cool_sum=pdf_cool_T[cool_list].to_array().sum(dim='variable')
    plt.step(x,pdf_cool_sum,label='cooling sum',where='mid',color='tab:cyan',lw=1,ls='-')

    plt.step(x,pdf_heat_T['total']*pdf_heat.attrs['total_heating']/pdf_cool.attrs['total_cooling'],
             label='total heating',where='mid',color='tab:red')
    pdf_heat_sum=pdf_heat_T[heat_list].to_array().sum(dim='variable')
    plt.step(x,pdf_heat_sum*pdf_heat.attrs['total_heating']/pdf_cool.attrs['total_cooling'],
             label='heating sum',where='mid',color='tab:pink',lw=1,ls='-')

    for ax in axes:
        plt.sca(ax)
        plt.legend(fontsize='x-small')
        plt.xlim(1,7)
        plt.xlabel(r'$\log\,T\,[{\rm K}]$')
        plt.yscale('log')
        plt.ylim(1.e-5,10)
        for T0 in s.get_phase_Tlist():
            plt.axvline(np.log10(T0),ls='--',lw=1,color='gray')
        # plt.tight_layout()
    plt.tight_layout()
    if save:
        if pdf is None:
            plt.savefig(os.path.join(savdir,'{}.{:04d}.coolheat.Tpdf.png'.format(s.problem_id,num)))
        else:
            plt.savefig(os.path.join(savdir,'{}.coolheat.Tpdf.png'.format(s.problem_id)))

    return fig

def draw_sorted_contribution(s,pdf,num,T1=6.e3,T2=1.2e4,phname='warm',thresh=1.e-2,save=True):
    savdir = s.get_savdir_pdf(zrange=None)
    pdf_cool, pdf_heat = s.get_coolheat_pdf(num,zrange=None)
    coolkey = set(list(pdf_cool.keys()))-{'total','OIold'}
    heatkey = set(list(pdf_heat.keys()))-{'total'}
    dT = pdf['T'][1]-pdf['T'][0]
    dnH = pdf.nH[1]-pdf.nH[0]

    warm = (pdf.sel(T=slice(np.log10(T1),np.log10(T2)))*dT).sum(dim='T').sum(dim='nH')*dnH

    vset0 = set.union(coolkey,heatkey)
    heatset = []
    coolset = []

    for v in vset0:
        if (warm[v]/warm['cool']).mean(dim='time').data>thresh:
            if v in coolkey:
                coolset.append(v)
            else:
                heatset.append(v)
    heatset += ['total_heating']
    coolset += ['total_cooling']

    coolset_sorted=np.array(coolset)[warm[coolset].to_array().mean(dim='time').argsort().data[::-1]]
    heatset_sorted=np.array(heatset)[warm[heatset].to_array().mean(dim='time').argsort().data[::-1]]

    fig,axes = plt.subplots(2,2,figsize=(15,10),sharey='row')
    axes = axes.flatten()
    plt.sca(axes[0])
    (warm[coolset_sorted]/warm['cool']).to_array().plot(hue='variable')
    plt.ylabel(r'$\mathcal{L}_{s}/\mathcal{L}_{\rm tot}$')
    plt.yscale('log')
    plt.sca(axes[1])
    plt.boxplot((warm[coolset_sorted]/warm['cool']).to_array().T,labels=coolset_sorted);
    plt.yscale('log')

    plt.sca(axes[2])
    (warm[heatset_sorted]/warm['cool']).to_array().plot(hue='variable')
    plt.ylabel(r'$\mathcal{G}_{s}/\mathcal{L}_{\rm tot}$')
    plt.yscale('log')
    plt.sca(axes[3])
    plt.boxplot((warm[heatset_sorted]/warm['cool']).to_array().T,labels=heatset_sorted);
    plt.yscale('log')

    plt.setp(axes,'ylim',(thresh/10,1))
    plt.suptitle('{}, {}<T<{}'.format(phname,T1,T2))
    plt.tight_layout()

    if save: plt.savefig(os.path.join(savdir,'{}.{}.tevol.png'.format(s.problem_id,phname)))
    return fig
