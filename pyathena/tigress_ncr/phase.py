import matplotlib as mpl
import numpy as np
import xarray as xr
import astropy.units as au
import astropy.constants as ac
import sys, os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse
import cmasher as cmr

sys.path.insert(0,'../../')

import pyathena as pa
import gc

def print_blue(txt):
    print('\033[34m', txt, '\033[0m', sep='', end=' ')
def print_red(txt):
    print('\033[31m', txt, '\033[0m', sep='', end=' ')
# recalculate PDFs with finer bins
def recal_nP(dchunk):
    dset = xr.Dataset()
    for xs in [None,'xHI','xHII']:
        for wf in ['vol','nH','net_cool_rate']:
            xbins=np.logspace(-6,6,601)
            ybins=np.logspace(0,10,501)
            if xs is None:
                cond = dchunk['nH']>0.0
            elif xs == 'xHI':
                cond = dchunk['xHII']<0.5
            elif xs == 'xHII':
                cond = dchunk['xHII']>0.5

            x=dchunk['nH'].where(cond).stack(xyz=['x','y','z']).dropna(dim='xyz').data
            y=dchunk['pok'].where(cond).stack(xyz=['x','y','z']).dropna(dim='xyz').data
            if wf is 'vol':
                w=None
            else:
                w=dchunk[wf].where(cond).stack(xyz=['x','y','z']).dropna(dim='xyz').data
            h,b1,b2=np.histogram2d(x,y,weights=w,
                                   bins=[xbins,ybins])
            dx = np.log10(b1[1]/b1[0])
            dy = np.log10(b2[1]/b2[0])
            pdf = h.T/dx/dy
            xbins = np.log10(xbins)
            ybins = np.log10(ybins)

            if wf == 'net_cool_rate':
                total = dchunk['cool_rate'].sum().data
            elif wf == 'vol':
                total = np.prod(dchunk['nH'].shape)
            else:
                total = dchunk[wf].sum().data
            da = xr.DataArray(pdf,coords=[0.5*(ybins[1:]+ybins[:-1]),0.5*(xbins[1:]+xbins[:-1])],dims=['pok','n_H'])
            dset['{}-{}'.format('all' if xs is None else xs,wf)] = da
            dset=dset.assign_coords({wf:total})
    return dset.assign_coords(time=dchunk.time)

def recal_xT(dchunk):
    hist_bin=[]
    T=['T','T1']
    xs=['xHI','xHII']
    log=[False,True]
    for T_ in T:
        for log_ in log:
            for xs_ in xs:
                x,y,w=np.log10(dchunk[T_].data.flatten()),dchunk[xs_].data.flatten(),dchunk['nH'].data.flatten()
                yr=[0,1]
                if log_:
                    y=np.log10(y)
                    yr=[-6,0]
                hist_bin.append(np.histogram2d(x,y,weights=w,range=[[1,8],yr],bins=[300,150]))

    return hist_bin

def add_phase_cuts(Tlist = [500,6000,15000,35000,5.e5],xs='xHI',xs_axis='y',T1=False,
                   xHIIcut=0.5,xH2cut=0.25,xHI_CIE=True,log=False):
    phcolors=get_phcolor_dict(cmap=None if T1 else cmr.pride,T1=T1,
                              cmin=0.1 if T1 else 0.1,cmax=0.9 if T1 else 0.8)
    # deviding lines
    lkwargs=dict(color='b',ls='--',lw=1)
    for T0 in Tlist:
        ymin=0
        ymax=1
        if (T0 == Tlist[0]) and (not T1):
            ymin=0.5
            if log: ymin = (6+np.log10(ymin))/6.
        else: ymin=0

        if (T0 == Tlist[2]) and (not T1):
            ymax=0.5
            if log: ymax = (6+np.log10(ymax))/6.
        else: ymax=1

        if xs=='xHI':
            if xs_axis=='y':
                plt.axvline(np.log10(T0),ymin=ymin,ymax=ymax,**lkwargs)
            elif xs_axis=='x':
                plt.axhline(np.log10(T0),xmax=ymax,**lkwargs)
        elif xs=='xHII':
            if xs_axis=='y':
                plt.axvline(np.log10(T0),ymin=1-ymax,ymax=1-ymin,**lkwargs)
            elif xs_axis=='x':
                plt.axhline(np.log10(T0),xmin=ymin,**lkwargs)
    # abundance cuts
    xHIIcut=0.5
    xH2cut=0.25
    if T1:
        pass
    else:
        if xs=='xHI':
            ion_cut = 1-np.array([xHIIcut,xHIIcut])
            mol_cut = 1-2.0*np.array([xH2cut,xH2cut])
            if log:
                ion_cut = np.log10(ion_cut)
                mol_cut = np.log10(mol_cut)
            if xs_axis=='y':
                plt.plot([np.log10(Tlist[1]),np.log10(Tlist[3])],ion_cut,**lkwargs)
                plt.plot([1,np.log10(Tlist[1])],mol_cut,**lkwargs)
            elif xs_axis=='x':
                plt.plot(ion_cut,[np.log10(Tlist[1]),np.log10(Tlist[3])],**lkwargs)
                plt.plot(mol_cut,[1,np.log10(Tlist[1])],**lwargs)
        elif xs =='xHII':
            ion_cut = np.array([xHIIcut,xHIIcut])
            if log: ion_cut = np.log10(ion_cut)
            if xs_axis=='y':
                plt.plot([0,np.log10(Tlist[3])],ion_cut,**lkwargs)
            elif xs_axis=='x':
                plt.plot(ion_cut,[0,np.log10(Tlist[3])],**lkwargs)
        elif xs == 'xH2':
            mol_cut = np.array([xH2cut,xH2cut])
            if log: mol_cut = np.log10(mol_cut)
            if xs_axis=='y':
                plt.plot([0,np.log10(Tlist[3])],mol_cut,**lkwargs)
            elif xs_axis=='x':
                plt.plot(mol_cut,[0,np.log10(Tlist[3])],**lkwargs)

    # annotate
    yl = 0.05
    yr = 0.95
    if log:
        yl = -5.95
        yr = np.log10(yr)
    if xs == 'xHI':
        label_infos=dict()
        label_infos['HIM']=dict(xy=(np.log10(1.1*Tlist[4]),yr),va='top',ha='left')
        label_infos['WHIM']=dict(xy=(np.log10(0.9*Tlist[4]),yr),va='top',ha='right')
        label_infos['WCIM']=dict(xy=(np.log10(0.9*Tlist[3]),yl),va='bottom',ha='right',rotation=90)
        label_infos['WPIM']=dict(xy=(np.log10(1.1*Tlist[1]),yl),va='bottom',ha='left',rotation=90)
        label_infos['UIM']=dict(xy=(np.log10(0.9*Tlist[1]),yl),va='bottom',ha='right')
        if not log: label_infos['WNM']=dict(xy=(np.log10(0.9*Tlist[3]),yr),va='top',ha='right',rotation=90)
        label_infos['UNM']=dict(xy=(np.log10(0.9*Tlist[1]),yr),va='top',ha='right')
        label_infos['CNM']=dict(xy=(np.log10(20),yr),va='top',ha='left')
        label_infos['CMM']=dict(xy=(np.log10(20),yl),va='bottom',ha='left',label='CMM')
        if T1:
            if not log: label_infos.pop('WNM')
            label_infos.pop('CMM')
            label_infos.pop('UIM')
            label_infos['WNM'] = label_infos.pop('WPIM')
            label_infos['WIM'] = label_infos.pop('WCIM')

    elif xs == 'xHII':
        label_infos=dict()
        label_infos['HIM']=dict(xy=(np.log10(1.1*Tlist[4]),yl),va='bottom',ha='left')
        label_infos['WHIM']=dict(xy=(np.log10(0.9*Tlist[4]),yl),va='bottom',ha='right')
        if not log:
            label_infos['WCIM']=dict(xy=(np.log10(0.9*Tlist[3]),yr),va='top',ha='right',rotation=90)
            label_infos['WPIM']=dict(xy=(np.log10(1.1*Tlist[1]),yr),va='top',ha='left',rotation=90)
        label_infos['UIM']=dict(xy=(np.log10(0.9*Tlist[1]),yr),va='top',ha='right')
        label_infos['WNM']=dict(xy=(np.log10(0.9*Tlist[3]),yl),va='bottom',ha='right',rotation=90)
        label_infos['UNM']=dict(xy=(np.log10(0.9*Tlist[1]),yl),va='bottom',ha='right')
        label_infos['CNM']=dict(xy=(np.log10(0.9*Tlist[0]),yl),va='bottom',ha='right',label='CM+NM')
        if T1:
            if not log:
                label_infos.pop('WNM')
                label_infos['WNM'] = label_infos.pop('WPIM')
                label_infos['WIM'] = label_infos.pop('WCIM')
            else:
                label_infos['WIM']=dict(xy=(np.log10(0.9*Tlist[3]),yl),va='bottom',ha='right',rotation=90)
                label_infos['WNM']=dict(xy=(np.log10(1.1*Tlist[1]),yl),va='bottom',ha='left',rotation=90)
            label_infos.pop('UIM')

    for k in label_infos:
        info = label_infos[k]
        if 'label' in info: l = info.pop('label')
        else: l = k
        if xs_axis=='x':
            info['xy']=(info['xy'][1],info['xy'][0])
            va='bottom' if info['ha'] == 'left'  else 'top'
            ha='left' if info['va'] == 'bottom'  else 'right'
            info['va']=va
            info['ha']=ha
            info['rotation']=0

        fg='k' if k in ['WNM','WIM','UIM','WCIM','WPIM'] else 'w'
        te=pa.classic.texteffect(fontsize='xx-small',foreground=fg,linewidth=2)
        plt.annotate(l,color=phcolors[k],weight='bold',**info,**te)


    #
    if xHI_CIE:
        T = np.logspace(3,6,100)
        kcoll = pa.microphysics.cool.coeff_kcoll_H(T)
        krec = pa.microphysics.cool.coeff_alpha_rr_H(T)
        xHI = krec/(kcoll+krec)
        if T1: logT = np.log10(T*(2.1-xHI))
        else: logT=np.log10(T)
        if xs=='xHII': xHI = 1-xHI
        if log: xHI = np.log10(xHI)
        if xs_axis=='y':
            plt.plot(logT, xHI, c='r', ls='--', label='CIE',lw=1,dashes=[6, 2])
        elif xs_axis=='x':
            plt.plot(xHI, logT, c='r', ls='--', label='CIE',lw=1,dashes=[6, 2])

def get_dchunk(s,num,scratch_dir='/scratch/gpfs/changgoo/TIGRESS-NCR/'):
    scratch_dir += os.path.join(s.basename,'midplane_chunk')
    chunk_file = os.path.join(scratch_dir,'{:s}.{:04d}.nc'.format(s.problem_id,num))
    if not os.path.isfile(chunk_file):
        raise IOError("File does not exist: {}".format(chunk_file))
    with xr.open_dataset(chunk_file) as chunk:
        chunk['Uion']=chunk['Erad_LyC']/((s.par['radps']['hnu_PH']*au.eV).cgs.value*chunk['nH'])
        chunk['xHII']=1-chunk['xHI']-chunk['xH2']*2
        chunk['T1']=chunk['pok']/chunk['nH']
    return chunk,scratch_dir

def define_phase(s,kind='full',verbose=False):
    """Phase definition

    kind : str
      'full' for full 9 phases
      'four' for CU, WNM, WIM, Hot
      'five1' for CNM, UNM, WNM, WIM, Hot
      'five2' for CU, WNM, WIM, WHIM, HIM
      'six' for CNM, UNM, WNM, WIM, WHIM, HIM
    """

    pcdict=get_phcolor_dict(cmap=cmr.pride,cmin=0.0,cmax=0.85)
    phdef=[]
    Tlist=list(np.concatenate([[0],s.get_phase_Tlist(),[np.inf]]))
    i=1
    if kind in ['four','five2']:
        # cold+unstable
        phdef.append(dict(idx=i,name='CU',Tmin=Tlist[0],Tmax=Tlist[2],abundance=None,amin=0.0,c=pcdict['CNM']))
    elif kind == 'full':
        # cold molecular (xH2>0.25, T<6000)
        phdef.append(dict(idx=i,name='CMM',Tmin=Tlist[0],Tmax=Tlist[2],abundance='xH2',amin=0.25,c=pcdict['CMM']))
        # cold neutral (xHI>0.5, T<500)
        i+=1
        phdef.append(dict(idx=i,name='CNM',Tmin=Tlist[0],Tmax=Tlist[1],abundance='xHI',amin=0.5,c=pcdict['CNM']))
        # unstable neutral (xHI>0.5, 500<T<6000)
        i+=1
        phdef.append(dict(idx=i,name='UNM',Tmin=Tlist[1],Tmax=Tlist[2],abundance='xHI',amin=0.5,c=pcdict['UNM']))
        # unstable ionized (xHII>0.5, T<6000)
        i+=1
        phdef.append(dict(idx=i,name='UIM',Tmin=Tlist[0],Tmax=Tlist[2],abundance='xHII',amin=0.5,c=pcdict['UIM']))
    else:
        # cold
        phdef.append(dict(idx=i,name='CNM',Tmin=Tlist[0],Tmax=Tlist[1],abundance=None,amin=0.0,c=pcdict['CNM']))
        # Unstable
        i+=1
        phdef.append(dict(idx=i,name='UNM',Tmin=Tlist[1],Tmax=Tlist[2],abundance=None,amin=0.0,c=pcdict['UNM']))

    # warm neutral (xHI>0.5, 6000<T<35000)
    i+=1
    phdef.append(dict(idx=i,name='WNM',Tmin=Tlist[2],Tmax=Tlist[4],abundance='xHI',amin=0.5,c=pcdict['WNM']))

    if kind == 'full':
        # warm photo-ionized (xHII>0.5, 6000<T<15000)
        i+=1
        phdef.append(dict(idx=i,name='WPIM',Tmin=Tlist[2],Tmax=Tlist[3],abundance='xHII',amin=0.5,c=pcdict['WPIM']))
        # warm collisonally-ionized (xHII>0.5, 15000<T<35000)
        i+=1
        phdef.append(dict(idx=i,name='WCIM',Tmin=Tlist[3],Tmax=Tlist[4],abundance='xHII',amin=0.5,c=pcdict['WCIM']))
    else:
        # combined warm ionzied
        i+=1
        phdef.append(dict(idx=i,name='WIM',Tmin=Tlist[2],Tmax=Tlist[4],abundance='xHII',amin=0.5,c=pcdict['WPIM']))

    if kind in ['four','five1']:
        # hot ionzied
        i+=1
        phdef.append(dict(idx=i,name='Hot',Tmin=Tlist[4],Tmax=Tlist[6],abundance=None,amin=0.0,c=pcdict['HIM']))
    else:
        # warm-hot ionized (35000<T<5.e5)
        i+=1
        phdef.append(dict(idx=i,name='WHIM',Tmin=Tlist[4],Tmax=Tlist[5],abundance=None,amin=0.0,c=pcdict['WHIM']))
        # hot ionized (5.e5<T)
        i+=1
        phdef.append(dict(idx=i,name='HIM',Tmin=Tlist[5],Tmax=Tlist[6],abundance=None,amin=0.0,c=pcdict['HIM']))

    if verbose:
        for ph in phdef:
            T1 = ph['Tmin']
            T2 = ph['Tmax']
            i = ph['idx']
            a = ph['abundance']
            amin = ph['amin']
            print('{:5s}'.format(ph['name']),'{:3d}'.format(i),'{:>10}'.format(T1),'<T<','{:10}'.format('{}'.format(T2)),
                  '{:^10s}'.format('{:4s}>{}'.format(a,amin) if a is not None else '...'))
    return phdef

def assign_phase(s,dslc,kind='full',verbose=False):
    from matplotlib.colors import ListedColormap
    phslc = xr.zeros_like(dslc['nH']).astype('int')-1
    phslc.name='phase'

    phdef=define_phase(s,kind=kind)
    phlist = []
    phcmap = []
    for ph in phdef:
        T1 = ph['Tmin']
        T2 = ph['Tmax']
        i = ph['idx']
        a = ph['abundance']
        amin = ph['amin']
        cond = (dslc['T']>T1)*(dslc['T']<=T2)
        phlist.append(ph['name'])
        phcmap.append(ph['c'])
        print(ph['name'],i,T1,T2,a,amin)
        if a is not None: cond *= (dslc[a]>amin)
        phslc += cond*i
    phslc.attrs['phdef']=phdef
    phslc.attrs['phlist']=phlist
    phslc.attrs['phcmap']=ListedColormap(phcmap)

    return phslc

def get_phcmap(T1=False,cmin=0,cmax=1,cmap=None):
    if T1:
        phlist=['CNM','UNM','WNM','WIM','WHIM','HIM']
    else:
        phlist=['CMM','CNM','UNM','UIM','WNM','WPIM','WCIM','WHIM','HIM']
    if cmap is None:
        from pyathena.plt_tools import cmap
        cmap=cmap.get_cmap_jh_colors()
    nph=len(phlist)
    phcmap=cmr.get_sub_cmap(cmap, cmin, cmax, N=nph)

    return phlist,phcmap

def get_phcolor_dict(cmap=None,T1=False,cmin=0,cmax=1):
    phlist,phcmap =get_phcmap(cmap=cmap,T1=T1,cmin=cmin,cmax=cmax)
    phcolors=dict()
    for phname,c in zip(phlist,phcmap.colors):
        phcolors[phname]=c
    return phcolors

def draw_phase(ph):
    phlist,phcmap=get_phcmap()
    nph=len(phlist)
    image_style={'axes.grid':False,'image.interpolation':'nearest','ytick.minor.visible':False}
    with mpl.rc_context(image_style):
        ph.plot(cmap=phcmap,vmin=0,vmax=nph,cbar_kwargs=dict(extend=None))
        axes=plt.gcf().axes
        axes[1].get_yaxis().set_ticks([])
        for j, lab in enumerate(phlist):
            color='w' if lab in ['CMM','CNM','HIM'] else 'k'
            axes[1].annotate(lab,(.5, (2*j+1)/(2*nph)), xycoords='axes fraction',color=color,
                             ha='center', va='center',rotation=90,fontsize='x-small')
        axes[1].set_ylabel('Phase')
        axes[0].set_aspect('equal')

if __name__ == '__main__':
    s = pa.LoadSimTIGRESSNCR('/tigress/changgoo/TIGRESS-NCR/R8_4pc_NCR.full/', verbose=True)

    for num in s.nums:
        try:
            dchunk,scratch_dir=get_dchunk(s,num)
        except IOError:
            continue

        phase_dir = scratch_dir.replace('midplane_chunk','phase')
        if not os.path.isdir(phase_dir): os.makedirs(phase_dir)

        phase_file = os.path.join(phase_dir,'{:s}.{:04d}.phase.nc'.format(s.basename,num))

        ph = define_phase(s,dchunk)

        ph.to_netcdf(phase_file)
        ph.close()
        print(phase_file)
