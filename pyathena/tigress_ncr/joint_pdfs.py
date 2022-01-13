import numpy as np
import xarray as xr
import astropy.units as au
import astropy.constants as ac
import sys, os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse
import pyathena as pa

def calc_pdf(data,xf,yf,wf=None,logs=dict(),dbins=dict(),ranges=dict()):
    xdata = data[xf].data.flatten()
    ydata = data[yf].data.flatten()
    if wf is None:
        wdata = None
    else:
        wdata = data[wf].data.flatten()
    try:
        if logs[xf]: xdata = np.log10(xdata)
    except:
        pass
    try:
        if logs[yf]: ydata = np.log10(ydata)
    except:
        pass
    if xf in dbins:
        dx = dbins[xf]
    else:
        dx = 0.1
    if yf in dbins:
        dy = dbins[yf]
    else:
        dy = 0.1

    if xf in ranges:
        xmin,xmax = ranges[xf]
    else:
        xmin,xmax = np.percentile(xdata,[5,95])
    if yf in ranges:
        ymin,ymax = ranges[yf]
    else:
        ymin,ymax = np.percentile(ydata,[5,95])
    r = [[xmin,xmax],[ymin,ymax]]
    xbin = int((xmax-xmin)/dx)
    ybin = int((ymax-ymin)/dy)


    h=np.histogram2d(xdata,ydata,bins=[xbin,ybin],range=r,weights=wdata)
    xc = 0.5*(h[1][1:]+h[1][:-1])
    yc = 0.5*(h[2][1:]+h[2][:-1])
    pdf = xr.DataArray(h[0].T/dx/dy,coords=[yc,xc],dims=[yf,xf])

    return pdf

def jointpdf(ds,par,zmin=0,zmax=300,fields='auto',wfields=['nH'],verbose=False):
    if fields == 'auto':
        fields = ['nH','pok','T','vz']
        if par['configure']['radps'] == 'ON':
            fields += ['Lambda_cool','xH2','xHI','xHII','ne']
            if (par['cooling']['iPEheating'] == 1):
                fields += ['chi_FUV']
            if (par['radps']['iPhotIon'] == 1):
                fields += ['Erad_LyC']
    nologs = ['vz']
    ranges = dict(nH=(-5,3),T=(1,8),pok=(0,8),vz=(-500,500),
                  chi_FUV=(-2,4),Erad_LyC=(-30,-10),Uion=(-20,5),
                  Lambda_cool=(-30,-20),xH2=(-5,0),xHI=(-5,0),xHII=(-5,0),ne=(-5,3))
    dbins = dict(vz=10)

    if verbose: print('time = {}'.format(ds.domain['time']))
    data=ds.get_field(fields)
    if 'Erad_LyC' in fields:
        data['Uion']=(data['Erad_LyC'])/(13.6*au.eV).cgs.value/data['nH']
        fields += ['Uion']

    logs = dict()
    for f in fields:
        if f in nologs:
            logs[f] = False
        else:
            logs[f] = True

    data_zcut = xr.concat([data.sel(z=slice(-zmax,-zmin)),data.sel(z=slice(zmin,zmax))],dim='z')
    pdf_dset = xr.Dataset()

    for i,xf in enumerate(fields):
        for yf in fields[i:]:
            if (xf != yf):
                if verbose: print(xf,yf)
                pdf = calc_pdf(data,xf,yf,logs=logs,dbins=dbins,ranges=ranges)
                pdf_dset['{}-{}-vol'.format(xf,yf)] = pdf
                for wf in wfields:
                    pdf = calc_pdf(data,xf,yf,wf=wf,logs=logs,dbins=dbins,ranges=ranges)
                    pdf_dset['{}-{}-{}'.format(xf,yf,wf)] = pdf
    return pdf_dset

def plot_pair(pdf,wf='vol',fields=None):
    if fields is None: fields = list(pdf.dims.keys())
    nf = len(fields)
    fig, axes = plt.subplots(nf,nf,figsize=(4*nf,3*nf))

    # 1D histogram
    for i,xf in enumerate(fields):
        plt.sca(axes[i,i])
        try:
            yf = fields[i+1]
            key = '{}-{}-{}'.format(xf,yf,wf)
        except:
            yf = fields[0]
            key = '{}-{}-{}'.format(yf,xf,wf)
        xc = pdf[xf]
        yc = pdf[yf]
        dy = yc[1]-yc[0]
        plt.step(xc,pdf[key].sum(dim=yf)*dy,where='mid')

        plt.yscale('log')
        plt.ylabel('d{}/dlog{}'.format(wf[0].upper(),xf))
        plt.xlabel(xf)

    # 2D histogram in bottom-left
    keylist=[]
    axlist=[]
    for i,xf in enumerate(fields):
        for j,yf in enumerate(fields):
            key ='{}-{}-{}'.format(xf,yf,wf)
            if key in pdf:
                keylist.append(key)
                axlist.append(axes[j,i])
            else:
                if i != j:
                    axes[j,i].axis('off')

    for key,ax in zip(keylist,axlist):
        xf,yf,wf = key.split('-')
        xc = pdf[xf]
        yc = pdf[yf]
        dx = xc[1]-xc[0]
        dy = yc[1]-yc[0]
        xmin = xc.min() - 0.5*dx
        xmax = xc.max() - 0.5*dx
        ymin = yc.min() - 0.5*dy
        ymax = yc.max() - 0.5*dy
        plt.sca(ax)
        plt.imshow(pdf[key],norm=LogNorm(),extent=[xmin,xmax,ymin,ymax],
                   origin='lower',interpolation='nearest')
        ax.set_aspect('auto')
        plt.xlabel(xf)
        plt.ylabel(yf)

if __name__ == '__main__':
    basedir_def = '/tigress/changgoo/TIGRESS-NCR/R8_8pc_NCR.full'
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--basedir', type=str,
                        default=basedir_def,
                        help='Name of the basedir.')
    args = vars(parser.parse_args())
    locals().update(args)

    s = pa.LoadSimTIGRESSNCR(basedir, verbose=True)
    for num in s.nums:
        ds = s.load_vtk(num)
        for zrange in [(0,300)]:#,(300,1000),(1000,2000)]:
            zmin,zmax = zrange
            savdir = '{}/jointpdf_z{:02d}-{:02d}/'.format(s.savdir,int(zmin/100),int(zmax/100))
            if not os.path.isdir(savdir): os.mkdir(savdir)
            fname = '{}{}.{:04d}.pdf.nc'.format(savdir,s.basename,num)
            if not os.path.isfile(fname):
                print("Creating...",end=" ")
                pdf = jointpdf(ds,s.par,zmin=zmin,zmax=zmax,verbose=False)
                pdf.to_netcdf(fname)
                pdf.close()
            print(fname)
