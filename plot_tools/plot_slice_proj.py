import glob, os
import numpy as np

import matplotlib.colorbar as colorbar
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, SymLogNorm, NoNorm, Normalize
import pickle

from pyathena import read_starvtk,texteffect,set_units
from .scatter_sp import scatter_sp

unit=set_units(muH=1.4271)
to_Myr=unit['time'].to('Myr').value

def plot_slice_proj(fname_slc, fname_proj, fname_sp, fields_to_draw,
                    savname=None, zoom=1., aux={}, time_stamp=True,
                    fig_zmargin=0.5,
                    sp_norm_factor=2):

    """
    Draw slices and projections
    """
    
    plt.rc('font', size=13)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)

    slc_data = pickle.load(open(fname_slc, 'rb'))
    proj_data = pickle.load(open(fname_proj, 'rb'))

    for x in ('x','y','z'):
        slc_data[x+'extent'] = np.array(slc_data[x+'extent'])/1e3
        slc_data[x+'yextent'] = np.array(slc_data[x+'extent'])/1e3
        slc_data[x+'xextent'] = np.array(slc_data[x+'extent'])/1e3

    # starting position
    x0 = slc_data['xextent'][0]
    y0 = slc_data['xextent'][1]
    Lx = slc_data['yextent'][1] - slc_data['yextent'][0]
    Ly = slc_data['zextent'][1] - slc_data['zextent'][0]
    Lz = slc_data['yextent'][3] - slc_data['yextent'][2]
    #print(x0,y0,Lx,Ly,Lz)
    
    # Set figure size in inches and margins
    Lz = Lz/zoom
    xsize = 3.0
    zsize = xsize*Lz/Lx
    nf = len(fields_to_draw)
    #print(xsize,zsize)
    
    # Need to adjust zmargin depending on number of fields and aspect_ratio
    zfactor = 1.0 + fig_zmargin
    fig = plt.figure(1, figsize=(xsize*nf, zsize + xsize*zfactor))
    gs = gridspec.GridSpec(2, nf, height_ratios=[zsize, xsize])
    gs.update(top=0.95, left=0.10, right=0.95, wspace=0.05, hspace=0)

    # Read starpar and time
    time_sp, sp = read_starvtk(fname_sp, time_out=True)
    if 'time' in slc_data:
        tMyr = slc_data['time']
    else:
        tMyr = time_sp*to_Myr

    # Sanity check
    if np.abs(slc_data['time']/(time_sp*to_Myr) - 1.0) > 1e-7:
        print('[plot_slice_proj]: Check time time_slc, time_sp', tMyr, time_sp*to_Myr)
        #raise
    
    images = []
    for i, axis in enumerate(['y', 'z']):
        for j, f in enumerate(fields_to_draw):
            ax = plt.subplot(gs[i, j])
            if f is 'star_particles': 
                scatter_sp(sp, ax, axis=axis, norm_factor=sp_norm_factor,
                           type='surf')
                # if axis is 'y':
                #     ax.set_xlim(x0, x0 + Lx)
                #     ax.set_ylim(y0, y0 + Lz)
                # if axis is 'z':
                #     ax.set_xlim(x0, x0 + Lx)
                #     ax.set_ylim(x0, x0 + Lx)
                extent = slc_data[axis+'extent']
                print(axis,extent)
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                ax.set_aspect(1.0)
            else:
                if f[-4:] == 'proj':
                    data = proj_data[axis][f[:-5]]
                else:
                    data = slc_data[axis][f]
                im=ax.imshow(data, origin='lower', interpolation='bilinear')
                if f in aux:
                    if 'norm' in aux[f]:
                        im.set_norm(aux[f]['norm']) 
                    if 'cmap' in aux[f]:
                        im.set_cmap(aux[f]['cmap'])
                    if 'clim' in aux[f]:
                        im.set_clim(aux[f]['clim'])

                extent = slc_data[axis+'extent']
                im.set_extent(extent)
                images.append(im)
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])

    for j, (im, f) in enumerate(zip(images, fields_to_draw[1:])):
        ax = plt.subplot(gs[0,j+1])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", "3%", pad="1%")
        cbar = fig.colorbar(im,cax=cax,orientation='horizontal')
        if f in aux:
            if 'label' in aux[f]:
                cbar.set_label(aux[f]['label'])
            if 'cticks' in aux[f]:
                cbar.set_ticks(aux[f]['cticks'])
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top')

    ax=plt.subplot(gs[0,0])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", "3%", pad="1%") 
    cbar = colorbar.ColorbarBase(cax, ticks=[0,20,40],
                                 cmap=plt.cm.cool_r,
                                 norm=Normalize(vmin=0, vmax=40),
                                 orientation='horizontal')
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')
    cbar.set_label(r'${\rm age [Myr]}$')

    s1 = ax.scatter(Lx*2, Lz*2,
                    s=np.sqrt(1.e3)/sp_norm_factor, color='k',
                    alpha=.8, label=r'$10^3 M_\odot$')
    s2 = ax.scatter(Lx*2, Lz*2,
                    s=np.sqrt(1.e4)/sp_norm_factor,color='k',
                    alpha=.8, label=r'$10^4 M_\odot$')
    s3 = ax.scatter(Lx*2, Lz*2,
                    s=np.sqrt(1.e5)/sp_norm_factor,
                    color='k', alpha=.8, label=r'$10^5 M_\odot$')

    #ax.set_xlim(x0, x0 + Lx)
    #ax.set_ylim(y0, y0 + Lz)
    legend = ax.legend((s1, s2, s3),
                       (r'$10^3 M_\odot$', r'$10^4 M_\odot$', r'$10^5 M_\odot$'),
                       scatterpoints = 1, loc='lower left',
                       fontsize='medium', frameon=True)

    axes = fig.axes
    plt.setp([ax.get_xticklabels() for ax in axes[:2*nf]], visible=False)
    plt.setp([ax.get_yticklabels() for ax in axes[:2*nf]], visible=False)
    plt.setp(axes[:nf],'ylim',(slc_data['yextent'][2]/zoom,slc_data['yextent'][3]/zoom))

    plt.setp(axes[nf:2*nf],'xlabel', 'x [kpc]')
    plt.setp(axes[0],'ylabel', 'z [kpc]')
    if time_stamp: 
        ax=axes[0]
        ax.text(0.5, 0.95, 't={0:3d} Myr'.format(int(tMyr)), size=16,
                horizontalalignment='center',
                transform=ax.transAxes, **(texteffect()))
    plt.setp(axes[nf], 'ylabel', 'y [kpc]')
    plt.setp([ax.get_xticklabels() for ax in axes[nf:]], visible=True)
    plt.setp([ax.get_yticklabels() for ax in axes[:2*nf:nf]], visible=True)
    plt.setp([ax.xaxis.get_majorticklabels() for ax in axes[nf:2*nf]], rotation=45)

    
    #pngfname=fname_slc+'ng'
    #canvas = mpl.backends.backend_agg.FigureCanvasAgg(fig)
    #canvas.print_figure(pngfname,num=1,dpi=150,bbox_inches='tight')
    if savname is None:
        return fig
    else:
        plt.savefig(savname, bbox_inches='tight', num=0, dpi=150)
        plt.close()
