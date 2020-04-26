import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ..util.units import Units

def projection(sp, axis):
    if axis == 0 or axis == 'z':
        spx = sp['x1']
        spy = sp['x2']
        spz = sp['x3']
    elif axis == 1 or axis == 'y':
        spx = sp['x1']
        spy = sp['x3']
        spz = sp['x2']
    elif axis == 2 or axis == 'x':
        spx = sp['x2']
        spy = sp['x3']
        spz = sp['x1']
    return spx,spy,spz

def projection_v(sp, axis):
    if axis == 0 or axis == 'z':
        spx = sp['v1']
        spy = sp['v2']
        spz = sp['v3']
    elif axis == 1 or axis == 'y':
        spx = sp['v1']
        spy = sp['v3']
        spz = sp['v2']
    elif axis == 2 or axis == 'x':
        spx = sp['v2']
        spy = sp['v3']
        spz = sp['v1']
    return spx, spy, spz

def scatter(sp, ax, u=None, axis=0, thickness=10.0, cmap=plt.cm.winter,
            norm_factor=4., kind='proj', kpc=True, runaway=True, agemax=40.0, plt_old=True):
    """Function to scatter plot star particles. (From pyathena classic)

    """

    if u is None:
        u = Units(kind='LV', muH=1.4271)

    Msun = u.Msun
    Myr = u.Myr
    
    if len(sp) > 0:
        runaways = (sp['mass'] == 0.0)
        # Clusters
        sp_cl = sp[~runaways]
        # Runaways
        sp_ru = sp[runaways]
        # Sources have negative values of id
        src_ru = (sp_ru['id'] < 0)
        sp_ru_src = sp_ru[src_ru]
        sp_ru_nonsrc = sp_ru[~src_ru]

        if len(sp_ru_nonsrc) > 0 and runaway:
            spx, spy, spz = projection(sp_ru_nonsrc, axis)
            spvx, spvy, spvz = projection_v(sp_ru_nonsrc, axis)
            if kpc:
                spx = spx/1.e3
                spy = spy/1.e3
            if kind == 'slice':
                islab=np.where(abs(spz) < thickness)

            ax.scatter(spx, spy, marker='o', color='k', alpha=1.0, s=10.0/norm_factor)

        if len(sp_ru_src) > 0 and runaway:
            spx, spy, spz = projection(sp_ru_src, axis)
            spvx, spvy, spvz = projection_v(sp_ru_src, axis)
            if kpc:
                spx = spx/1.e3
                spy = spy/1.e3
            if kind == 'slice':
                islab=np.where(abs(spz) < thickness)

            ax.scatter(spx, spy, marker='*', color='r', alpha=1.0, s=10.0/norm_factor)
        
        if len(sp_cl) > 0:
            spx, spy, spz = projection(sp_cl, axis)
            if kpc:
                spx = spx/1.e3
                spy = spy/1.e3
            if kind == 'slice':
                xbool = abs(spz) < thickness

            spm = np.sqrt(sp_cl['mass']*Msun)/norm_factor
            spa = sp_cl['age']*Myr
            if plt_old:
                iyoung = np.where(spa < 1e10)
            else:
                iyoung = np.where(spa < agemax)

            if kind == 'slice':
                if plt_old:
                    islab = np.where(xbool*(spa < 1e10))
                else:
                    islab = np.where(xbool*(spa < agemax))                    
                ax.scatter(spx.iloc[islab], spy.iloc[islab], marker='o',
                           s=spm.iloc[islab], c=spa.iloc[islab],
                           vmin=0, vmax=agemax, cmap=cmap, alpha=1.0)

            ax.scatter(spx.iloc[iyoung], spy.iloc[iyoung], marker='o',
                       s=spm.iloc[iyoung], c=spa.iloc[iyoung],
                       vmin=0, vmax=agemax, cmap=cmap, alpha=0.7)


def legend(ax, norm_factor, mass=[1e2, 1e3], location="top", fontsize='medium',
           bbox_to_anchor=None):
    """Add legend for sink particle mass.
    
    Parameters
    ----------
    ax : matplotlib axes
    norm_factor : float
        Normalization factor for symbol size.
    mass: sequence of float
        Sink particle masses in solar mass.
    location: str
        "top" or "right"
    """
    
    if bbox_to_anchor is None:
        bbox_to_anchor = dict(top=(0.1, 0.95),
                              right=(0.88, 0.83))
    else:
        if location not in bbox_to_anchor:
            raise(
                "bbox_to_anchor[localtion] must be a tuple specifying legend location")

    ext = ax.images[0].get_extent()

    ss = []
    labels = []
    for m in mass:
        label = r"$10^{0:g}\;M_\odot$".format(np.log10(m))
        s = ax.scatter(ext[1]*2, ext[3]*2,
                       s=np.sqrt(m)/norm_factor, lw=1.0,
                       color='k', alpha=1.0, label=label, facecolors='k')
        ss.append(s)
        labels.append(label)
    
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[3], ext[2])
    if location == 'top':
        legend = ax.legend(ss, labels, scatterpoints=1, fontsize=fontsize,
                           ncol=len(ss), frameon=False, loc=2,
                           bbox_to_anchor=bbox_to_anchor[location],
                           bbox_transform=plt.gcf().transFigure,
                           columnspacing=0.1, labelspacing=0.1, handletextpad=0.05)
        # for t in legend.get_texts():
        #         t.set_va('bottom')
    else:
        legend = ax.legend(ss, labels, scatterpoints=1, fontsize=fontsize,
                           frameon=False, loc=2,
                           columnspacing=0.02, labelspacing=0.02, handletextpad=0.02,
                           bbox_to_anchor=bbox_to_anchor[location],
                           bbox_transform=plt.gcf().transFigure)

    return legend

def colorbar(fig, agemax, cmap=plt.cm.winter, bbox=[0.125, 0.9, 0.1, 0.015]):

    # Add starpar age colorbar
    norm = mpl.colors.Normalize(vmin=0., vmax=agemax)
    cmap = mpl.cm.winter
    cax = fig.add_axes(bbox)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal',
                                   ticks=[0, agemax/2.0, agemax], extend='max')

    # cbar_sp.ax.tick_params(labelsize=14)
    cb.set_label(r'${\rm age}\;[{\rm Myr}]$', fontsize=14)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    # # Add legends for starpar mass
    # legend_sp(axes[0], norm_factor=1.0, mass=[1e2, 1e3], location='top', fontsize='medium',
    #           bbox_to_anchor=dict(top=(0.22, 0.97), right=(0.48, 0.91)))
    