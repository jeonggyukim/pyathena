import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ..util.units import Units

def projection(sp, dim):
    if dim == 0 or dim == 'z':
        spx = sp['x1']
        spy = sp['x2']
        spz = sp['x3']
    elif dim == 1 or dim == 'y':
        spx = sp['x1']
        spy = sp['x3']
        spz = sp['x2']
    elif dim == 2 or dim == 'x':
        spx = sp['x2']
        spy = sp['x3']
        spz = sp['x1']
    return spx,spy,spz

def projection_v(sp, dim):
    if dim == 0 or dim == 'z':
        spx = sp['v1']
        spy = sp['v2']
        spz = sp['v3']
    elif dim == 1 or dim == 'y':
        spx = sp['v1']
        spy = sp['v3']
        spz = sp['v2']
    elif dim == 2 or dim == 'x':
        spx = sp['v2']
        spy = sp['v3']
        spz = sp['v1']
    return spx, spy, spz

def scatter_sp(sp, ax, dim, cmap=plt.cm.cool_r,
               norm_factor=4., kind='prj', dist_max=50.0,
               marker='o', edgecolors=None, linewidths=None, alpha=1.0,
               kpc=False, runaway=False, agemax=20.0, agemax_sn=40.0,
               plt_old=False, u=None):
    """Function to scatter plot star particles. (From pyathena classic)

    Parameters
    ----------
    sp : DataFrame
        Star particle data
    ax : Axes
        matplotlib axes
    dim : 'x' or 'y' or 'z' (or 0, 1, 2)
        Line-of-sight direction
    norm_factor: float
        Symbol size normalization (bigger for smaller norm_factor)
    kind : 'prj' or 'slc'
        Slice or projection.
    cmap : matplotlib colormap
    dist_max : float
        maximum perpendicular distance from the slice plane (slice only)
    """
    if sp.empty:
        return None

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
            spx, spy, spz = projection(sp_ru_nonsrc, dim)
            spvx, spvy, spvz = projection_v(sp_ru_nonsrc, dim)
            if kpc:
                spx = spx/1.e3
                spy = spy/1.e3
            if kind == 'slc':
                islab=np.where(abs(spz) < dist_max)

            ax.scatter(spx, spy, color='k',
                       marker=marker, edgecolors=edgecolors, linewidths=linewidths,
                       alpha=alpha, s=10.0/norm_factor)

        if len(sp_ru_src) > 0 and runaway:
            spx, spy, spz = projection(sp_ru_src, dim)
            spvx, spvy, spvz = projection_v(sp_ru_src, dim)
            if kpc:
                spx = spx/1.e3
                spy = spy/1.e3
            if kind == 'slc':
                islab=np.where(abs(spz) < dist_max)

            ax.scatter(spx, spy, marker='*', color='r',
                       edgecolors=edgecolors, linewidths=linewidths,
                       alpha=alpha, s=10.0/norm_factor)

        if len(sp_cl) > 0:
            spx, spy, spz = projection(sp_cl, dim)
            if kpc:
                spx = spx/1.e3
                spy = spy/1.e3
            if kind == 'slc':
                xbool = abs(spz) < dist_max

            spm = np.sqrt(sp_cl['mass']*Msun)/norm_factor
            spa = sp_cl['age']*Myr
            if plt_old:
                iyoung = np.where(spa < 1e10)
            else:
                iyoung = np.where(spa < agemax)
                iyoung2 = np.where(np.logical_and(spa >= agemax, spa < agemax_sn))

            if kind == 'slc':
                if plt_old:
                    islab = np.where(xbool*(spa < 1e10))
                else:
                    islab = np.where(xbool*(spa < agemax))
                ax.scatter(spx.iloc[islab], spy.iloc[islab],
                           s=spm.iloc[islab], c=spa.iloc[islab],
                           marker=marker, edgecolors=edgecolors, linewidths=linewidths,
                           vmin=0, vmax=agemax, cmap=cmap, alpha=alpha)

            ax.scatter(spx.iloc[iyoung], spy.iloc[iyoung],
                       s=spm.iloc[iyoung], c=spa.iloc[iyoung],
                       marker=marker, edgecolors=edgecolors, linewidths=linewidths,
                       vmin=0, vmax=agemax, cmap=cmap, alpha=alpha)
            if not plt_old:
                ax.scatter(spx.iloc[iyoung2], spy.iloc[iyoung2],
                           s=spm.iloc[iyoung2], c='grey',
                           marker=marker, edgecolors=edgecolors, linewidths=linewidths,
                           alpha=alpha)


def legend_sp(ax, norm_factor, mass=[1e2, 1e3], location="top", fontsize='medium',
              facecolors='k', linewidths=1.0, bbox_to_anchor=None):
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
                "bbox_to_anchor[top/right] must be a tuple specifying legend location")

    ext = ax.images[0].get_extent()

    ss = []
    labels = []
    for m in mass:
        label = r"$10^{0:g}\;M_\odot$".format(np.log10(m))
        s = ax.scatter(ext[1]*2, ext[3]*2,
                       s=np.sqrt(m)/norm_factor,
                       color='k', alpha=1.0, label=label,
                       linewidths=linewidths, facecolors=facecolors)
        ss.append(s)
        labels.append(label)

    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
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

def colorbar_sp(fig, agemax, cmap=plt.cm.cool_r, bbox=[0.125, 0.9, 0.1, 0.015]):

    # Add starpar age colorbar
    norm = mpl.colors.Normalize(vmin=0., vmax=agemax)
    cax = fig.add_axes(bbox)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal',
                                   ticks=[0, agemax/2.0, agemax], extend='max')

    # cbar_sp.ax.tick_params(labelsize=14)
    cb.set_label(r'${\rm age}\;[{\rm Myr}]$', fontsize=14)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    return cb

    # # Add legends for starpar mass
    # legend_sp(axes[0], norm_factor=1.0, mass=[1e2, 1e3], location='top', fontsize='medium',
    #           bbox_to_anchor=dict(top=(0.22, 0.97), right=(0.48, 0.91)))
