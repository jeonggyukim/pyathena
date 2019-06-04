import numpy as np
import matplotlib.pyplot as plt
from pyathena import read_starvtk, texteffect, set_units

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

def scatter_sp(sp, ax, axis=0, thickness=10.0, norm_factor=4.,
               type='slice', kpc=True, runaway=True, agemax=40.0):
    """
    Function to scatter plot star particles
    """
    
    unit = set_units(muH=1.4271)
    Msun = unit['mass'].to('Msun').value
    Myr = unit['time'].to('Myr').value

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
            if type == 'slice':
                islab=np.where(abs(spz) < thickness)

            ax.scatter(spx, spy, marker='o', color='k', alpha=1.0, s=10.0/norm_factor)

        if len(sp_ru_src) > 0 and runaway:
            spx, spy, spz = projection(sp_ru_src, axis)
            spvx, spvy, spvz = projection_v(sp_ru_src, axis)
            if kpc:
                spx = spx/1.e3
                spy = spy/1.e3
            if type == 'slice':
                islab=np.where(abs(spz) < thickness)

            ax.scatter(spx, spy, marker='*', color='r', alpha=1.0, s=10.0/norm_factor)
        

        if len(sp_cl) > 0:
            spx, spy, spz = projection(sp_cl, axis)
            if kpc:
                spx = spx/1.e3
                spy = spy/1.e3
            if type == 'slice':
                xbool = abs(spz) < thickness

            spm = np.sqrt(sp_cl['mass']*Msun)/norm_factor
            spa = sp_cl['age']*Myr
            iyoung = np.where(spa < 40.)

            if type == 'slice':
                islab = np.where(xbool*(spa < agemax))
                ax.scatter(spx.iloc[islab], spy.iloc[islab], marker='o',
                           s=spm.iloc[islab], c=spa.iloc[islab],
                           vmin=0, vmax=agemax, cmap=plt.cm.cool_r, alpha=1.0)

            ax.scatter(spx.iloc[iyoung], spy.iloc[iyoung], marker='o',
                       s=spm.iloc[iyoung], c=spa.iloc[iyoung],
                       vmin=0, vmax=agemax, cmap=plt.cm.cool_r, alpha=0.7)

            
