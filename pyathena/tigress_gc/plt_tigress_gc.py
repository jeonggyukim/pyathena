"""
*This module is not intended to be used as a script*
Putting a script inside a module's directory is considered as an antipattern
(see rejected PEP 3122).
You are encouraged to write a seperate script that executes the functions in
this module. - SMOON
"""
import os
import time
import os.path as osp
import matplotlib as mpl
import matplotlib.pyplot as plt

def plt_proj_density(s, num, dat=None, gs=None, savfig=True):

    if dat is None:
        ds = s.load_vtk(num=num)
        dat = ds.get_field(field='density', as_xarray=True)
    else:
        ds = s.load_vtk(num=num)
       
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    dat['surface_density'] = (dat['density']*s.u.Msun/s.u.pc**3
            *ds.domain['dx'][2]*s.u.pc).sum(dim='z')
    dat['surface_density'].plot.imshow(ax=ax, norm=mpl.colors.LogNorm(),
            cmap='pink_r', vmin=1e0, vmax=1e4)
    ax.set_aspect('equal')
    plt.suptitle('{0:s}, time: {1:.1f} Myr'.format(s.name, ds.domain['time']*s.u.Myr))
    
    if savfig:
        savdir = osp.join('./figures-proj')
        if not os.path.exists(savdir):
            os.makedirs(savdir)
        plt.savefig(osp.join(savdir, 'proj-density.{0:s}.{1:04d}.png'
            .format(s.name, ds.num)),bbox_inches='tight')
    
    return plt.gcf()
