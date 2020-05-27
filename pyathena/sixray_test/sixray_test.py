import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..load_sim import LoadSimAll

def scp_to_pc(source, target='NEWCOOL',
              hostname='kimjg.astro.princeton.edu', username='jgkim'):
    """Function to copy files to my directory
    """
    from paramiko import SSHClient
    from scp import SCPClient

    if target == 'NEWCOOL':
        target = '~/Dropbox/Apps/Overleaf/NEWCOOL/figures'
    elif target == 'GMC-MHD':
        target = '~/Dropbox/Apps/Overleaf/GMC-MHD/figures'
        
    try:
        client = SSHClient()
        client.load_system_host_keys()
        client.connect(hostname,username=username)
        with SCPClient(client.get_transport()) as scp:
            scp.put(source, target)
    finally:
        if client:
            client.close()

def get_data_all(cool=False, dust_model='WD01'):

    # Load data and compute abundances and cooling rates
    models = dict(
        Unshld_CRvar_Z1='/tigress/jk11/NEWCOOL/Unshld.CRvar.Z1/',
        #Unshld_CRvar_Z1='/tiger/scratch/gpfs/jk11/NEWCOOL/Unshld.CRvar.Z1.again/',
        Unshld_CRconst_Z1='/tigress/jk11/NEWCOOL/Unshld.CRconst.Z1/',
        Jeans_CRvar_Z1='/tigress/jk11/NEWCOOL/Jeans.CRvar.Z1/')

    sa = LoadSimAll(models)
    da = dict()
    print('[get_data_all] reading data:', end=' ')
    for i, mdl in enumerate(sa.models):
        print(mdl, end=' ')
        s = sa.set_model(mdl)
        da[mdl] = get_data(s, s.nums[-2], sel_kwargs=dict(z=0, method='nearest'),
                           cool=cool, dust_model=dust_model)

    return sa, da


def get_data(s, num, sel_kwargs=dict(), cool=True, dust_model='WD01'):

    D0 = 5.7e-11 # Dissociation rate for unshielded ISRF [s^-1]
    
    ds = s.load_vtk(num)
    if not 'CR_ionization_rate' in ds.field_list:
        dd = ds.get_field(['nH','nH2','nHI','xH2','xHII','xe',
                           'xHI','xCII','chi_PE_ext',
                           'chi_LW_ext','chi_H2_ext','chi_CI_ext',
                           'T','pok','cool_rate','heat_rate'])
        dd = dd.assign(xi_CR=dd['z']*0.0 + s.par['problem']['xi_CR0'])
    else:
        dd = ds.get_field(['nH','nH2','nHI','xH2','xHII','xe',
                           'xHI','xCII','chi_PE_ext', 'xi_CR',
                           'chi_LW_ext','chi_H2_ext','chi_CI_ext',
                           'T','pok','cool_rate','heat_rate'])
    print('name:',s.basename, end=' ')
    print('time:',ds.domain['time'])
    
    from pyathena.microphysics.cool import \
        get_xCO, heatPE, heatPE_BT94, heatPE_W03,\
        heatCR, heatH2form, heatH2pump, heatH2diss,\
        coolCII, coolOI, coolRec, coolRec_BT94, coolRec_W03,\
        coolLya, coolCI, coolCO

    Z_d = s.par['problem']['Z_dust']
    Z_g = s.par['problem']['Z_gas']
    xCstd = s.par['cooling']['xCstd']
    xOstd = s.par['cooling']['xOstd']
    
    xCO, ncrit = get_xCO(dd.nH, dd.xH2, dd.xCII, Z_d, Z_g,
                         dd['xi_CR'], dd['chi_LW_ext'], xCstd)
    dd['xCO'] = xCO
    dd['ncrit'] = ncrit
    dd['xOI'] = np.maximum(0.0, xOstd*Z_g - dd['xCO'])
    dd['xCI'] = np.maximum(0.0, xCstd*Z_g - dd.xCII - dd.xCO)

    # Set nH and chi_PE as new dimensions
    log_nH = np.log10(dd.sel(z=0,y=0,method='nearest')['nH'].data)
    log_chi_PE = np.log10(dd.sel(z=0,x=0,method='nearest')['chi_PE_ext'].data)
    dd = dd.rename(dict(x='log_nH'))
    dd = dd.assign_coords(dict(log_nH=log_nH))

    dd = dd.rename(dict(y='log_chi_PE'))
    dd = dd.assign_coords(dict(log_chi_PE=log_chi_PE))

    #dd = dd.drop(['nH'])
    #dd = dd.drop(['y'])
    
    # dd = dd.rename(dict(y='log_chi_PE', chi_PE_ext='chi_PE'))
    # dd = dd.assign_coords(dict(log_chi_PE=log_chi_PE))

    # print(sel_kwargs)
    d = dd.sel(**sel_kwargs)

    # Calculate heat/cool rates
    if cool:
        if dust_model == 'BT94':
            d['heatPE'] = heatPE_BT94(d['nH'], d['T'], d['xe'], Z_d, d['chi_PE_ext'])
        elif dust_model == 'W03':
            d['heatPE'] = heatPE_W03(d['nH'], d['T'], d['xe'], Z_d, d['chi_PE_ext'])
        else:
            d['heatPE'] = heatPE(d['nH'], d['T'], d['xe'], Z_d, d['chi_PE_ext'])
        d['heatCR'] = heatCR(d['nH'], d['xe'], d['xHI'], d['xH2'], d['xi_CR'])
        d['heatH2pump'] = heatH2pump(d['nH'], d['T'], d['xHI'], d['xH2'], d['chi_H2_ext']*D0)
        d['heatH2form'] = heatH2form(d['nH'], d['T'], d['xHI'], d['xH2'], Z_d)
        d['heatH2diss'] = heatH2diss(d['xH2'], d['chi_H2_ext']*D0)
        d['coolCII'] = coolCII(d['nH'],d['T'],d['xe'],d['xHI'],d['xH2'],d['xCII'])
        d['coolOI'] = coolOI(d['nH'],d['T'],d['xe'],d['xHI'],d['xH2'],d['xOI'])
        if dust_model == 'BT94':
            d['coolRec'] = coolRec_BT94(d['nH'],d['T'],d['xe'],Z_d,d['chi_PE_ext'])
        elif dust_model == 'W03':
            d['coolRec'] = coolRec_W03(d['nH'],d['T'],d['xe'],Z_d,d['chi_PE_ext'])
        else:
            d['coolRec'] = coolRec(d['nH'],d['T'],d['xe'],Z_d,d['chi_PE_ext'])
        d['coolLya'] = coolLya(d['nH'],d['T'],d['xe'],d['xHI'])
        d['coolCI'] = coolCI(d['nH'],d['T'],d['xe'],d['xHI'],d['xH2'],d['xCI'])
        d['coolCO'] = coolCO(d['nH'],d['T'],d['xe'],d['xHI'],d['xH2'],d['xCO'],3e-14)
        d['cool'] = d['coolCI']+d['coolCII']+d['coolOI']+d['coolRec']+d['coolLya']+d['coolCO']
        d['heat'] = d['heatPE']+d['heatCR'] + d['heatH2pump']
        
        # Note that G_0 is in Habing units
        d['charging'] = 1.7*d['chi_PE_ext']*d['T']**0.5/(d['nH']*d['xe'])
        
    return d

def get_PTn_at_Pminmax(d, j=1, kernel_width=12):
    
    from scipy import interpolate
    from scipy.signal import find_peaks
    from astropy.convolution import Gaussian1DKernel, Box1DKernel, convolve

    x = np.linspace(np.log10(d['nH']).min(),np.log10(d['nH']).max(),1200)

    # Currently, cooling solver produces glitches (should be fixed),
    # so we need to smooth data
    gP = Gaussian1DKernel(kernel_width)
    gT = Gaussian1DKernel(kernel_width)
    #gT = Box1DKernel(15)
    
    Pmin = np.zeros_like(d['log_chi_PE'][::j])
    Pmax = np.zeros_like(d['log_chi_PE'][::j])
    Tmin = np.zeros_like(d['log_chi_PE'][::j])
    Tmax = np.zeros_like(d['log_chi_PE'][::j])
    nmin = np.zeros_like(d['log_chi_PE'][::j])
    nmax = np.zeros_like(d['log_chi_PE'][::j])
    for i, log_chi_PE in enumerate(d['log_chi_PE'].data[::j]):
        dd = d.sel(log_chi_PE=float(log_chi_PE), method='nearest')

        fP = interpolate.interp1d(np.log10(dd['nH']), np.log10(dd['pok']), kind='cubic')
        fT = interpolate.interp1d(np.log10(dd['nH']), np.log10(dd['T']), kind='cubic')
        yP = convolve(fP(x), gP, boundary='fill', fill_value=np.nan)
        yT = convolve(fT(x), gT, boundary='fill', fill_value=np.nan)
        try:
            ind1 = find_peaks(-yP)[0]
            ind2 = find_peaks(yP)[0]
            # print(ind1,ind2)
            if len(ind1) > 1:
                print('Multiple local minimum log_chi,idx:',log_chi_PE,ind1)
                i1 = ind1[0]
            else:
                i1 = ind1[0]
            if len(ind2) > 1:
                print('Multiple local maximum log_chi,idx:',log_chi_PE,ind2)
                i2 = ind2[0]
            else:
                i2 = ind2[0]

            Pmin[i] = 10.0**float(yP[i1])
            Pmax[i] = 10.0**float(yP[i2])
            Tmin[i] = 10.0**float(yT[i1])
            Tmax[i] = 10.0**float(yT[i2])
            nmin[i] = 10.0**float(x[i1])
            nmax[i] = 10.0**float(x[i2])
        except IndexError:
            # print('Failed to find Pmin/Pmax, log_chi_PE:',log_chi_PE)
            pass
        # break
    r = dict()
    r['Pmin'] = Pmin
    r['Pmax'] = Pmax
    r['Tmin'] = Tmin
    r['Tmax'] = Tmax
    r['nmin'] = nmin
    r['nmax'] = nmax
    
    return r

def plt_nP_nT(da, model, savefig=True):
    # Plot equilibrium density pressure and temperature relation
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(-1,3.7)

    fig, axes = plt.subplots(2, 1, figsize=(7, 10), sharex=False)
    axes = axes.flatten()
    dd = da[model]
    log_chi = np.array([-1.0,0.0,1.0,2.0,3.0])
    lw = [1.5,3,1.5,1.5,1.5]
    labels = [r'$\chi=0.1$',r'$\chi=1$',r'$\chi=10$',r'$\chi=10^2$',r'$\chi=10^3$']
    for i,log_chi_PE_ in enumerate(log_chi):
        d_ = dd.sel(log_chi_PE=log_chi_PE_, method='nearest')
        plt.sca(axes[0])
        l, = plt.loglog(d_['nH'], d_['pok'], label=labels[i],
                        c=cmap(norm(log_chi_PE_)), lw=lw[i])
        plt.sca(axes[1])
        plt.loglog(d_['nH'],d_['T'], c=l.get_color(), lw=lw[i])

    plt.sca(axes[0])
    plt.ylim(1e2,1e8)
    plt.sca(axes[1])
    plt.ylim(1e1,3e4)

    for ax in axes:
        ax.set_xlim(1e-2,1e5)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.grid()

    for ax in axes:
        ax.set_xlabel(r'$n_{\rm H}\;[{\rm cm}^{-3}]$')
        
    axes[0].set_ylabel(r'$P/k_{\rm B}\;[{\rm K\,cm^{-3}}]$')
    axes[1].set_ylabel(r'$T\;[{\rm K}]$')

    if model == 'Unshld_CRvar_Z1':
        # Label chi manually using plt.text()
        logchi = [-1,0,1,2,3]
        texts = [r'$0.1$',r'$\chi=1$',r'$10$',r'$10^2$',r'$10^3$']
        xpos = [1, 22, 1e2, 0.8e3, 5e3]
        ypos = [4e2, 4e3, 4.5e4, 7e5, 1.5e7]
        for x,y,text,logchi_ in zip(xpos,ypos,texts,logchi):
            axes[0].text(x, y, text,
                    verticalalignment='bottom', ha='right',
                    # transform=ax.transAxes,
                    color=cmap(norm(logchi_)), fontsize=14)
    else:
        # Label using legend
        axes[0].legend(ncol=3, labelspacing=0.01)# , bbox_to_anchor=(0.85, 1.5)
    
    # Add suptitle
    if model == 'Unshld_CRvar_Z1':
        suptitle = r'$\xi/\chi = 2\times 10^{-16}\,{\rm s}^{-1}$'
    elif model == 'Unshld_CRconst_Z1':
        suptitle = r'$\xi = 2\times 10^{-16}\,{\rm s}^{-1}$'
    else:
        suptitle = model
        
    fig.suptitle(suptitle, x=0.55, ha='center', va='bottom')
    plt.tight_layout()
    
    if savefig:
        savname = '/tigress/jk11/figures/NEWCOOL/paper/fig-equil-{0:s}.png'.format(model)
        plt.savefig(savname, dpi=200, bbox_inches='tight')
        scp_to_pc(savname)
        print('saved to',savname)

    return fig
