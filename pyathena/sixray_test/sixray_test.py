import numpy as np

def get_data(s, num, sel_kwargs=dict(z=0.0, method='nearest')):

    ds = s.load_vtk(num)
    dd = ds.get_field(['nH','nH2','nHI','xH2','xHII','xe',
                       'xHI','xCII','chi_PE_ext', 'xi_CR',
                       'chi_LW_ext','chi_H2_ext','chi_CI_ext',
                       'T','pok','cool_rate','heat_rate'])
    print('name:',s.basename, end='  ')
    print('time:',ds.domain['time'])
    
    from pyathena.microphysics.cool import get_xCO, heatPE, heatCR, heatH2pump,\
        coolCII, coolOI, coolRec, coolLya, coolCI, coolCO

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
    
    d = dd.sel(**sel_kwargs)
    d['heatPE'] = heatPE(d['nH'], d['T'], d['xe'], Z_d, d['chi_PE_ext'])
    d['heatCR'] = heatCR(d['nH'], d['xe'], d['xHI'], d['xH2'], d['xi_CR'])
    d['heatH2pump'] = heatH2pump(d['nH'], d['T'], d['xHI'], d['xH2'], d['chi_H2_ext']*5.8e-11)
    d['coolCII'] = coolCII(d['nH'],d['T'],d['xe'],d['xHI'],d['xH2'],d['xCII'])
    d['coolOI'] = coolOI(d['nH'],d['T'],d['xe'],d['xHI'],d['xH2'],d['xOI'])
    d['coolRec'] = coolRec(d['nH'],d['T'],d['xe'],Z_d,d['chi_PE_ext'])
    d['coolLya'] = coolLya(d['nH'],d['T'],d['xe'],d['xHI'])
    d['coolCI'] = coolCI(d['nH'],d['T'],d['xe'],d['xHI'],d['xH2'],d['xCI'])
    d['coolCO'] = coolCO(d['nH'],d['T'],d['xe'],d['xHI'],d['xH2'],d['xCO'],3e-14)
    d['cool'] = d['coolCI']+d['coolCII']+d['coolOI']+d['coolRec']+d['coolLya']+d['coolCO']
    d['heat'] = d['heatPE']+d['heatCR'] + d['heatH2pump']

    return d

def get_PTn_at_Pminmax(d, j=1):
    
    from scipy import interpolate
    from scipy.signal import find_peaks
    from astropy.convolution import Gaussian1DKernel, Box1DKernel, convolve

    x = np.linspace(np.log10(d['nH']).min(),np.log10(d['nH']).max(),1200)
    gP = Gaussian1DKernel(10)
    gT = Gaussian1DKernel(10)
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
                i1 = ind1[1]
            else:
                i1 = ind1[0]
            if len(ind2) > 1:
                print('Multiple local maximum log_chi,idx:',log_chi_PE,ind2)
                i2 = ind2[1]
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
