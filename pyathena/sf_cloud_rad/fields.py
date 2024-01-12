# fields.py

import numpy as np

class Fields():

    def get_chi(self, ds, fields='nH', freq=['LW','PE']):

        s = self
        par = s.par
        freq = np.atleast_1d(freq)
        fields = list(np.atleast_1d(fields))
        if not 'nH' in fields:
            fields.append('nH')

        sigmad = dict()
        for f in freq:
            fields.append('chi_{0:s}'.format(f))
            sigmad[f] = par['opacity']['sigma_dust_{0:s}0'.format(f)]*\
                        s.par['problem']['Z_dust']

        dd = ds.get_field(fields)

        chi_ext = dict()
        dims = ['x','y','z']
        dtoi = dict(x=0,y=1,z=2)

        for f in freq:
            tmp = dict()
            for dim in dims:
                dx_cgs = s.domain['dx'][dtoi[dim]]*s.u.length.cgs.value
                chi0p = s.par['sixray'][r'chi_x{0:d}p'.format(dtoi[dim]+1)]
                chi0m = s.par['sixray'][r'chi_x{0:d}m'.format(dtoi[dim]+1)]
                # print(dim, chi0p, chi0m, end=' ')

                dd['dtau'] = (dd['nH']*sigmad[f]*dx_cgs)
                dd['taup'] = dd['nH'].cumsum(dim=dim)*sigmad[f]*dx_cgs
                dd['taum'] = dd.sel(**{dim:dd[dim].max()})['taup'] - dd['taup']
                dd['taup'] = dd['taup'].shift(**{dim:1, 'fill_value':0.0})
                tmp[dim] = chi0p*np.exp(-dd['taup'])*(1.0 - np.exp(-dd['dtau']))/dd['dtau'] + \
                       chi0m*np.exp(-dd['taum'])*(1.0 - np.exp(-dd['dtau']))/dd['dtau']

            chi_ext[f] = tmp['x'] + tmp['y'] + tmp['z']
            dd[f'chi_{f}_ext'] = chi_ext[f]

        dd = dd.drop(['dtau','taup','taum'])

        if 'LW' in freq and 'PE' in freq:
            Erad_PE0 = s.par['cooling']['Erad_PE0']
            Erad_LW0 = s.par['cooling']['Erad_LW0']
            dd['chi_FUV_ext'] = (Erad_PE0*dd['chi_PE_ext'] +
                                 Erad_LW0*dd['chi_LW_ext'])/(Erad_PE0 + Erad_LW0)
            dd['chi_FUV'] = (Erad_PE0*dd['chi_PE'] +
                             Erad_LW0*dd['chi_LW'])/(Erad_PE0 + Erad_LW0)

        return dd
