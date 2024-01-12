# virial.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as au
import astropy.constants as ac

from ..load_sim import LoadSim

class Outflow:

    @LoadSim.Decorators.check_pickle
    def read_outflow_all(self, nums=None, prefix='outflow_all',
                         savdir=None, force_override=False):
        rr = dict()
        if nums is None:
            nums = self.nums

        for i,num in enumerate(nums):
            print(i, end=' ')
            r = self.read_outflow(num=num, force_override=False)
            if i == 0:
                for k in r.keys():
                    rr[k] = []

            for k in r.keys():
                try:
                    rr[k].append(r[k].value.item())
                except:
                    rr[k].append(r[k])

        rr = pd.DataFrame(rr)

        def integ(x, tck, constant=0.0):
            x = np.atleast_1d(x)
            out = np.zeros(x.shape, dtype=x.dtype)
            for n in range(len(out)):
                out[n] = interpolate.splint(x[0], x[n], tck)
            out += constant
            return out

        # Calculate time integral of outflow rate
        from scipy import integrate, interpolate
        u = self.u
        rr['time_code'] = rr['time']
        # Time in Myr
        rr['time'] *= u.Myr
        x = rr['time']
        for c in rr.columns: # assume that all columns are outflow rate
            if c == 'time':
                continue
            # Constant factor need to be taken out later (along with bug fix below)
            #y = rr[c]*0.3125*1e6
            y = rr[c]

            xn = np.linspace(x.min(),x.max(),(len(x)-1)*10+1)
            spl = interpolate.splrep(x, y, k=1, s=0)
            yn = interpolate.splev(xn, spl, der=0)
            yint = integ(x, spl)
            #yint = integrate.cumtrapz(yn, xn, initial=0.0)
            #yint = integrate.cumtrapz(y, x, initial=0.0)
            rr[c + '_int'] = yint

        return rr

    @LoadSim.Decorators.check_pickle
    def read_outflow(self, num, prefix='outflow',
                     savdir=None, force_override=False):
        """
        Function to calculate outflow rate using surface integral
        of momentum flux at the computational boundary
        """

        ds = self.load_vtk(num)
        dd = ds.get_field(['rho','specific_scalar_CL',
                           'xH2','xHI','xHII','vx','vy','vz'])

        # Found a bug: need to run script again for all models
        # dA = self.domain['dx'][0]*au.pc**2
        # conv = (dA*au.g/au.cm**3*au.km/au.s).to('Msun yr-1').value

        dA = self.domain['dx'][0]**2*au.pc**2
        conv = (dA*au.g/au.cm**3*au.km/au.s).to('Msun Myr-1').value

        keys = ['x', 'y', 'z', 'tot', 'xcl', 'ycl', 'zcl', 'totcl']
        phases = ['HI','HII','H2']
        fac = dict(HI=1.0,HII=1.0,H2=2.0)

        Mof = dict()
        Mof['time'] = ds.domain['time']

        for ph_ in phases:
            ph = 'x' + ph_
            Mx = fac[ph_]*(dd[ph]*dd['rho']*dd['vx']).data
            My = fac[ph_]*(dd[ph]*dd['rho']*dd['vy']).data
            Mz = fac[ph_]*(dd[ph]*dd['rho']*dd['vz']).data
            Mxcl = fac[ph_]*(dd[ph]*dd['rho']*dd['vx']*dd['specific_scalar_CL']).data
            Mycl = fac[ph_]*(dd[ph]*dd['rho']*dd['vy']*dd['specific_scalar_CL']).data
            Mzcl = fac[ph_]*(dd[ph]*dd['rho']*dd['vz']*dd['specific_scalar_CL']).data
            Mof['x_'+ph_] = (Mx[:,:,-1].sum() - Mx[:,:,0].sum())*conv
            Mof['y_'+ph_] = (My[:,-1,:].sum() - My[:,0,:].sum())*conv
            Mof['z_'+ph_] = (Mz[-1,:,:].sum() - Mz[0,:,:].sum())*conv
            Mof['tot_'+ph_] = Mof['x_'+ph_] + Mof['y_'+ph_] + Mof['z_'+ph_]
            Mof['xcl_'+ph_] = (Mxcl[:,:,-1].sum() - Mxcl[:,:,0].sum())*conv
            Mof['ycl_'+ph_] = (Mycl[:,-1,:].sum() - Mycl[:,0,:].sum())*conv
            Mof['zcl_'+ph_] = (Mzcl[-1,:,:].sum() - Mzcl[0,:,:].sum())*conv
            Mof['totcl_'+ph_] = Mof['xcl_'+ph_] + Mof['ycl_'+ph_] + Mof['zcl_'+ph_]

        return Mof
