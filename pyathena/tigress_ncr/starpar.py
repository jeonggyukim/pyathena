# starpar.py

import numpy as np
import pandas as pd
import xarray as xr
import astropy.units as au
import astropy.constants as ac
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import matplotlib as mpl

from ..load_sim import LoadSim
from ..util.mass_to_lum import mass_to_lum

class StarPar():

    @LoadSim.Decorators.check_pickle
    def read_starpar_all(self, nums=None, prefix='starpar_all',
                         savdir=None, force_override=False):
        """Function to read all post-processed starpar dump
        """
        if nums is None:
            nums = self.nums_starpar

        rr = dict()
        for i,num in enumerate(nums):
            # print(num, end=' ')
            r = self.read_starpar(num=num, force_override=True)
            if i == 0:
                for k in r.keys():
                    rr[k] = []

            for k in r.keys():
                try:
                    rr[k].append(r[k].value.item())
                except:
                    rr[k].append(r[k])

        rr = pd.DataFrame(rr)
        return rr

    @LoadSim.Decorators.check_pickle
    def read_starpar(self, num, savdir=None, force_override=False):
        """Function to read post-processed starpar dump
        """

        sp = self.load_starpar_vtk(num)
        u = self.u
        domain = self.domain
        par = self.par
        LxLy = domain['Lx'][0]*domain['Lx'][1]

        try:
            agemax = par['radps']['agemax_rad']
        except KeyError:
            self.logger.warning('agemax_rad was not found and set to 40 Myr.')
            agemax = 20.0

        sp['age'] *= u.Myr
        sp['mage'] *= u.Myr
        sp['mass'] *= u.Msun

        # Select non-runaway starpar particles with mass-weighted age < agemax_rad
        isrc = np.logical_and(sp['mage'] < agemax,
                              sp['mass'] != 0.0)

        if np.sum(isrc) == 0:
            return None

        # Center of mass, luminosity, standard deviation of z-position
        # Summary
        r = dict()
        r['time'] = sp.time
        r['nstars'] = sp.nstars
        r['isrc'] = isrc
        r['nsrc'] = np.sum(isrc)

        # Select only sources
        sp = sp[(sp['mage'] < agemax) & (sp['mass'] != 0.0)].copy()

        # Calculate luminosity of source particles
        LtoM = mass_to_lum(model='SB99')
        sp['Qi'] = LtoM.calc_Qi_SB99(sp['mass'], sp['mage'])
        sp['L_PE'] = LtoM.calc_LPE_SB99(sp['mass'], sp['mage'])
        sp['L_LW'] = LtoM.calc_LLW_SB99(sp['mass'], sp['mage'])
        sp['L_FUV'] = sp['L_PE'] + sp['L_LW']

        # Save source as separate DataFrame
        r['sp_src'] = sp

        r['z_max'] = np.max(sp['x3'])
        r['z_min'] = np.min(sp['x3'])
        r['z_mean_mass'] = np.average(sp['x3'], weights=sp['mass'])
        r['z_mean_Qi'] = np.average(sp['x3'], weights=sp['Qi'])
        r['z_mean_LFUV'] = np.average(sp['x3'], weights=sp['L_FUV'])
        r['stdz_mass'] = np.sqrt(np.average((sp['x3'] - r['z_mean_mass'])**2,
                                            weights=sp['mass']))
        r['stdz_Qi'] = np.sqrt(np.average((sp['x3'] - r['z_mean_Qi'])**2,
                                          weights=sp['Qi']))
        r['stdz_LFUV'] = np.sqrt(np.average((sp['x3'] - r['z_mean_LFUV'])**2,
                                            weights=sp['L_FUV']))

        r['Qi_tot'] = np.sum(sp['Qi'])
        r['L_LW_tot'] = np.sum(sp['L_LW'])
        r['L_PE_tot'] = np.sum(sp['L_PE'])
        r['L_FUV_tot'] = np.sum(sp['L_FUV'])

        # Ionizing photon per unit area
        r['Phi_i'] = r['Qi_tot']/LxLy
        # FUV luminosity per unit area
        r['Sigma_FUV'] = r['L_FUV_tot']/LxLy

        r['sp'] = sp

        return r

    @staticmethod
    def does_domain_intersect_circle(domain, x, y, r):
        # Reference
        # https://stackoverflow.com/a/4579069

        dsq = r**2 # Distance squared
        if (x < domain['le'][0]):
            dsq -= (x - domain['le'][0])**2
        elif (x > domain['re'][0]):
            dsq -= (x - domain['re'][0])**2

        if (y < domain['le'][1]):
            dsq -= (y - domain['le'][1])**2
        elif (y > domain['re'][1]):
            dsq -= (y - domain['re'][1])**2

        # For TIGRESS, no need to check conditions for the z-coord
        # since |z_src| < Lz/2 always
        # if (z < domain['le'][2]):
        #     dsq -= (z - domain['le'][2])**2
        # elif (z > domain['re'][2]):
        #     dsq -= (z - domain['re'][2])**2

        return dsq > 0


    def starpar_copy_location(self, sp, factor=1.0):
        """
        Function to calculate copied location of star particles

        Parameters
        ----------
        sp : Pandas DataFrame
            starpar data
        factor: float
            mulutiplication factor used for determining the extended domain size
            x_(max/min)_ext = x_(max/min) + factor*Lx

        Returns
        -------
        dictionary containing copied location, id of star particles
        """

        par = self.par
        domain = self.domain
        u = self.u

        deltay = self.calc_deltay(sp.time)

        # source selection criteria implemented in Athena-TIGRESS
        #sp_ = sp[(sp['mass'] > 0.0) & (sp['flag'] != -2) & \
        #         (sp['mage'] < agemax_rad)].copy(deep=True)

        #mage_Myr = sp_src['mage']*u.Myr
        #mass_Msun = sp_src['mass']*u.Msun

        xmin,ymin,zmin = domain['le']
        xmax,ymax,zmax = domain['re']
        Lx,Ly,Lz = domain['Lx']

        # Set extended domain
        if (par['domain1']['bc_ix1'] == 4) and (par['domain1']['bc_ox1'] == 4):
            xmin -= factor*Lx
            xmax += factor*Lx

        if (par['domain1']['bc_ix2'] == 4) and (par['domain1']['bc_ox2'] == 4):
            ymin -= factor*Ly
            ymax += factor*Ly

        if (par['domain1']['bc_ix3'] == 4) and (par['domain1']['bc_ox3'] == 4):
            zmin -= factor*Lz
            zmax += factor*Lz

        r = dict()
        r['x1copy'] = []
        r['x2copy'] = []
        r['x3copy'] = []
        r['idcopy'] = []
        r['ncopy'] = []

        #for x1,x2,x3,Qi,xymax in zip(sp['x1'],sp['x2'],sp['x3'],Qiall,xymaxall):
        for x1,x2,x3,spid in zip(sp['x1'],sp['x2'],sp['x3'],sp['id']):
            # the range from -1 to 1 is enough since xymaxPP does not exceed Lx, Ly
            x1copy = []
            x2copy = []
            x3copy = []
            idcopy = []
            ncopy = 0
            for k in range(-1,2):
                for j in range(-1-int(factor),2+int(factor)):
                    for i in range(-1,2):
                        # Consider all possible positions
                        newx1 = x1 + i*Lx
                        newx2 = x2 + j*Ly
                        newx3 = x3 + k*Lz
                        if par['configure']['ShearingBox'] == 'yes':
                            newx2 -= i*deltay

                        # if periodic in x and new position falls in an extended domain
                        if ((newx1 >= xmin) and (newx1 <= xmax) and \
                            (newx2 >= ymin) and (newx2 <= ymax) and \
                            (newx3 >= zmin) and (newx3 <= zmax)):
                            x1copy.append(newx1)
                            x2copy.append(newx2)
                            x3copy.append(newx3)
                            idcopy.append(spid)
                            ncopy += 1


            r['x1copy'].append(np.array(x1copy))
            r['x2copy'].append(np.array(x2copy))
            r['x3copy'].append(np.array(x3copy))
            r['idcopy'].append(idcopy)
            r['ncopy'].append(ncopy)

        return r

    def starpar_select_and_calc_rad_sources(self, sp):
        """Select radiation sources
        (s > 0 & flag != -2 & mage < agemax_rad)
        and calculate luminosity and dmax_eff
        """

        u = self.u
        par = self.par
        domain = self.domain

        agemax_rad = self.par['radps']['agemax_rad']/u.Myr
        sp_src = sp[(sp['mass'] > 0.0) & # exclude runaways
                    (sp['flag'] != -2) & # -2 for star particle created at t=0
                    (sp['mage'] < agemax_rad)].copy(deep=True)

        sp_src.time = sp.time

        mray = par['radps']['ray_number']
        eps_PP = par['radps']['eps_extinct']

        from pyathena.util.mass_to_lum import mass_to_lum
        MtoL = mass_to_lum(model='SB99')

        # Max age of source particles in code unit
        # Select source particles
        agemax_rad = par['radps']['agemax_rad']/u.Myr
        sp_src['Qi'] = MtoL.calc_Qi_SB99(sp_src['mass']*u.Msun, sp_src['mage']*u.Myr)
        sp_src['L_LyC'] = (sp_src['Qi']*par['radps']['hnu_PH']*(1.0*au.eV).cgs.value)/(ac.L_sun.cgs.value)
        sp_src['L_PE'] = MtoL.calc_LPE_SB99(sp_src['mass']*u.Msun, sp_src['mage']*u.Myr)
        sp_src['L_LW'] = MtoL.calc_LLW_SB99(sp_src['mass']*u.Msun, sp_src['mage']*u.Myr)
        sp_src['L_FUV'] = sp_src['L_PE'] + sp_src['L_LW']
        sp_src['L_tot'] = sp_src['L_FUV'] + sp_src['L_LyC']

        # Fraction of total Qi
        sp_src['eps_src_LyC'] = sp_src['Qi']/sp_src['Qi'].sum()
        sp_src['eps_src_FUV'] = sp_src['L_FUV']/sp_src['L_FUV'].sum()

        # Maximum distance that PPs can travel from individual sources
        # in the optically-thin limit
        tau = 0.0 # optical depth from the source
        try:
            sp_src['dmax_eps'] = 1/(4.0*np.pi*mray)**0.5*sp_src['eps_src_LyC']**0.5*\
                np.exp(-0.5*tau)*eps_PP**-0.5*domain['dx'][0]
        except ZeroDivisionError:
            sp_src['dmax_eps'] = par['radps']['xymaxPP']

        sp_src['dmax_eff'] = np.minimum(par['radps']['xymaxPP'], sp_src['dmax_eps'])
        self.sp_src = sp_src

        return sp_src

    def starpar_mark_image_sources(self, sp_copy, sp_src,
                                   flatten=True, kind='rad'):

        domain = self.domain
        par = self.par

        sp_copy['intersect'] = []
        if kind == 'rad':
            for x1copy,x2copy,radius in zip(sp_copy['x1copy'],sp_copy['x2copy'],sp_src['dmax_eff']):
                intersect = []
                for x1,x2 in zip(x1copy,x2copy):
                    # True if photon packets launched from this image source can reach the
                    # computational domain (in the optically thin limit)
                    intersect.append(self.does_domain_intersect_circle(domain, x1, x2, radius))

                sp_copy['intersect'].append(np.array(intersect))

        elif kind == 'sn':
            # Maximum feedback radius
            radius = min(domain['Lx'].min()*0.5, par['feedback']['rfb_sn_max'])
            for x1copy,x2copy in zip(sp_copy['x1copy'],sp_copy['x2copy']):
                intersect = []
                for x1,x2 in zip(x1copy,x2copy):
                    intersect.append(self.does_domain_intersect_circle(domain, \
                                x1, x2, radius))

                sp_copy['intersect'].append(np.array(intersect))

        if flatten:
            for k in sp_copy.keys():
                try:
                    sp_copy[k] = np.concatenate(sp_copy[k]).ravel()
                except ValueError:
                    pass


        return sp_copy

    def plt_starpar_copy(self, sp_src, sp_copy, kind='rad'):
        """
        Plot radiation sources (real and image sources)
        """

        par = self.par
        u = self.u
        domain = self.domain

        r = sp_copy

        cmap = plt.cm.cool_r
        norm = Normalize(vmin=0., vmax=par['radps']['agemax_rad'])
        idx = r['intersect']
        fig, ax = plt.subplots(1,1,figsize=(15,15))

        plt.scatter(r['x1copy'], r['x2copy'], s=5, fc='none', ec='grey', lw=1)

        plt.scatter(r['x1copy'][idx], r['x2copy'][idx], s=10, fc='none', ec='b', lw=1)
        plt.scatter(sp_src['x1'], sp_src['x2'], s=np.sqrt(sp_src['mass']*u.Msun),
                    fc=cmap(norm(sp_src['mage']*u.Myr)), ec='k', lw=1)

        deltay = self.calc_deltay(sp_src.time)

        ii = int(par['radps']['xymaxPP']/domain['Lx'][0] - 1)

        # Draw image domains
        for i in range(-1-ii,2+ii):
            rect  = [mpl.patches.Rectangle((domain['le'][0] + i*domain['Lx'][0],
                                            domain['le'][1] - i*deltay + j*domain['Lx'][1]),
                                            domain['Lx'][0], domain['Lx'][1]) for j in range(-2-ii,3+ii)]
            pc  = mpl.collections.PatchCollection(rect, alpha=1, facecolor='none', edgecolor='k')
            ax.add_collection(pc)

        # Draw dmax_eff as circles
        if kind == 'rad':
            circ = [mpl.patches.Circle((x1,x2), r) for x1,x2,r in \
                    zip(sp_src['x1'],sp_src['x2'], sp_src['dmax_eff'])]
        elif kind == 'sn':
            radius = min(domain['Lx'].min()*0.5, par['feedback']['rfb_sn_max'])
            circ = [mpl.patches.Circle((x1,x2), radius) for x1,x2 \
                    in zip(sp_src['x1'],sp_src['x2'])]

        pc2  = mpl.collections.PatchCollection(circ, alpha=1, facecolor='none',
                                               edgecolor='grey', linewidth=0.5, linestyle='-')
        ax.add_collection(pc2)

        # re-draw the real domain with a thick line
        rect  = mpl.patches.Rectangle((domain['le'][0],domain['le'][1]),
                                      domain['Lx'][0], domain['Lx'][1])
        pc  = mpl.collections.PatchCollection([rect], alpha=1,
                                              facecolor='none', edgecolor='k', linewidth=2)
        plt.gca().add_collection(pc)

        Lx,Ly = domain['Lx'][0], domain['Lx'][0]
        plt.xlim(-1.5*Lx,1.5*Lx)
        plt.ylim(-1.5*Ly,1.5*Ly)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.suptitle(r'$t=$' + r'{0:0.1f}'.format(sp_src.time) +
                     r', $\Delta y=$' + r'{0:0.1f}'.format(deltay), y=0.92)

        return fig

    @staticmethod
    def calc_Erad_in_uniform_medium(dd, sp_src, r, k_atten=0.005, bands=['LyC','PE']):
        """
        Function to compute Erad in a uniform medium
        given radiation sources that allow for shear-periodic BCs

        Parameters
        ----------
        dd : xarray Dataset
            Athena vtk dump
        bands : list of strings
            'LyC', 'LW', 'PE', 'FUV'
        k_atten : float
            attenuation factor : exp(-k_atten*dist_pc)
            1/k_atten is the mean free path in pc
            Use k_atten=0 for optically thin limit
        """

        pc_cgs = (1.0*au.pc).cgs.value
        c_cgs = ac.c.cgs.value
        Lsun_cgs = ac.L_sun.cgs.value

        for b in bands:
            name = 'Erad_{0:s}_uni'.format(b)
            dd[name] = xr.zeros_like(dd['nH'])
            var = 'Erad_{0:s}_uni'.format(b)

        idx = r['intersect']
        for i,(x1,x2,x3,spid) in enumerate(zip(r['x1copy'][idx], r['x2copy'][idx], r['x3copy'][idx],
                                               r['idcopy'][idx])):
            print(i,end=' ')
            rsq = (dd.x - x1)**2+(dd.y - x2)**2+(dd.z - x3)**2

            for b in bands:
                name = 'Erad_{0:s}_uni'.format(b)
                lum = (sp_src.loc[sp_src['id'] == spid, 'L_{0:s}'.format(b)]).values
                dd[name] += lum*Lsun_cgs*np.exp(-k_atten*np.sqrt(rsq))/\
                    (4.0*np.pi*rsq*pc_cgs**2)/c_cgs

        return dd
