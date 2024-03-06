import numpy as np
import astropy.units as au
import astropy.constants as ac

from ..io.read_hst import read_hst
from ..decorators import check_pickle_hst

class ReadHstRadiation:

    @check_pickle_hst
    def read_hst_rad(self, savdir=None, force_override=False, suffix='rad'):

        u = self.u
        LxLy = self.domain['Lx'][0]*self.domain['Lx'][1]*u.pc**2
        hnu_LyC = (self.par['radps']['hnu_PH']*au.eV).cgs.value
        h = read_hst(self.files['hst'])
        h['tMyr'] = h['time']*u.Myr
        h['fesc_LyC'] = (h['Lpp0'] + h['Lesc0'] + h['Lxymax0'])/h['Ltot0']
        h['fesc_FUV'] = (h['Lpp1'] + h['Lesc1'] + h['Lxymax1'] + h['Leps1'] +\
                         h['Lpp2'] + h['Lesc2'] + h['Lxymax2'] + h['Leps2'])/(h['Ltot1'] + h['Ltot2'])
        # Luminosity per area [Lsun/pc^2]
        h['Sigma_LyC'] = h['Ltot0']*u.Lsun/LxLy
        h['Sigma_FUV'] = (h['Ltot1'] + h['Ltot2'])*u.Lsun/LxLy
        # Ionizing photon rate per area [#/s/kpc^2]
        h['Phi_LyC'] = h['Sigma_LyC']*(1.0*au.Lsun).cgs.value/hnu_LyC/u.kpc**2

        h = h.replace([-np.inf, np.inf], 0.0)

        return h
