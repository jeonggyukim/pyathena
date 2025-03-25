import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as au
import astropy.constants as ac
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid, cumulative_trapezoid
from string import ascii_letters, digits
from dataclasses import dataclass, field

from .photx import PhotX
from .rec_rate import RecRate
from .ct_rate import ChargeTransferRate
from .ci_rate import CollIonRate
try:
    from .hii_wind import HIIWind
except ImportError:
    # Silent failure
    pass

@dataclass(frozen=True)
class Ion:
    """Strictly speaking, ions refer to charged particles. However, neutral particles
    (atoms, e.g., "H0", "He0") are also treated on the same footing here.

    """

    # Ion name
    name: str
    # True if the ion can be ionized.
    ionize: bool
    # True if the ion can recombine.
    recomb: bool
    # Element name
    element: str = field(init=False)
    # Atomic number
    Z: int = field(init=False)
    # Electron number
    N: int = field(init=False)
    # Charge number
    q: int = field(init=False)
    # Set in post_init
    wav_thres: float = field(init=False)

    def __post_init__(self):
        atomic_number = dict(H=1, He=2, Li=3, Be=4, B=5, C=6, N=7, O=8, F=9,
                             Ne=10, Na=11, Mg=12, Al=13, Si=14, P=15, S=16,
                             Cl=17, Ar=18, K=19, Ca=20, Sc=21, Ti=22, V=23,
                             Cr=24, Mn=25, Fe=26, Co=27, Ni=28, Cu=29, Zn=30)

        # Set element, Z, N, and q
        object.__setattr__(self, 'element', self.name.rstrip(digits))
        object.__setattr__(self, 'Z', atomic_number[self.element])
        object.__setattr__(self, 'N',
                           self.Z - int(self.name.lstrip(ascii_letters)))
        object.__setattr__(self, 'q', self.Z - self.N)

        # Sanity check for ionize/recombine flags
        if self.Z == self.N and self.recomb:
            raise ValueError(
                'Check recomb flag! name={0:s} Z={1:d} N={2:d}, recomb={3}'.\
                            format(self.name, self.Z, self.N, self.recomb))
        if self.N == 0 and self.ionize:
            raise ValueError(
                'Check ionize flag! name={0:s} Z={1:d} N={2:d}, ionize={3}'.\
                            format(self.name, self.Z, self.N, self.ionize))

        # Set threshold wavelength
        if self.ionize:
            px = PhotX()
            object.__setattr__(self, 'wav_thres',
                               px.get_Eth(self.Z, self.N, unit='Angstrom'))
        else:
            object.__setattr__(self, 'wav_thres', None)

@dataclass(frozen=True)
class SpeciesSet:
    # List of dataclass Ion
    ions: list[Ion]
    # Number of ions for each element
    # (Mutable, although immutable dict can be implemented..)
    num_ions: dict = field(init=False)
    num_ions_tot: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'names',
                           [ion.name for ion in self.ions])
        object.__setattr__(self, 'elements',
                           list(dict.fromkeys([ion.element for ion in self.ions])))

        num_ions = dict.fromkeys(self.elements)
        for e in self.elements:
            n = 0
            for ion in self.ions:
                if e == ion.element:
                    n += 1
            num_ions[e] = n

        object.__setattr__(self, 'num_ions', num_ions)
        object.__setattr__(self, 'num_ions_tot', sum(num_ions.values()))

def get_species_set(num):
    # Set ion name, ionize flag, recomb flag
    ss = np.empty(4, dtype=object)

    # Set ion name, ionize/recomb flags
    ss[0] = SpeciesSet([Ion('H0', True, False),
                        Ion('H1', False, True)])

    # H, He, O, S
    ss[1] = SpeciesSet([Ion('H0', True, False),
                        Ion('H1', False, True),
                        Ion('He0', True, False),
                        Ion('He1', True, True),
                        Ion('He2', False, True),
                        Ion('O0', True, False),
                        Ion('O1', True, True),
                        Ion('O2', False, True),
                        Ion('S0', True, False),
                        Ion('S1', True, True),
                        Ion('S2', True, True),
                        Ion('S3', True, True),
                        Ion('S4', False, True)])

    ss[2] = SpeciesSet([Ion('H0', True, False),
                        Ion('H1', False, True),
                        Ion('He0', True, False),
                        Ion('He1', True, True),
                        Ion('He2', False, True),
                        Ion('C0', True, False),
                        Ion('C1', True, True),
                        Ion('C2', False, True),
                        Ion('O0', True, False),
                        Ion('O1', True, True),
                        Ion('O2', False, True)])

    ss[3] = SpeciesSet([Ion('H0', True, False),
                        Ion('H1', False, True),
                        Ion('He0', True, False),
                        Ion('He1', True, True),
                        Ion('He2', False, True),
                        Ion('N0', True, False),
                        Ion('N1', True, True),
                        Ion('N2', False, True),
                        Ion('O0', True, False),
                        Ion('O1', True, True),
                        Ion('O2', False, True),
                        Ion('S0', True, False),
                        Ion('S1', True, True),
                        Ion('S2', True, True),
                        Ion('S3', True, True),
                        Ion('S4', False, True)])

    return ss[num]

class PhotChem(object):

    def __init__(self, fname_sb99_sed, species_set_id=1, age_Myr=0.1, Z_g=1.0):

        self._Z_g = Z_g

        self.clear_sigma_pi_above_thres = True
        self.species_set_id = species_set_id
        self.species_set = get_species_set(species_set_id)
        self.ions = pd.DataFrame(self.species_set.ions)
        self.ions.set_index('name', inplace=True)

        # Read SED
        # TODO: useful to implement SED as a separate class
        self.dfa_sed = self.read_sb99_sed_all(fname_sb99_sed)
        self.df_sed = self.get_sb99_sed(self.dfa_sed, age_Myr)

        # Modules for rate coefficients
        self.px = PhotX()
        self.rc = RecRate()
        self.ci = CollIonRate()
        self.ct = ChargeTransferRate()

        self.set_wavelength_bins()
        self.calc_mean_over_sed()

        # Set index
        self.ions['idx'] = range(self.ions.shape[0])
        idx = self.ions.pop('idx')
        self.ions.insert(0, 'idx', idx)

        self.set_abd()
        self.set_density()

        self._set_colors()

    @property
    def Z_g(self):
        return self._Z_g

    def set_abd(self):
        # Default elemental abundances (C and O consistent with Gong et al. chemistry)
        # Other heavy elements from Asplund
        # TODO: Consider depletion (e.g, Jenkins 2009)
        abd_def = dict(He=0.1, C=1.6e-4, N=7.4e-5, O=3.2e-4, S=1.45e-5)
        self.abd = dict()
        for e in set(self.species_set.elements) - {'H','He'}:
            self.abd[e] = abd_def[e]*self.Z_g

        self.abd['H'] = 1.0
        self.abd['He'] = 0.1

    def set_density(self, nH=1.0, xHI=None, Nr=2):
        # Add one more species to store free electrons
        if np.isscalar(nH):
            self.nH = np.repeat(nH, Nr)
        else:
            # Ignore Nr parameter
            self.nH = nH
            Nr = len(self.nH)

        self.den = np.zeros((self.species_set.num_ions_tot + 1, Nr))
        for name in self.ions.index:
            idx, e, q = self.ions.loc[name, ['idx','element','q']]
            # if e == 'H' and xHI is not None:
            if xHI is not None:
                if q == 0:
                    self.den[idx,:] = self.nH*self.abd[e]*xHI
                elif q == 1:
                    self.den[idx,:] = self.nH*self.abd[e]*(1.0 - xHI)
                else:
                    self.den[idx,:] = 0.0
            else:
                if q == 0:
                    self.den[idx,:] = self.nH*self.abd[e]
                else:
                    self.den[idx,:] = 0.0

            # Add to the free electron density
            self.den[-1,:] += q*self.den[idx,:]

    def _set_colors(self):
        """Assign color for each ion for visualization
        """
        # Colormap for each element
        self.cmap = dict()
        self.cmap['H'] = mpl.cm.Greys
        self.cmap['He'] = mpl.cm.Purples
        self.cmap['C'] = mpl.cm.Blues
        self.cmap['N'] = mpl.cm.Oranges
        self.cmap['O'] = mpl.cm.Greens
        self.cmap['S'] = mpl.cm.Reds

        # Normalization for each element (vmin=0, vmax=num_ions)
        self.norm = dict()
        for e in self.species_set.elements:
            self.norm[e] = mpl.colors.Normalize(0, self.species_set.num_ions[e])

        # Assign color to each ion
        c = dict()
        label = dict()
        for name in self.ions.index:
            e, q = self.ions.loc[name, ['element', 'q']]
            num_ions = self.species_set.num_ions[e]
            c[name] = mpl.colors.rgb2hex(self.cmap[e](self.norm[e](num_ions - q)))
            # Python one-liner to print Roman numerals
            # Ref: https://www.johndcook.com/blog/2020/10/07/roman-numerals
            label[name] = r'${\rm ' + f'{e}' + r'} $' + chr(0x215F + q + 1)

        # Add a new column
        self.ions['color'] = pd.Series(c)
        self.ions['label'] = pd.Series(label)

    def calc_mean_over_sed(self):
        # Photoionized ions
        ions_phot = self.ions.loc[self.ions['ionize']]
        df = self.df_sed
        # \int Xi_lambda dlambda
        Xi_cumul = cumulative_trapezoid(df['Xi'], df['wav'], initial=0.0)
        # \int hnu*Xi_lambda dlambda (= \int hnu*Xi_nu dnu)
        hnu_Xi_cumul = cumulative_trapezoid(df['hnu']*df['Xi'], df['wav'], initial=0.0)
        # \int sigma_pi*Xi_lambda dlambda
        sigma_pi_Xi_cumul = dict()
        for name, ion in ions_phot.iterrows():
            df[f'sigma_pi_{name}'] = self.px.get_sigma(ion['Z'], ion['N'],
                                                       df['hnu'].values)
            sigma_pi_Xi_cumul[name] = cumulative_trapezoid(
                df[f'sigma_pi_{name}']*df['Xi'], df['wav'], initial=0.0)

        idx = []
        Xi_tmp = []
        hnu_Xi_tmp = []
        sigma_pi_Xi_tmp = dict()
        for name in ions_phot.index:
            sigma_pi_Xi_tmp[name] = []

        for i, wav_bdry_ in enumerate(self.wav_bdry):
            idx_ = np.where(df['wav'] <= wav_bdry_)[0][-1]
            idx.append(idx_)
            Xi_tmp.append(Xi_cumul[idx_])
            hnu_Xi_tmp.append(hnu_Xi_cumul[idx_])
            for name, ion in self.ions.query('ionize == True').iterrows():
                sigma_pi_Xi_tmp[name].append(sigma_pi_Xi_cumul[name][idx_])

        # Number of photons per solar mass per second
        self.Xi = np.diff(np.array(Xi_tmp))
        hnu_Xi = np.diff(np.array(hnu_Xi_tmp))
        sigma_pi_Xi = dict()
        for name in ions_phot.index:
            sigma_pi_Xi[name] = np.diff(np.array(sigma_pi_Xi_tmp[name]))

        # Mean photon energy [eV] and the corresponding wavelenth [angstrom]
        self.hnu_mean = hnu_Xi/self.Xi
        self.wav_mean = (ac.h*ac.c/(self.hnu_mean*au.eV)).to('angstrom').value

        # Mean photoionization cross section
        self.sigma_pi_mean = dict()
        for name in ions_phot.index:
            self.sigma_pi_mean[name] = sigma_pi_Xi[name]/self.Xi

        if self.clear_sigma_pi_above_thres:
            for name in ions_phot.index:
                self.sigma_pi_mean[name][~(self.wav_thres[name] > self.wav_mean)] = 0.0

        # Save to DataFrame
        self.ions['sigma_pi'] = pd.Series(self.sigma_pi_mean)

    def set_wavelength_bins(self):
        # Set boundaries based on photoionization threshold
        # TODO: implement better version
        if self.species_set_id == 0:
            self.ions_wav_bdry = ['H0']
        elif self.species_set_id == 1:
            #self.ions_wav_bdry = ['He1','O1','He0','H0']
            self.ions_wav_bdry = ['He1','S3','O1','He0','S1','H0']
        elif self.species_set_id == 2:
            self.ions_wav_bdry = ['He1','O1','He0','H0','C0']
        elif self.species_set_id == 3:
            self.ions_wav_bdry = ['S3','O1','He0','H0','S0']

        self.wav_thres = dict()
        for name, ion in self.ions.iterrows():
            if ion['ionize']:
                self.wav_thres[name] = ion['wav_thres']

        self.wav_bdry = [self.df_sed['wav'].min()]
        for ion in self.ions_wav_bdry:
            self.wav_bdry.append(self.wav_thres[ion])

        self.wav_bdry = np.array(self.wav_bdry)
        self.nbin = len(self.wav_bdry) - 1


    def plt_rate_coeffs(self):

        fig, axes = plt.subplots(3, 1, figsize=(8, 14), constrained_layout=True)
        T = np.logspace(2, 7, 1000)
        for name, ion in self.ions[self.ions['recomb']].iterrows():
            # Radiative/dielectronic recombination
            l, = axes[0].loglog(T, self.rc.get_rec_rate(ion['Z'], ion['N'], T),
                                label=ion['label'], c=ion['color'])
            try:
                l, = axes[0].loglog(T, self.rc.get_dr_rate(ion['Z'], ion['N'], T),
                                    label='_no_legend_', c=ion['color'], ls=':')
            except:
                pass
            # CT recombination
            try:
                axes[2].loglog(T, self.ct.get_ct_rec_rate(ion['Z'], ion['N'], T),
                               label=ion['label'], c=l.get_color())
            except IndexError:
                # print('Ion {0:s}: ct_rec does not exist!'.format(name))
                pass

        for name,ion in self.ions[self.ions['ionize']].iterrows():
            # Collisional ionization
            l, = axes[1].loglog(T, self.ci.get_ci_rate(ion['Z'], ion['N'], T),
                                label=ion['label'], c=ion['color'])
            # CT ionization
            try:
                axes[2].loglog(T, self.ct.get_ct_rec_rate(ion['Z'], ion['N'], T),
                               label=ion['label'], c=l.get_color(),ls='--')
            except IndexError:
                # print('Ion {0:s}: ct_ion does not exist!'.format(name))
                pass

        # Draine11 rates
        #axes[0].loglog(T, self.rc.get_rec_rate_H_caseA_Dr11(T), c='red',ls='--')
        #axes[0].loglog(T, self.rc.get_rec_rate_H_caseB_Dr11(T), c='red',ls=':')

        axes[0].legend(loc='upper left', fontsize='small')
        axes[1].legend(loc='upper left', fontsize='small')
        axes[2].legend(loc='upper left', fontsize='small', ncols=2)

        #     def alphaB(T):
        #         l = 315614.0 / T
        #         return 2.753e-14*l**1.5/(1. + (l/2.74)**0.407)**2.242
        # axes[0].loglog(T, alphaB(T), c='r', ls='--')

        plt.setp(axes, xlabel=r'$T\;[{\rm K}]$')
        plt.setp(axes[0], ylabel='recomb rates', ylim=(1e-14,1e-9))
        plt.setp(axes[1], ylabel='coll ion rates', ylim=(1e-14,1e-7))
        plt.setp(axes[2], ylabel='CT rates', ylim=(1e-16,1e-7))

    def plt_sed_sigma_pi(self):
        xlim = (80, 2.0e3)
        ylim = (1e38, 1e48)
        fig, axes = plt.subplots(2,1,figsize=(8,8),constrained_layout=True)

        df = self.df_sed
        plt.sca(axes[0])
        plt.loglog(df['wav'], df['wav']*df['Xi'], c='k')
        plt.scatter(self.wav_mean, self.Xi, color='r')
        for i in range(self.nbin):
            ion = self.ions_wav_bdry[i]
            plt.fill_betweenx(ylim, self.wav_bdry[i], self.wav_bdry[i+1],
                              alpha=0.1, color=self.ions['color'][ion])

        plt.ylim(ylim)
        for k, v in self.wav_thres.items():
            vline_color = self.ions['color'][k]
            label = self.ions['label'][k]
            plt.axvline(v, label=label, color=vline_color)

        plt.legend(loc='upper left', fontsize='small')
        plt.sca(axes[1])
        for name, ion in self.ions.query('ionize == True').iterrows():
            plt.loglog(df['wav'], df[f'sigma_pi_{name}'],
                       #label=name, c=ion['color'])
                       label=ion['label'], c=ion['color'])
            plt.scatter(self.wav_mean, self.sigma_pi_mean[name], color=ion['color'])

        plt.legend(loc='upper left', fontsize='small')
        plt.setp(axes, xlim=xlim, xlabel=r'$\lambda [\AA]$')
        plt.setp(axes[0], xlim=xlim, ylim=ylim,
                 ylabel=r'$\lambda Q_{\lambda}/M_*\;[{\rm photons}\;' +\
                 r'{\rm s}^{-1}\;M_\odot^{-1}]$')
        plt.setp(axes[1], ylabel=r'$\sigma_{\rm pi}\;[{\rm cm^{2}}]$')

    @staticmethod
    def read_sb99_sed_all(fname='/Users/jgkim/Z1_M1E6.spectrum1', logMstar=6.0):
        dfa = pd.read_csv(fname, skiprows=6, sep=r'\s+',
                          names=['age', 'wav', 'logf', 'logfstar', 'logfneb'])
        dfa = dfa.rename(columns={'age': 'age_yr'})
        dfa['age_Myr'] = dfa['age_yr']*1e-6
        # Shift column
        cols = dfa.columns.tolist()
        dfa = dfa[cols[-1:] + cols[:-1]]
        dfa.attrs = dict(logMstar=6.0)
        return dfa

    @staticmethod
    def get_sb99_sed(dfa, age_Myr, wav_max=2000.0):
        idx_age = (dfa['age_Myr'] - age_Myr).abs() == \
            (dfa['age_Myr'] - age_Myr).abs().min()
        idx_wav = dfa['wav'] <= wav_max
        df = (dfa.loc[idx_wav & idx_age]).\
            copy(deep=True).reset_index(drop=True)
        
        # idx = ((dfa['time_Myr'] - time).abs() ==\
        #     (dfa['time_Myr'] - time).abs().min()) &&\
        #     (dfa['wav'] < wav_max)
                                                              
        # df = (dfa.loc[(dfa['time_Myr'] >= time) &
        #               (dfa['time_Myr'] < time + dt) &
        #               (dfa['wav'] < wav_max)]).\
        #         copy(deep=True).reset_index(drop=True)
        eV_cgs = (1.0*au.eV).cgs.value
        # Luminosity per stellar mass [erg/s/Angstrom/Msun]
        df['Psi'] = 10.0**(df.logf - df.attrs['logMstar'])
        df['hnu'] = (((ac.c*ac.h)/(df['wav'].values*au.angstrom)).to('eV')).value
        # Xi = Psi/hnu
        df['Xi'] = df['Psi']/(df['hnu']*eV_cgs)

        return df

    def init_with_HIIWind(self,
                          Mstar=1e3,
                          R0=10*au.pc,
                          par=dict(a=1.0, alpha_rocket=0.5, vesc0=0.0,
                                   r_star=0.2, f_star=0.1, k_rho=1.0,
                                   taud_0=0.0, f_esci=0.9, Tion=1e4),
                          nH_const=False):

        self.pc_cgs = (1.0*au.pc).cgs.value
        self.Qi = Mstar*self.Xi/au.s
        self.R0 = R0
        self.Tion = par['Tion']
        self.w = HIIWind.from_dict(copy.deepcopy(par))
        self.df_w = HIIWind.calc_all(self.w)
        #print(self.df_w.f_esci)
        #print(par)
        #print('f_esci, f_ion:', self.w.f_esci, self.w.f_ion)
        self.nrms = self.w.calc_nrms(self.Qi.sum()*self.w.f_ion,
                                     self.R0, self.w.Tion)*1.0

        # nH in cm^-3
        if not nH_const:
            nH_ = self.nrms*self.w.rho
        else:
            nH_ = np.repeat(self.nrms, len(self.w.rho))

        # Radius in pc
        r_ = self.R0*self.w.r
        # Approximate HI fraction
        phi_ = self.w.phi
        Qi_sigma_pi_H0_ = np.sum(self.Qi*self.sigma_pi_mean['H0']*au.cm**2)
        xHI_ = (self.w.alphaB*nH_/(Qi_sigma_pi_H0_*phi_/(4.0*np.pi*r_**2))).cgs.value

        Nr = 1000
        # Volume centered position
        self.r = np.logspace(np.log10(r_.min().value),
                             np.log10(r_.max().value), Nr)
        # Face centered position
        dlogr = np.diff(np.log(self.r))[0]
        self.rf = self.r*np.exp(-0.5*dlogr)
        self.rf = np.append(self.rf, self.r[-1]*np.exp(0.5*dlogr))
        self.dVol_inv_cgs = 1/(4.0*np.pi/3.0*np.diff(self.rf**3)*self.pc_cgs**3)

        # Interpolated density
        self.nH_interp = interp1d(r_, nH_, fill_value='extrapolate')
        self.xHI_interp = interp1d(r_, xHI_, fill_value='extrapolate')
        # phi_interp = interp1d(r_, phi_, fill_value='extrapolate')
        self.set_density(self.nH_interp(self.r), self.xHI_interp(self.r))
        # self.set_density(self.nH_interp(self.r))
        # Set opacity array [1/cm]
        self.chi = np.zeros((self.den.shape[0]-1, self.den.shape[1], self.nbin))

    def calc_radiation_field(self):
        for i, name in enumerate(self.ions.index):
            if self.ions.loc[name, 'ionize']:
                self.chi[i,:,:] = np.outer(self.den[i,:], self.sigma_pi_mean[name])
            else:
                self.chi[i,:,:] = 0.0

        # Calculate dtau and tau at cell faces
        self.dtau = (self.chi.sum(axis=0).T*np.diff(self.rf*self.pc_cgs)).T
        self.tauf = np.zeros((self.dtau.shape[0]+1, self.dtau.shape[1]))
        self.tauf[1:,:] = self.dtau.cumsum(axis=0)
        self.Fphot = np.einsum('ij,i->ij', self.Qi.value*np.exp(-self.tauf[:-1,:])*\
                               (1 - np.exp(-self.dtau)) / self.chi.sum(axis=0),
                               self.dVol_inv_cgs)

    def evolve_one_species(self, idx, T, dt, axes=None):
        den = self.den.view()
        nH = self.nH.view()
        n = den[idx]
        ne = den[-1]
        T = self.w.Tion
        Fphot = self.Fphot.view()
        name = self.ions.index[idx]
        ionize, recomb, Z, N, q, element = \
            self.ions.iloc[idx][['ionize','recomb','Z','N','q','element']]
        sigma_pi = self.ions.iloc[idx]['sigma_pi']

        if recomb:
            # Compute pi and ci rates of the ion with one less charge
            n_m1 = den[idx-1]
            sigma_pi_m1 = self.ions.iloc[idx-1]['sigma_pi']
            pi_rate_m1 = n_m1*(sigma_pi_m1*Fphot).sum(axis=1)
            ci_rate_m1 = n_m1*ne*self.ci.get_ci_rate(Z, N+1, T)
            rc_rate = ne*self.rc.get_rec_rate(Z, N, T)
        else:
            pi_rate_m1 = 0.0
            ci_rate_m1 = 0.0
            rc_rate = 0.0

        if ionize:
            # Compute recombination rate of the ion with one more charge
            n_p1 = den[idx+1]
            rc_rate_p1 = n_p1*ne*self.rc.get_rec_rate(Z, N-1, T)
            ci_rate = ne*self.ci.get_ci_rate(Z, N, T)
            pi_rate = (sigma_pi*Fphot).sum(axis=1)
        else:
            rc_rate_p1 = 0.0
            ci_rate = 0.0
            pi_rate = 0.0

        # Note that all destruction rates divided by n (hence in units of [1/time])
        # while creation rates have [1/time/volume]
        drate = rc_rate + ci_rate + pi_rate
        crate = rc_rate_p1 + ci_rate_m1 + pi_rate_m1
        n_after = (n + crate*dt)/(1 + drate*dt)

        self.den[idx,:] = n_after
        # Update electron density every time?
        # self.den[-1,:] += q*(n_after - n)

        if not ionize:
            # Update free electron number density
            # Renormalization assuming that the element update is finished
            # CAUTION: the condition (not ionized) may not work always
            num_ion = self.species_set.num_ions[element]
            n_elem = nH*self.abd[element]
            corr = n_elem/den[idx - num_ion+1:idx+1].sum(axis=0)
            den[idx - num_ion+1:idx+1] *= corr

        if axes is not None:
            axes[0].semilogy(self.r, den[0,:]/nH, c='k')
            axes[0].semilogy(self.r, den[1,:]/nH, c='b')
            axes[1].set_title('He')

            axes[1].plot(self.r, den[2,:]/(nH*self.abd['He']), c='k')
            axes[1].plot(self.r, den[3,:]/(nH*self.abd['He']), c='b')
            axes[1].plot(self.r, den[4,:]/(nH*self.abd['He']), c='r')
            axes[1].set_title('He')

            axes[2].plot(self.r, den[5,:]/(nH*self.abd['N']), c='k')
            axes[2].plot(self.r, den[6,:]/(nH*self.abd['N']), c='b')
            axes[2].plot(self.r, den[7,:]/(nH*self.abd['N']), c='r')

            axes[3].plot(self.r, den[8,:]/(nH*self.abd['O']), c='k')
            axes[3].plot(self.r, den[9,:]/(nH*self.abd['O']), c='b')
            axes[3].plot(self.r, den[10,:]/(nH*self.abd['O']), c='r')

            axes[4].plot(self.r, den[11,:]/(nH*self.abd['S']), c='k')
            axes[4].plot(self.r, den[12,:]/(nH*self.abd['S']), c='b')
            axes[4].plot(self.r, den[13,:]/(nH*self.abd['S']), c='r')
            axes[4].plot(self.r, den[14,:]/(nH*self.abd['S']), c='g')
            axes[4].plot(self.r, den[15,:]/(nH*self.abd['S']), c='y')

            axes[-1].plot(self.r, Fphot[:,0], c='g')
            axes[-1].plot(self.r, Fphot[:,1], c='r')
            axes[-1].plot(self.r, Fphot[:,2], c='b')
            axes[-1].plot(self.r, Fphot[:,3], c='k')


    def evolve_all(self, dt = 1e-1*au.Myr, tlim=10.0*au.Myr, axes=None):
        for i in range(int(tlim/dt)):
            # print(i, end=' ')
            self.calc_radiation_field()
            for idx in range(self.species_set.num_ions_tot):
                self.evolve_one_species(idx, self.Tion, dt.cgs.value, axes=None)
                self.den[-1,:] = self.den[1,:]

            # Set free electron density
            self.den[-1,:] = 0.0
            for name in self.ions.index:
                idx, q = self.ions.loc[name, ['idx','q']]
                self.den[-1,:] += q*self.den[idx,:]


# if __name__ == '__main__':
#     # Example usage
#     pc = PhotChem('/Users/jgkim/Dropbox/Projects/Starburst99/galaxy/output/standard.spectrum1',
#                   species_set_id=1, age_Myr=3.0, Z_g=1.0)
#     pc.plt_sed_sigma_pi()
#     pc.plt_rate_coeffs()
