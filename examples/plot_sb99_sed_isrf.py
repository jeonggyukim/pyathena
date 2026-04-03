"""
Three-panel figure: dust opacity / SED evolution / ISRF.

Reproduces the figure from Leitherer et al. (1999) / Kim et al. analysis,
using the pyathena.util.sb99 API.

Usage
-----
    conda run -n pyathena python examples/plot_sb99_sed_isrf.py
"""
import pathlib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as au
import astropy.constants as ac
from matplotlib.colors import Normalize
from scipy import integrate
from scipy.special import expn
from scipy.interpolate import interp1d

import pyathena as pa
from pyathena.util import sb99
from pyathena.util.rad_isrf import nuJnu_Dr78, Jlambda_MMP83, Jnu_vD82
from pyathena.util.rad_uvb import read_FG20
from pyathena.microphysics.dust_draine import DustDraine
from pyathena.plt_tools.set_plt import set_plt_fancy

set_plt_fancy()

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO   = pathlib.Path(pa.__file__).parent.parent
DATA   = REPO / 'data' / 'sb99'
SHORT  = DATA / 'Z014_M1E6_GenevaV00_dt02'
LONG   = DATA / 'Z014_M1E6_GenevaV00_logdt_10Gyr'

# ── Load datasets ──────────────────────────────────────────────────────────────
sb_short = sb99.SB99(str(SHORT), verbose=True)
rr_short = sb_short.read_rad()

sb_long = sb99.SB99(str(LONG), verbose=True)
rr_long = sb_long.read_rad()

# ── ISRF via updated API ───────────────────────────────────────────────────────
Sigma_gas = 10.0 * au.M_sun / au.pc**2
Sigma_SFR = 2.5e-3 * au.M_sun / au.kpc**2 / au.yr

r_isrf = sb99.get_ISRF_SB99_plane_parallel(
    rr=rr_short, rr_long=rr_long,
    Sigma_gas=Sigma_gas,
    Sigma_SFR=Sigma_SFR,
    verbose=True)

w_ang   = r_isrf['w_angstrom']
Jlam    = r_isrf['Jlambda']           # attenuated
Jlam_u  = r_isrf['Jlambda_unatt']     # unattenuated

# ── Zari et al. (2022) SFH cases (Appendix C) ─────────────────────────────────
# History 1: Sigma_SFR,-3 = 4 for t < 10 Myr, 2 otherwise  (blue thick)
# History 2: Sigma_SFR,-3 = 1 for t < 5 Myr, 4 for 5-10 Myr, 2 otherwise (red thick)
# Sigma_SFR,-3 = Sigma_SFR / (1e-3 Msun yr-1 kpc-2)
_u = au.M_sun / au.kpc**2 / au.yr

def sfh_zari1(t_Myr):
    return np.where(t_Myr < 10.0, 4e-3, 2e-3) * _u

def sfh_zari2(t_Myr):
    return np.where(t_Myr < 5.0, 1e-3,
           np.where(t_Myr < 10.0, 4e-3, 2e-3)) * _u

r_z1 = sb99.get_ISRF_SB99_plane_parallel(
    rr=rr_short, rr_long=rr_long,
    Sigma_gas=Sigma_gas, sfh=sfh_zari1, verbose=True)

r_z2 = sb99.get_ISRF_SB99_plane_parallel(
    rr=rr_short, rr_long=rr_long,
    Sigma_gas=Sigma_gas, sfh=sfh_zari2, verbose=True)

# ── Figure layout ──────────────────────────────────────────────────────────────
tmax    = 50.0   # Myr shown in SED panel
nstride = 5
xmax    = 1e4
xscale  = 'log'

fig, axes = plt.subplots(
    3, 2, figsize=(12, 12),
    gridspec_kw=dict(width_ratios=(0.98, 0.02),
                     height_ratios=(1/3., 1/3., 1/3.),
                     wspace=0.05, hspace=0.11))

# ── Panel 1: dust opacity ──────────────────────────────────────────────────────
ax = axes[0, 0]
plt.sca(ax)
muH = (1.4 * au.u).cgs.value
d   = DustDraine()
df  = d.dfa['Rv31']
ax.semilogy(df.lwav * 1e4, df.Cext / muH,   c='k',      label='Extinction')
ax.semilogy(df.lwav * 1e4, df.kappa_abs,     c='k', ls='--', label='Absorption')
ax.set_xlim(1e2, xmax)
ax.set_ylim(5e1, 2e3)
ax.set_ylabel(r'$\kappa_{\rm d}(\lambda)\;[{\rm cm}^2\,{\rm g}^{-1}]$')
ax.legend(loc=2)

# Secondary y-axis: sigma_d per H atom
def kappa2sigma(x): return x * muH
def sigma2kappa(x): return x / muH
sax = ax.secondary_yaxis('right', functions=(kappa2sigma, sigma2kappa))
sax.set_ylabel(r'$\sigma_{\rm d}(\lambda)\;[{\rm cm}^2\,{\rm H}^{-1}]$')

# Band labels
ytext = 1.4e3
for label, x in [('LyC',  np.sqrt(912 * ax.get_xlim()[0])),
                  ('LW',   np.sqrt(912 * 1108)),
                  ('PE',   np.sqrt(1108 * 2068)),
                  ('NUV+OPT', np.sqrt(2068 * 1e4))]:
    ax.annotate(label, (x, ytext), xycoords='data', ha='center')

axes[0, 1].axis('off')

# ── Panel 2: SED evolution ─────────────────────────────────────────────────────
ax = axes[1, 0]
plt.sca(ax)
cmap = mpl.cm.jet_r
norm = Normalize(0., tmax)
dfg  = rr_short['df'].groupby('time_Myr')
logM = rr_short['logM']

for i, (t, df_) in enumerate(dfg):
    if t > tmax:
        continue
    if i % nstride == 0:
        ax.plot(df_.wav, df_.wav * 10.0 ** (df_.logf - logM),
                c=cmap(norm(t)))

ax.set_yscale('log')
ax.set_xlim(1e2, xmax)
ax.set_ylim(1e31, 1e37)
ax.set_ylabel(r'$\lambda L_\lambda/M_*\;[{\rm erg\,s^{-1}\,M_\odot^{-1}}]$')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, tmax))
plt.colorbar(sm, cax=axes[1, 1], label=r'$t_{\rm age}\;[{\rm Myr}]$')

# ── Panel 3: ISRF ─────────────────────────────────────────────────────────────
ax = axes[2, 0]
plt.sca(ax)

# SB99 ISRF — constant SFR (unattenuated thin, attenuated thick)
idx_all = w_ang >= 912
ax.loglog(w_ang,          Jlam_u * w_ang,
          c='gray', lw=1, alpha=0.7, label='_nolegend_')
ax.loglog(w_ang[idx_all], Jlam[idx_all] * w_ang[idx_all],
          c='gray', lw=2, alpha=0.9, ls='--',
          label=r'SB99 + const. SFR ($\Sigma_{\rm SFR,-3}=2.5$) + OML10')

# Zari et al. (2022) SFH histories
# H1: Sigma_SFR,-3 = 4 for t < 10 Myr, 2 otherwise
# H2: Sigma_SFR,-3 = 1 for t < 5 Myr, 4 for 5-10 Myr, 2 otherwise
ax.loglog(w_ang[idx_all], r_z1['Jlambda'][idx_all] * w_ang[idx_all],
          c='C0', lw=3, alpha=0.8,
          label=r'SB99 + Zari+22 ($\Sigma_{\rm SFR,-3}=4,t<10\,{\rm Myr}$) + OML10')
ax.loglog(w_ang[idx_all], r_z2['Jlambda'][idx_all] * w_ang[idx_all],
          c='C3', lw=3, alpha=0.8,
          label=r'SB99 + Zari+22 ($\Sigma_{\rm SFR,-3}=1,t<5\,{\rm Myr}$) + OML10')

# Reference ISRFs
wav_Dr = np.logspace(np.log10(912), np.log10(2500), 800) * au.angstrom
nu_Dr  = (ac.c / wav_Dr).to('Hz')
E_Dr   = (nu_Dr * ac.h).to('eV')
ax.semilogy(wav_Dr, nuJnu_Dr78(E_Dr),
            c='k', lw=1.5, label='Draine (1978)')

wav_MMP = np.logspace(np.log10(912), np.log10(1e4), 800) * au.angstrom
ax.semilogy(wav_MMP, Jlambda_MMP83(wav_MMP) * wav_MMP,
            c='green', ls='--', lw=1.5, label='Mathis et al. (1983)')

wav_vD = np.logspace(np.log10(1000), np.log10(2e4), 800) * au.angstrom
nu_vD  = (ac.c / wav_vD).to('Hz')
ax.semilogy(wav_vD,
            Jnu_vD82(wav_vD) * (ac.h * ac.c / wav_vD).to('erg') * nu_vD / (4 * np.pi),
            ls=':', c='k', lw=1.5, label='van Dishoeck & Black (1982)')

w_Hab = np.array([1000, 1400, 2200])
u_Hab = np.array([40e-18, 50e-18, 30e-18])
ax.semilogy(w_Hab, w_Hab * u_Hab * ac.c.cgs.value / (4 * np.pi),
            ls=':', c='magenta', lw=1.5, label='Habing (1968)')

# Observations
w_H80 = np.array([1195, 1250, 1300, 1350, 1400, 1450,
                  1500, 1550, 1600, 1650]) * au.angstrom
u_H80 = (np.array([5.29, 6.46, 6.53, 6.41, 5.29, 5.40,
                   5.22, 5.26, 5.41, 6.26]) * 1e-17
         * au.erg / au.cm**3 / au.angstrom)
J_H80 = (u_H80 / (4 * np.pi * au.sr) * ac.c).to('erg s-1 cm-2 sr-1 angstrom-1')
ax.scatter(w_H80.value, (w_H80 * J_H80).value,
           marker='s', facecolors='none', edgecolors='k', lw=1.1,
           label='Henry et al. (1980)')

w_G83 = np.array([1565, 1965, 2365, 2740]) * au.angstrom
F_G83 = (np.array([13.19, 9.89, 6.95, 5.37]) * 1e-7
         * au.erg / au.cm**2 / au.s / au.angstrom)
J_G83 = F_G83 / (4.0 * np.pi * au.sr)
ax.scatter(w_G83.value, (w_G83 * J_G83).value,
           marker='D', facecolors='none', edgecolors='k', lw=1.1,
           label='Gondhalekar et al. (1983)')

# UVB
r_uvb = read_FG20()
Jnu_uvb = r_uvb['ds']['Jnu']
ax.loglog(r_uvb['wav'], 1e3 * r_uvb['nu'] * Jnu_uvb.sel(z=0),
          c='gold', ls='-.', lw=1.5, label=r'$10^3\times{\rm UVB}\;(z=0)$',
          zorder=10)

ax.set_xlim(1e2, xmax)
ax.set_ylim(1e-5, 2e-3)
ax.set_ylabel(r'$\lambda J_\lambda\;[{\rm erg\,cm^{-2}\,s^{-1}\,sr^{-1}}]$')
ax.legend(fontsize='x-small', loc=2)

# Secondary y-axis: energy density
def nuJnu_to_u(x): return x / (ac.c.cgs.value / (4.0 * np.pi))
def u_to_nuJnu(x): return x * (ac.c.cgs.value / (4.0 * np.pi))
sax3 = ax.secondary_yaxis('right', functions=(nuJnu_to_u, u_to_nuJnu))
sax3.set_ylabel(r'$\lambda \mathcal{E}_\lambda\;[{\rm erg\,cm}^{-3}]$')

axes[2, 1].axis('off')

# ── Shared x-axis formatting ───────────────────────────────────────────────────
for ax in axes[:, 0]:
    ax.axvspan(100,  912,  color='grey', alpha=0.20)
    ax.axvspan(912,  1108, color='grey', alpha=0.15)
    ax.axvspan(1108, 2068, color='grey', alpha=0.10)
    ax.axvspan(2068, 1e4,  color='grey', alpha=0.05)
    ax.set_xlabel(r'$\lambda\;[\AA]$')
    ax.set_xlim(1e2, xmax)
    ax.set_xscale(xscale)

outfile = REPO / 'examples' / 'fig_sb99_sed_isrf.pdf'
fig.savefig(outfile, dpi=150, bbox_inches='tight')
print(f'Saved {outfile}')
plt.show()
