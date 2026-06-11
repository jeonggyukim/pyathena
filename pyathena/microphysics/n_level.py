"""5-level statistical-equilibrium solver for metal-ion cooling.

All the followed coolants in the photchem subpackage use 5 levels:

- np^2 ions (CI, NII, OIII, SIII): 3P_0 / 3P_1 / 3P_2 / 1D_2 / 1S_0
- np^3 ions (NI, OII, SII): 4S°_3/2 / 2D°_3/2 / 2D°_5/2 / 2P°_1/2 / 2P°_3/2
- np^4 ions (OI, SI): 3P_2 / 3P_1 / 3P_0 / 1D_2 / 1S_0 (Hund's 3rd inverts)

Levels are indexed by EXCITATION ENERGY (level 0 = ground, level 4 =
highest of the 5). This is the standard nebular truncation; higher
CHIANTI levels (typically 2s 2p^3, 2p^4, Rydberg states) sit above
60000 K = exp(-6) ~ 0.002 Boltzmann factor at HII temperatures and
contribute negligibly to cooling and to ionization-relevant level
populations.

Two-level ions (C II 2P) are handled separately, not via this module.
"""

import numpy as np

NLEV = 5

# Boltzmann constant in erg / K (CGS), used only here for the
# detailed-balance upward-rate factors. Imported by ion files via
# the helper functions.
KB_CGS = 1.380649e-16


def solve_5level_steady_state(A, E, g, T, n_coll_times_q):
    """Solve the 5-level steady-state population.

    Parameters
    ----------
    A : (5, 5) array
        Einstein A coefficients [s^-1]. `A[i, j]` is the spontaneous
        decay rate from upper level `i` to lower level `j` (nonzero
        for `i > j`).
    E : (5,) array
        Level energies above ground [erg]. `E[0] = 0`.
    g : (5,) array
        Statistical weights `g_i = 2 * J_i + 1`.
    T : scalar or array
        Temperature [K].  Used for the Boltzmann upward-rate factors.
    n_coll_times_q : (5, 5, ...) array
        Total downward collisional rate matrix already summed over
        colliders and multiplied by their densities, i.e.
            n_coll_times_q[i, j] = sum_c n_c * q^c[i, j]
        for i > j (downward rate from level i to level j). Trailing
        axes hold per-cell broadcasting. Each ion file builds this
        from its tabulated rate fits and the local densities.

    Returns
    -------
    f : (5, ...) array
        Level fractional populations summing to unity along the
        leading axis.

    Notes
    -----
    For each cell the routine builds the 5x5 rate-balance matrix
    A_mat such that A_mat @ f = b, where the first 4 rows enforce
    d f_i / dt = 0 (sum of in-rates = sum of out-rates) and the
    last row enforces sum_i f_i = 1. The system is then solved
    with numpy.linalg.solve. For N cells this loops over cells
    once -- acceptable for diagnostic test plots; the production
    sweep uses the offline-built (T, n_e) lookup tables instead.
    See Osterbrock & Ferland 2006 sec 3.5 or Draine 2011 Eqs.
    17.3-17.5 for the derivation.
    """
    A = np.asarray(A, dtype=float)
    E = np.asarray(E, dtype=float)
    g = np.asarray(g, dtype=float)
    if A.shape != (NLEV, NLEV) or E.shape != (NLEV,) or g.shape != (NLEV,):
        raise ValueError("A must be 5x5; E and g must each be length 5")
    T = np.asarray(T, dtype=float)
    Cdown = np.asarray(n_coll_times_q, dtype=float)
    if Cdown.shape[:2] != (NLEV, NLEV):
        raise ValueError("n_coll_times_q must have leading shape (5, 5)")

    # Cell-loop output shape
    cell_shape = Cdown.shape[2:] if Cdown.ndim > 2 else ()
    if cell_shape:
        f = np.zeros((NLEV,) + cell_shape, dtype=float)
        # Broadcast T to cell_shape
        T_b = np.broadcast_to(T, cell_shape)
        for idx in np.ndindex(*cell_shape):
            T_i = float(T_b[idx])
            C_i = Cdown[(slice(None), slice(None)) + idx]
            f[(slice(None),) + idx] = _solve_one(A, E, g, T_i, C_i)
        return f
    else:
        return _solve_one(A, E, g, float(T), Cdown)


def _solve_one(A, E, g, T, Cdown):
    """Solve one cell.  All inputs are bare arrays / scalars."""
    R = np.zeros((NLEV, NLEV), dtype=float)
    kT = KB_CGS * T
    for i in range(NLEV):
        for j in range(NLEV):
            if i == j:
                continue
            if i > j:
                # Downward: spontaneous + collisional
                R[i, j] = A[i, j] + Cdown[i, j]
            else:
                # Upward by detailed balance from the (j, i) downward
                # collisional rate. Energy gap E[j] - E[i] > 0.
                R[i, j] = (
                    Cdown[j, i] * (g[j] / g[i]) * np.exp(-(E[j] - E[i]) / kT)
                )

    # Steady state: sum_j R[j, i] f_j - f_i sum_j R[i, j] = 0
    M = R.T - np.diag(R.sum(axis=1))
    # Detect isolated / dummy levels (no rates in or out). Happens
    # for ions with fewer than NLEV physical levels (e.g. 2-level
    # CII / NeII) padded with high-E dummies. Force f = 0 there.
    isolated = (R.sum(axis=1) == 0) & (R.sum(axis=0) == 0)
    for k in np.where(isolated)[0]:
        M[k, :] = 0.0
        M[k, k] = 1.0
    # Closure sum_i f_i = 1. Put it on the ground level (row 0),
    # which is always non-isolated; the last row may be a dummy.
    M[0, :] = 1.0
    b = np.zeros(NLEV)
    b[0] = 1.0
    return np.linalg.solve(M, b)


def cooling_from_populations(f, A, E):
    """Cooling rate per ion [erg s^-1].

    Lambda / n_X = sum_{i > j} f_i * A[i, j] * (E[i] - E[j]).
    """
    A = np.asarray(A, dtype=float)
    E = np.asarray(E, dtype=float)
    f = np.asarray(f, dtype=float)
    dE = E[:, None] - E[None, :]                       # (5, 5)
    # Per-cell sum over all (i, j) with i > j of f[i] A[i,j] dE[i,j]
    return np.einsum('i...,ij,ij->...', f, A, dE)
