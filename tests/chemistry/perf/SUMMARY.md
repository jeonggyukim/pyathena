# Phase 4.5 -- go/no-go for sorted-strip dispatch

Measured 2026-07-30 on Apple arm64 (NEON, 128-bit vectors), Python
3.13, NumPy 2.4.6, numba 0.66.0. Raw outputs in `results/`. Reproduce
with:

```
conda run -n pyathena python -m pytest tests/chemistry/perf --bench -q -s
```

These numbers are from the solver **after** the 2026-07-30 fix to the
per-H to per-volume conversion in `cooling/base.py` (see
`pyathena/chemistry/CHANGES.md`). The first run of these benchmarks,
on 2026-07-29, was made against a solver whose cooling rate was
independent of density; every per-regime substep count in it was an
artifact of that defect. Do not cite the earlier figures.

## Standing

Superseded as a decision input. The project chose on 2026-07-29 to try
rate tabulation and within-cell vectorization first
(`ai-notes/docs-claude/pyathena/rate-tabulation-scoping.md`). Sorted
strips are held in reserve. What follows records what was measured, not
a recommendation to build.

| Deliverable | Plan threshold | Measured | Reading |
|---|---|---|---|
| 1 strip-size sweet spot | knee >= 8 cells | saturation past ~1000 cells | Python-only number |
| 2 sorted vs mixed | > 2x | 1.45x (exp/log), 1.02x (poly) | below threshold |
| 3 SIMD ceiling | > 3x | 0.48x | inconclusive |
| 4 strip-split load balance | (not in plan) | 3.39x waste, 2.5x recovered | the substantive result |

## Deliverable 4 -- strip splitting

`ExplicitSubcyclingSolver` takes a strip-MIN `dt_sub`, so every cell
walks the substep count set by the stiffest cell in its strip. Substep
counts per regime over a 1 Myr step, after the cooling fix:

| regime | n_H [cm^-3] | T [K] | nsub |
|---|---|---|---|
| cold molecular | 1e3 | 50 | 37 |
| CNM | 5e1 | 100 | 40 |
| WNM | 5e-1 | 8e3 | 1 |
| HII | 1e1 | 1e4 | 142 |
| hot | 1e-2 | 1e6 | 1 |

The stiffest regime is HII, and the two lowest-density regimes now cost
one substep each -- the reverse of the pre-fix ordering, where hot gas
was stiffest at 350 substeps.

On a 256-cell strip sampled uniformly across the five regimes:

| | cell-substeps | vs ideal |
|---|---|---|
| ideal (each cell in its own strip) | 11,161 | 1.00x |
| one strip (what the solver does today) | 37,888 | 3.39x |
| split into 4, unsorted | 37,888 | 3.39x |
| split into 4, sorted by stiffness | 15,232 | 1.36x |
| split into 8, sorted | 13,408 | 1.20x |
| split into 16, sorted | 12,544 | 1.12x |

A single strip discards 70 percent of its chemistry work. Unsorted
splitting recovers none of it: a random chunk almost always contains an
HII cell, so the strip-MIN is unchanged. Sorting by stiffness and
splitting into 4 cuts total work 2.5x, and the gain saturates near a
1.1x floor by 8 chunks, the residual being chunk boundaries that do not
align with regime boundaries.

The unit is cell-substeps, not seconds, so these counts hold for the
C++ port as they hold for the NumPy one. Two limits on reading them.
The ratio bounds the speedup from above: splitting multiplies the fixed
per-strip cost, which is severe in Python and small in C++. And 3.39x
is a property of the regime mix, sampled uniformly here rather than
drawn from a simulation snapshot; a domain that is 99 percent CNM would
show a ratio near 1.0.

## Deliverable 3 -- SIMD ceiling (inconclusive)

Branch-free and branched numba kernels over the two gates the solver
takes (the `temp_hot1` gate, the `x_H2 < 0.25` split in the Cramer
solve). Branch-free measured 0.48 to 0.57x the throughput of branched
across all four (fastmath, arithmetic-form) combinations, which is to
say consistently slower, by the ratio of their flop counts.

Getting exactly the flop ratio is consistent with two different
situations: neither form vectorizes, or both vectorize equally. Those
imply opposite conclusions and the measurement does not separate them.
Attempts to settle it from the generated assembly failed. What was
established is that numba on this machine does vectorize a trivial
`a[i]*b[i] + c` loop (`fmla.2d`, 21 q-registers), so the toolchain is
capable of it and the null is specific to the kernel shape.

These kernels are independent of the solver, so the cooling fix did not
change them, and the figures match the 2026-07-29 run to two digits.

The plan's premise -- that a Python-level throughput ratio bounds the
C++ SIMD payoff -- does not hold regardless. numba and a C++ compiler
differ in vectorizer, in math library (SVML on x86, nothing equivalent
here), and in vector width (NEON 128-bit against the AVX2 256-bit
target). Answer this one in C++ with a compiler vectorization report.

## Deliverable 2 -- sorted vs mixed strip

Branched numba kernel, identical cells, permuted order: 1.45x with
exp/log path bodies, 1.02x with polynomial bodies of similar flop
count. Below the plan's 2x bar.

The split between the two is informative. Cheap side-effect-free branch
bodies get if-converted into predicated code, leaving no branch to
mispredict, so ordering cannot matter. Expensive library calls cannot
be if-converted, so ordering does matter.

## Deliverable 1 -- strip size (a pyathena number, not a C++ one)

Throughput in cell-substeps per second against strip width, with
marginal scaling per doubling (1.0 is perfect linear scaling):

| ncell | cell-substeps/s (CNM) | marginal |
|---|---|---|
| 1 | 1.71e3 | -- |
| 8 | 1.85e4 | 0.99 |
| 64 | 1.36e5 | 0.95 |
| 2048 | 1.01e6 | 0.61 |

Scaling stays near linear out to 64 cells and falls to 0.61 by 2048,
which is where the strip stops fitting in cache. The whole range below
about 600 cells is dominated by NumPy dispatch rather than arithmetic:
per substep there is 0.44 ms of width-independent overhead against
0.74 us per cell-substep of marginal cost, and the measured ufunc
dispatch floor is a flat 0.125 us per call from n=1 to n=256.

For the Python solver this sets a strip window of **512 to 4096
cells**: at least 512 to clear the dispatch floor, at most about 4096
because a strip carries 210 scratch buffers at 1875 bytes per cell, so
4096 cells is 7.7 MB. Chemistry is cell-local, so strips may be
flattened across j and k rather than following i-rows, which is
necessary since a 64^3 block has only 64 cells along i.

None of this constrains the C++ `CHUNK_SIZE`. C++ has no dispatch floor
to amortize and fuses most of those scratch buffers into registers, so
it wants a small strip sized to cache and vector width, for reasons
opposite to the ones that set the Python window.

## What these benchmarks found that the plan did not ask for

The 2026-07-29 run produced a hot regime needing 350 substeps and
collapsing from 1e6 K to 1e3 K within 1 Myr. Chasing that led to a
missing factor of `n_H` between the cooling aggregator and the
temperature kernel, which made the cooling rate independent of density
and left the solver correct only at `n_H = 1` cm^-3. Fixed 2026-07-30,
with a regression test asserting that `d(temp_mu)/dt` is linear in
`n_H`.

A second defect found the same way is still open: pyathena has no
hot-gas regime branch on cooling, where the C++ `EvaluateThermal`
blends to CIE between `temp_hot0` and `temp_hot1` and uses CIE only
above `temp_hot1`. At 1e6 K `GrainRecombinationCooling` gives
`Lambda_e = 3.45e-23` against a metal-free CIE total of 5.00e-24.
