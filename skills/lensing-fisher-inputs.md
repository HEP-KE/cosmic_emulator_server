---
name: lensing-fisher-inputs
description: Prepare weak-lensing Fisher-forecast inputs — finite-difference C_ell derivatives in Omega_m and sigma8, source-redshift (tomographic-bin) sets, step-size checks, and MG/baryon B(k)*S(k) tables to hand to a coding task; states what this server computes and what it cannot
---

# Weak-lensing Fisher inputs recipe

`compute_lensing_cls` gives the cosmic-shear convergence C_ell (jax-cosmo,
Limber) for ONE Smail source bin per call, in LCDM/wCDM: parameters Om,
Ob, h, ns, sigma8, w0. Use this server for the parts it computes exactly;
hand the rest to a coding task with a precise file list.

## 1. Fiducial and derivative spectra (this server)

Keep `ell_min`, `ell_max`, `n_ell` IDENTICAL in every call — derivatives
difference files row by row, so the ell grids must match.

Central differences in one call each, with `combine="zip"`:

    scan_emulator_parameters(tool="compute_lensing_cls", combine="zip",
      scan={"Om":     [0.31, 0.30, 0.32, 0.31, 0.31],
            "sigma8": [0.81, 0.81, 0.81, 0.80, 0.82]},
      fixed_args={"z_source": 1.0, "ell_min": 100, "ell_max": 5000,
                  "n_ell": 80})

Several source bins: add `"z_source"` to the zip lists (repeat the five
points per bin), or run one scan per bin. The manifest JSON maps every
parameter set to its file — pass the manifest to the coding task.

**Step-size check:** repeat with half the step (0.005 instead of 0.01) and
report whether the derivatives agree to a few percent. Do it before
quoting any Fisher error.

## 2. MG and baryon response tables (this server)

The lensing tool takes no P(k) modifier, so the MG/baryon parameters enter
through tables the coding task integrates:

- B(k, z) and S(k, z) on one k-grid (`k_min=0.03, k_max=4.9`) at
  z in [0, 0.5, 1, 1.5, 2] for the fiducial AND for the +/- steps of each
  MG or feedback parameter (e.g. minus_log10_fR0 5 +/- 0.25, bacco
  log10_M_c 14 +/- 0.25) — use scans as in the mg-baryon-degeneracy skill
  (keep bacco at z <= 1.5; say what you did above it).
- A fiducial nonlinear P(k) at the same z list (`compute_nonlinear_pk`,
  `scan={"z": [...]}`).

## 3. The coding-task handoff

Give the coding task: the manifest(s); which column is which (C_ell files:
`ell, Cl_kappa`; boost/suppression: `k_h_per_Mpc` plus `boost` or
`suppression`); the survey assumptions it must add itself (shape noise
sigma_e, n_eff per bin, f_sky); and the instruction to compute
Delta C_ell / C_ell from P_LCDM * B * S with the same Limber kernel.

## Caveats you must state

- Separate z_source calls are AUTO-spectra of independent single bins: no
  cross-bin spectra and no cross-bin covariance. A "5-bin" Fisher built
  from them treats bins as independent — an optimistic approximation; say
  so, or have the coding task compute the full tomographic matrix.
- k > 4.9 h/Mpc is beyond bacco and is reached at high ell / low z:
  report the ell_max actually supported by the tables, or the
  extrapolation used.
- Gaussian covariance only; no super-sample or non-Gaussian terms unless
  the coding task adds them.
