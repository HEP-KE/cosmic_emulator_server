---
name: baryon-budget
description: Build a baryonic-feedback error budget — compare P(k) suppression across SP(k), bacco, and four CAMELS hydro suites, and connect feedback strength to CRK-HACC subgrid predictions with GP uncertainties
---

# Baryonic feedback budget recipe

Quantify the dominant small-scale P(k) systematic — baryonic feedback — by
comparing independent models, and tie feedback strength to observable hydro
quantities.

## Part 1: suppression spread

1. Call `compute_baryon_suppression` for:
   - `model="spk"` (defaults are a BAHAMAS-like fb; SP(k)'s fb is in units
     of Omega_b/Omega_m — say so if the user supplies numbers)
   - `model="bacco"` (defaults ~ moderate feedback)
   - all four CAMELS suites: `syren_IllustrisTNG`, `syren_Astrid`,
     `syren_SIMBA`, `syren_Swift_EAGLE` at the same A_SN/A_AGN (1.0 each =
     fiducial)
2. Plot all files with `plot_pk_comparison` (no ratio panel needed —
   suppression curves are already ratios; pick any `reference_index`).
3. Report per model: maximum suppression (%) and the k where it occurs,
   plus the **envelope** — the min/max suppression across models at k = 1,
   5, 8 h/Mpc. The envelope IS the current theory uncertainty; SIMBA is
   typically the aggressive outlier.

## Part 2: subgrid physics connection (CRK-HACC)

If the user wants to know what feedback does besides suppress P(k):

1. `emulate_subgrid_statistic` with `statistic="Pk"` at weak
   (v_kin=0.2, e_kin=0.1) and strong (v_kin=1.0, e_kin=1.0) AGN settings.
2. Repeat for `"GSMF"` and `"fGas"` at the same two settings — show that
   the parameters that kill small-scale power also suppress massive
   galaxies and expel cluster gas.
3. Always mention the `gp_std` column: these are GP emulators with honest
   uncertainties; differences smaller than gp_std are not significant.

## Composing with gravity-only P(k)

To hand the user a baryon-corrected spectrum, do it SERVER-SIDE:
`baryonify_pk(pk_file=<compute_nonlinear_pk output>, model=..., params...)`
— it evaluates the suppression on the input file's own k-grid and z, and
records full provenance. For an MG universe with feedback, feed it the
output of `compute_mg_pk`. Generic arithmetic between any two CSVs is
`compose_spectra`. Never multiply numbers client-side. Alternative:
`baccoemu` end-to-end (gravity + baryons self-consistently in one
emulator).
