---
name: baryon-budget
description: Baryonic-feedback error budget — compare the hydro-simulation P(k) suppression S(k) across SP(k), bacco and four CAMELS suites (TNG, Astrid, SIMBA, EAGLE), sweep feedback strength, and connect it to CRK-HACC subgrid-physics predictions (gas fractions, stellar mass function) with GP uncertainties
---

# Baryonic feedback budget recipe

Quantify the dominant small-scale P(k) systematic — baryonic feedback — by
comparing independent models, and tie feedback strength to observable hydro
quantities.

## Part 1: suppression spread

1. One call: `scan_emulator_parameters(tool="compute_baryon_suppression",
   scan={"model": ["spk", "bacco", "syren_IllustrisTNG", "syren_Astrid",
   "syren_SIMBA", "syren_Swift_EAGLE"]}, fixed_args={cosmology,
   "k_min": 0.1, "k_max": 4.9, "n_points": 200})`. Notes:
   - spk defaults are a BAHAMAS-like fb; SP(k)'s fb is in units of
     Omega_b/Omega_m — say so if the user supplies numbers. SP(k) is
     calibrated for k >= 0.1 h/Mpc.
   - bacco defaults ~ moderate feedback; bacco stops at k = 4.9 h/Mpc and
     z = 1.5 (pass a lower k_max for the whole scan, not per model).
   - the four CAMELS suites run at the same A_SN/A_AGN (1.0 each =
     fiducial).
   Add `"z": [0, 0.5, 1]` to the scan for redshift evolution — but not with
   spk at fixed fb (its z-trend is an artifact; see the tool's note).
2. The scan returns a figure; for a paper version re-plot the files with
   `plot_emulator_curves` and short `labels` (e.g. ["SP(k)", "BACCO",
   "TNG", "Astrid", "SIMBA", "EAGLE"]). Feedback-strength sweeps (bacco
   `log10_M_c` 13-15, or spk `fb_a`) are one more scan each and plot with a
   colorbar automatically.
3. Report per model: maximum suppression (%) and the k where it occurs,
   plus the **envelope** — the min/max suppression across models at k = 1,
   5, 8 h/Mpc. The envelope IS the current theory uncertainty; SIMBA is
   typically the aggressive outlier.

## Part 2: subgrid physics connection (CRK-HACC)

If the user wants to know what feedback does besides suppress P(k):

1. `scan_emulator_parameters(tool="emulate_subgrid_statistic",
   combine="zip", scan={"v_kin": [0.2, 1.0], "e_kin": [0.1, 1.0]},
   fixed_args={"statistic": "Pk"})` — weak vs strong AGN in one call.
2. Repeat with `"statistic": "GSMF"` and `"fGas"` — show that the
   parameters that kill small-scale power also suppress massive galaxies
   and expel cluster gas.
3. Always mention the `gp_std` column: these are GP emulators with honest
   uncertainties; differences smaller than gp_std are not significant.

Comparing CRK-HACC to the Part 1 models:
- The x-axis is the FIRST CSV column, in real units (Pk: k in h/Mpc on
  the emulator's native 255-point grid, 0.07-12.5; fGas: halo mass; GSMF:
  stellar mass) — `metadata.x_axis` names it. Never reconstruct a grid by
  hand: a guessed grid once shifted S(k) by up to ~7x in k and
  reversed a published conclusion.
- The emulator is at its fixed training cosmology; evaluate the other
  models at that cosmology too, compare only on the overlapping k range,
  and quote S at the same physical k from each model.

## Composing with gravity-only P(k)

To hand the user a baryon-corrected spectrum, do it SERVER-SIDE:
`baryonify_pk(pk_file=<compute_nonlinear_pk output>, model=..., params...)`
— it evaluates the suppression on the input file's own k-grid and z, and
records full provenance. For an MG universe with feedback, feed it the
output of `compute_mg_pk`. Generic arithmetic between any two CSVs is
`compose_spectra`. Never multiply numbers client-side. Alternative:
`baccoemu` end-to-end (gravity + baryons self-consistently in one
emulator).
