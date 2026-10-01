---
name: mg-baryon-degeneracy
description: Quantify the degeneracy between modified gravity and baryonic feedback in P(k) — MG boost B(k) x baryon suppression S(k), cancellation scales where B*S = 1, spread across feedback models and redshift, and which hydro-simulation signatures mimic or hide f(R)/nDGP/Galileon gravity
---

# MG x baryons degeneracy recipe

MG boosts P(k) on nonlinear scales; feedback suppresses it on overlapping
scales. The joint signal is modeled as P = B(k) * S(k) * P_LCDM. This recipe
builds B*S grids efficiently and reports where and how robustly the two
effects cancel.

## 1. Fix ONE cosmology inside every box

Pick the cosmology first and reuse it for every call. It must sit inside
the MG emulators' box (Om 0.2365-0.3941, sigma8 0.6-1.0 for fofr; nDGP
also needs Ob 0.04-0.06, ns 0.92-1.0, As 1.7-2.5e-9, h 0.61-0.73) and the
baryon models' box (bacco: sigma8 0.73-0.90, z <= 1.5). Use
`convert_cosmology` if the user gives As instead of sigma8. Planck-like
defaults (Om 0.31, sigma8 0.81, h 0.67, ns 0.965, Ob 0.049) work everywhere.

## 2. Use the SAME k-grid and the SAME z for B and S

- Grid: `k_min=0.03, k_max=4.9, n_points=200` for both boosts and
  suppressions (bacco stops at 4.9 h/Mpc; SP(k) is calibrated only for
  k >= 0.1, below that S ~ 1 is padding, not a prediction).
- Redshift: compose_spectra refuses inputs at different z — compute B and
  S on the same z list.

## 3. Build the grids with scans, not single calls

- Boosts: `scan_emulator_parameters(tool="compute_mg_boost",
  scan={"minus_log10_fR0": [4, 5, 6], "z": [0, 0.5, 1]}, fixed_args={...
  cosmology, k grid ...})`; repeat with `model="ndgp"` (`H0rc`) or
  `"cubic_galileon"` (`f_phi`) as needed.
- Suppressions: `scan_emulator_parameters(tool="compute_baryon_suppression",
  scan={"model": ["spk", "bacco", "syren_IllustrisTNG", "syren_SIMBA"],
  "z": [0, 0.5, 1]}, fixed_args={... same cosmology and k grid ...})`.
  Feedback strength sweeps: bacco `log10_M_c` (13-15) or spk `fb_a`.
- Pair each boost with each suppression at matching z via `compose_spectra`
  (op multiply, `output_name` like `BS_F5_TNG_z0.5`). Each result's
  metadata gives `unity_crossings_k` — the cancellation scales. Quote
  them; never reload CSVs to find them.

## 4. Figures

- One figure per redshift with `plot_emulator_curves` over the B*S files,
  short `labels` (e.g. "F5 x TNG"); the y=1 line and linear axis are
  automatic.
- For a feedback-strength or z sweep of one pairing, pass `color_values`
  (the swept value per file) and `colorbar_label`.
- Show B, S and B*S together only for a single pairing (3 curves) — they
  share the dimensionless axis without any override.

## 5. Report

Per pairing: peak |B*S - 1| and its k; the cancellation scale(s); and the
**spread of the cancellation scale across baryon models** — that spread,
not any single model, is the robustness statement. Then the caveats below.

## Caveats you must state

- **Factorization.** B*S assumes MG and feedback act independently. The
  literature (SHYBONE f(R)+TNG, Arnold et al. arXiv:1907.02977) supports
  it at ~1-2% for |f_R0| <= 1e-6 and nDGP, ~3-5% for |f_R0| = 1e-5 at
  k > 5 h/Mpc; F4 has no modern hydro validation. Treat B*S departures
  smaller than this as unresolved.
- **SP(k) redshift scans.** fb_a/fb_pow are z-independent inputs; an SP(k)
  z-scan at fixed fb holds the gas fraction fixed while halos evolve, so
  its S(k, z) trend is an artifact. Prefer bacco or syren suites for z
  evolution, or supply z-appropriate fb values.
- **CRK-HACC subgrid emulator** (`emulate_subgrid_statistic`,
  statistic "Pk") gives S(k) on its own native k-grid (first CSV column,
  0.07-12.5 h/Mpc) at its fixed training cosmology, with a `gp_std` band.
  Compare it to other models only on overlapping k, and never rebuild
  its k-grid by hand.
- **Emulator boxes.** Any `in_training_box: false` in a scan run means
  that point is an extrapolation — say so next to the number.

## What this server cannot do (hand off honestly)

- Propagate B*S into lensing C_ell or tomography: `compute_lensing_cls` is
  LCDM-only (no P(k) modifier). See the lensing-fisher-inputs skill for
  what to compute here and what to hand to a coding task.
- Full MG+hydro simulations; the factorization caveat is the substitute.
