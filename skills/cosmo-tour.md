---
name: cosmo-tour
description: Guided tour of this server — one representative computation per tool family (P(k), modified gravity, CMB, clustering, lensing, baryons, halos, Lyman-alpha), ending with a capability summary. Use for demos or "what can this server do?"
---

# Cosmic emulator tour

Run one representative computation per family of this server and summarize.
Total runtime is under a minute; every step returns a CSV or PNG artifact.

## Before starting

1. Call `list_emulators` and show the family/status table to the user.
2. Pick an output directory: use what the user gave, else `output/tour`
   (on a hosted server any path is remapped under the artifact root and
   results carry browsable URLs — share those URLs, do not read the files
   back).

## The tour (run in this order)

1. **Matter power** — `compute_nonlinear_pk` twice at the same cosmology
   with `backend="baccoemu"` and `backend="csst"`, then `plot_pk_comparison`
   with both files. Point out the ratio panel: two independent simulation
   suites agreeing at the percent level.
2. **Modified gravity** — `compute_mg_boost` with `model="cubic_galileon"`,
   `f_phi=0.8`. Note the `gp_std` column: this emulator reports its own
   uncertainty. Then `compute_mg_pk` with `model="fofr"` to show boost x
   baseline composition (metadata records the baseline used).
3. **CMB** — `compute_cmb_cls` with `backend="capse"` and again with
   `backend="cosmopower_jax"`, then `plot_cmb_spectra` with both. Two
   different neural emulators, two different native conventions, one Dl
   [muK^2] output — they overlay to sub-percent.
4. **Galaxy clustering** — `compute_linear_pk` (`backend="camb"`,
   `k_min=1e-4`, `k_max=1.0`, `n_points=300`, `z=0.5`), then feed that file
   to `compute_galaxy_multipoles` at the same z. One-loop EFTofLSS from any
   linear spectrum.
5. **Weak lensing** — `compute_lensing_cls` with defaults.
6. **Baryons** — `compute_baryon_suppression` with `model="spk"` and
   `model="syren_IllustrisTNG"`, plot both with `plot_pk_comparison`.
   Mention the inter-suite spread as a real systematic.
7. **Halos** — `compute_hmf` with defaults; note the emulator_std column.
8. **Lyman-alpha** — `emulate_lya_p1d` with defaults; flag the unit switch
   (comoving Mpc, no h — deliberate, matches DESI convention).

## Wrap-up

End with a compact summary table: family | tool used | headline number
(e.g. "nonlinear P(k), two suites agree to 2%"). Mention the deeper
skills available via `load_skill`: pk-crosscheck, mg-explore,
baryon-budget.

## Cautions

- Stay inside parameter ranges — call `describe_emulator` if the user
  requests an unusual cosmology; quote the violated range if a tool errors.
- Pass file paths between tools, never paste CSV contents into context.
