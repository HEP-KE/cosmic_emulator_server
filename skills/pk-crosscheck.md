---
name: pk-crosscheck
description: Cross-validate the nonlinear matter power spectrum across all six P(k) backends at one cosmology and quantify the inter-emulator spread — the honest error bar on any emulated P(k)
---

# Nonlinear P(k) cross-check recipe

Quantify emulator systematics by running every nonlinear backend at the
same cosmology and reporting the spread.

1. Confirm the cosmology with the user (defaults: Om=0.31, Ob=0.049,
   h=0.67, ns=0.965, As=2.1e-9, mnu=0, w0=-1, wa=0, z=0). Check each
   backend's box with `describe_emulator` (keys: `baccoemu`, `euclidemu2`,
   `csst`, `gokunemu`, `cosmicemu_mt4`, `camb`, `syren`) — for non-vanilla cosmologies some
   backends will be out of range; run the subset that is valid and say so.
2. Call `compute_nonlinear_pk` once per backend, same parameters and k-grid
   (suggest `k_min=0.01`, `k_max=4.5`, `n_points=200`): `baccoemu`,
   `euclidemu2`, `csst`, `gokunemu`, `miratitan`, `camb_hmcode`,
   `syren_halofit`. miratitan (Mira-Titan IV / CosmicEmu, HACC) has the
   narrowest box (sigma8 0.7-0.9, z <= 2) — drop it when out of range and
   say so.
3. Plot all files with `plot_pk_comparison`, `reference_index` pointing at
   `baccoemu` (or the user's preferred reference).
4. Read the per-backend metadata (never the CSVs themselves) and report:
   - max |ratio - 1| per backend vs the reference, and at which k
   - the k-range where all backends agree within 2%
   - which backend to use for the stated purpose (speed -> syren; wide
     cosmology -> gokunemu; k to 10 -> csst/euclidemu2; baryon-ready ->
     baccoemu)

Interpretation guide:

- 1-3% scatter for k <= 2 h/Mpc at LCDM is expected and healthy; larger
  spread flags extrapolation — recheck ranges.
- camb_hmcode is a halo-model fit (~2.5% by construction), not an N-body
  emulator; treat it as the outlier detector, not truth.
- If the user needs baryons on top, continue with the baryon-budget skill.
