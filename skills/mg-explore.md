---
name: mg-explore
description: Explore modified-gravity signatures — f(R), nDGP, or cubic Galileon power spectrum boosts, parameter sweeps, and composed P(k) against the LCDM baseline
---

# Modified-gravity explorer recipe

Map out how a gravity model deviates from LCDM using the three MG boost
emulators, with correct composition against a LCDM baseline.

## Model dictionary

| model | parameter | meaning | range |
|---|---|---|---|
| `fofr` | `minus_log10_fR0` | Hu-Sawicki -log10\|f_R0\| (5 => \|f_R0\|=1e-5; larger = weaker MG) | 4-7 |
| `ndgp` | `H0rc` | crossover scale H0*rc (smaller = stronger MG) | 0.2-20 |
| `cubic_galileon` | `f_phi` | Galileon dark-energy fraction (larger = stronger) | 0.02-1.0 |

All three share a cosmology box around Om in [0.24, 0.39] — check
`describe_emulator` before unusual values.

## Workflow

1. Ask which model(s) and strength(s) if not stated; otherwise show one
   moderate case per model (fR0=1e-5, H0rc=1, f_phi=0.5).
2. **Single model:** `compute_mg_boost` at 2-3 parameter values, z=0 (and
   z=1 if the user cares about evolution); plot the boost files together
   with `plot_pk_comparison` (`reference_index` on the weakest case).
3. **Observable-level:** `compute_mg_pk` (records the LCDM baseline in
   metadata — always quote it), and optionally a plain
   `compute_nonlinear_pk` at the same cosmology to overlay MG vs LCDM.
4. Report: peak deviation (% and the k where it happens), the k-range above
   1% deviation, and — for cubic_galileon — the GP uncertainty from the
   `gp_std` column and the snapshot z actually used (`snapshot_z_used` in
   metadata, since z snaps to the training grid).

## Physics framing for the summary

- f(R): chameleon screening => boost peaks at k ~ 0.5-2 h/Mpc and shuts off
  at large scales; deviations grow toward low z.
- nDGP: Vainshtein screening => broader, flatter boost.
- Cubic Galileon: boost strength tracks f_phi; below f_phi ~ 0.1 the model
  is near-LCDM.
- For survey-detectability questions, pair with the pk-crosscheck skill:
  the MG signal must exceed the inter-emulator spread to be credible.
