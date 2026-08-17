# Environment notes — `cosmic-emu`

One conda env (Python 3.12, numpy 2.x) runs **every active emulator**.
Recreate it with `bash scripts/setup_env.sh`; validate with
`python tests/smoke_env.py` (18 checks) and `python -m pytest tests/ -q`.

## What's installed and verified working

| Cluster | Packages |
|---|---|
| numpy/scipy | camb, symbolic_pofk (syren), pyspk, MiraTitanHMFemulator, csstemu (vendored), CCToolkit-style fits |
| sklearn GP | emantis, nDGPemu, subgrid_emu, GPy (for LaCE) |
| SEPIA GP | sepia (vendored patch), CubicGalileonEmu, subgrid_emu |
| JAX | jax 0.4.38 (pinned by jaxcapse), jax-cosmo, cosmopower-jax, jaxcapse, jaxeffort, baccoemu, picasso |
| PyTorch (CPU) | torch, gokunemu, LaCE-NN |
| TensorFlow | py21cmemu |
| compiled | euclidemu2 (conda-forge GSL), classy_sz |

## Vendored patches (in `external/`, applied by setup_env.sh)

These are one-line compatibility fixes to third-party code, all candidates
for upstream PRs:

1. **SEPIA** (`external/SEPIA`): scipy >= 1.11 removed
   `scipy.linalg.solve(..., sym_pos=True)`; replaced with the equivalent
   `assume_a='pos'` (4 sites in `SepiaPredict.py` etc.). This is the entire
   reason CubicGalileonEmu pinned scipy <= 1.10.1 — with the patch both
   CubicGalileonEmu and subgrid_emu run on scipy 1.17.
2. **csstemu** (`external/csstemu`): numpy 2 raises on assigning a
   shape-(1,) array into a scalar slot; `...predict(Normcosmo)` →
   `...predict(Normcosmo)[0]` (10 sites across `CEmulator/emulator/*.py`).
3. **LaCE** (`external/LaCE`): `gp_emulator.py` imports GPy inside
   `__init__` but uses it at module scope in `_build_interp`; added a
   module-level `import GPy`. Also: LaCE must be installed from the repo
   clone — the pip-from-git wheel omits the `data/sim_suites` training data.

## Install-order and pin gotchas

- `setuptools<81` — `pkg_resources` still needed by jax_cosmo and
  CubicGalileonEmu.
- **jaxcapse pins jax==0.4.x**; installing it together with latest-jax
  packages makes pip's resolver fail. Install jaxcapse first, everything
  else settles against its jax. If a future package needs jax >= 0.5,
  jaxcapse is the constraint to revisit.
- **euclidemu2's PyPI wheel** hardcodes a link path to `libgsl.28.dylib`
  inside a conda-style env; `conda install -c conda-forge gsl` into the env
  provides exactly that file (brew GSL does not).
- **subgrid_emu / CubicGalileonEmu** metadata pins (python<3.12, old numpy,
  scipy<=1.10.1) are stale — installed with
  `--no-deps [--ignore-requires-python]` and verified working. TODO: relax
  the pins upstream (both are our own repos).
- Emulator parameter-order traps (encoded in the tool layer):
  - jaxcapse: `[ln10As, ns, H0, wb, wc, tau]`
  - cosmopower-jax: `[wb, wc, h, tau, ns, ln10As]`
  - emantis: `logfR0` argument is **-log10|f_R0|** (positive, 4..7); time is `aexp`
  - CAMB: `halofit_version=` alone does NOT enable nonlinear P(k);
    `pars.NonLinear = NonLinear_pk` is required (was a silent-linear bug).
- Unit conventions verified numerically:
  - jaxcapse returns **Dl in muK^2** (matches CAMB to ~0.01%)
  - cosmopower-jax returns **raw dimensionless Cl** (CLASS convention);
    converted in tools/cmb with (T_CMB*1e6)^2 * l(l+1)/2pi
  - LaCE works in comoving Mpc (no h) — flagged in its tool schema
  - pyspk's fb is normalized by Omega_b/Omega_m

## Dropped as exact parallels (design decision)

- **hmcode (pip)**: requires numpy<2; CAMB's built-in HMcode-2020 is the
  same model — use `compute_nonlinear_pk(backend="camb_hmcode")`.
- **BCemu**: hard `smt==1.0.0` pin poisons a shared env; baryon suppression
  covered by baccoemu + pyspk + syren-baryon.
- **globalemu / 21cmVAE / 21cmLSTM**: py21cmemu covers global signal + PS.
- **FREmu / FORGE / mgemu**: e-MANTIS is the f(R) pick (FREmu would add
  massive-nu f(R) later; mgemu is a TF1 time capsule).

## Installed but deferred (no tools yet)

- **classy_sz / classy_szfast** — first use downloads ~GB of emulator data;
  pre-bake at deployment, then add tSZ/cluster-count tools.
- **py21cmemu** — first use downloads ~100s MB from HuggingFace.
- **forestflow** — installed; add a P3D tool alongside emulate_lya_p1d.
- **jaxeffort** — needs a trained model release from the
  CosmologicalEmulators zoo; PyBird covers multipoles meanwhile.

## Could not resolve (left out, revisit if needed)

- Nothing in the approved list is unresolved. The closest calls were
  csstemu/SEPIA/LaCE, all fixed by the vendored patches above.
