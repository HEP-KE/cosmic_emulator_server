# Cosmological Emulator Catalog

**Purpose:** master inventory of public cosmological-emulator codes for building the
`cosmic_emulator_server` MCP server — a production, CPU-hosted service exposing many
emulators as tools, organized in sub-directories with skills for navigation.

**Compiled:** 2026-08-16, from a five-track web sweep (repos, PyPI, readthedocs,
arXiv) covering matter power spectra, modified gravity / beyond-ΛCDM, CMB and
angular statistics, baryons / hydro / clusters / IGM, and field-level emulators
plus emulation frameworks.

**How to read the tables.** `MCP` column is a deployment verdict for a CPU-only 4 GB host:
- **A** — wrap first: pip-clean, ms-scale eval, small data, no exotic pins
- **B** — worth wrapping, heavier: compiled deps, big weight downloads, version pins, or GP-era stacks
- **C** — catalog only: GPU-class, legacy/frozen environments, or framework rather than emulator
- ⚠ marks a known gotcha explained in the notes under each table.

Accuracy figures and parameter ranges are the papers'/repos' claims.
**Verify ranges against the shipped model metadata before hard-coding them into tool schemas.**

---

## 1. Matter power spectrum — P(k)

### 1.1 Reference codes (ground truth, not emulators)

| Code | Output | Install | License | MCP | Notes |
|---|---|---|---|---|---|
| CAMB | linear + halofit/HMcode P(k), any cosmology | `pip install camb` | LGPL-mod | A | ~1 s/eval; natural "exact mode" |
| CLASS | linear + halofit/HMcode P(k) | `pip install classy` (compiles) | CLASS/GPL-like | A | already wrapped in spectra-mcp-server |
| HMcode-python | nonlinear semi-analytic P(k), baryon feedback (log10 T_AGN) | `pip install hmcode` (+ camb) | MIT | A | ~2.5% (HMcode-2020); ~0.1 s |
| NGenHalofit | recalibrated halofit, ≲1% Planck-neighborhood | Bitbucket, compile C | — | C | dormant; superseded in practice |

### 1.2 Linear emulators & differentiable solvers

| Emulator | Output | k-range | z | Install | Deps | License | MCP |
|---|---|---|---|---|---|---|---|
| **symbolic_pofk / syren** | closed-form linear + nonlinear + baryon P(k); ν+w0wa (syren-new) | 9e-3–9 h/Mpc | ≲3 | git + pip | numpy only | MIT | **A** |
| CosmoPower (PKLIN/PKNLBOOST) | linear P(k) + NL boost on fixed grid | 1e-5–10 Mpc⁻¹ | 0–5 | `pip install cosmopower` | TensorFlow ⚠ | GPL-3 + non-commercial rider ⚠ | B |
| cosmopower-jax | loads any CosmoPower model, differentiable | per model | per model | `pip install cosmopower-jax` | JAX | GPL-3 + NC rider ⚠ | A |
| DISCO-EB (DISCO-DJ) | differentiable Boltzmann solve, per-mille vs CLASS | any | any | git + pip | JAX, diffrax | GPL-3 | B |
| Mapse.jl / jaxmapse | linear P(k) (2026, docs thin) | TBD | TBD | Julia / pip | Julia or JAX | MIT | watch |
| emuPK | GP linear P(k) (KiDS ranges) | — | — | git | sklearn-era | — | C |

### 1.3 Nonlinear emulators — absolute P(k)

| Emulator | Params (n) | k-range [h/Mpc] | z | Accuracy | Install | Deps | License | MCP |
|---|---|---|---|---|---|---|---|---|
| **CSST emulator (csstemu)** | 8 (νw0wa) | lin 1e-5–100; NL 0.006–10 | 0–3 | 1% to k=10, z≤2 | git + pip | numpy/scipy only | MIT | **A** |
| **GokuNEmu** | 10 (νw0wa + Neff + αs) | 0.006–10 | 0–3 | ~0.5% avg | `pip install gokunemu` | PyTorch (CPU ok) | MIT | **A** |
| CosmicEmu (Mira-Titan IV) | 9 (νw0wa) | to ~5 **Mpc⁻¹, h-free units** ⚠ | 0–2.02 | 2–3% | compile C + GSL | C, GSL | LANL BSD-style | B |
| DarkEmulator2 (Dark Quest II) | 9 (νw0wa + Ω_K) | to k_Ny ≈ 10 | paper | sub-% | `pip install dark_emulator2` | NN (torch?) | MIT | B (new 2026) |
| aemulus_heft | 7 (wνCDM) + HEFT bias spectra | ≤1 (1%), ≤4 (2%) | 0–3 | 1–2% | git + pip | numpy, velocileptors | MIT | B |

### 1.4 Boost emulators — B(k) = P_nl/P_lin (compose with a linear source)

| Emulator | Params | k-range [h/Mpc] | z | Accuracy | Install | Deps | License | MCP |
|---|---|---|---|---|---|---|---|---|
| **baccoemu** | 8–9 (νw0wa, **σ8_cold/Ω_cold** ⚠) | boost 0.01–5 (Angulo21) / –10 (Aricò23); lin 1e-4–50 | ≤1.5 / ≤3 | 1–3% | `pip install baccoemu` | JAX (was TF ⚠) | MIT | **A** |
| **EuclidEmulator2** | 8 (νw0wa, Σmν ≤ 0.15 eV) | 8.7e-3–9.41 | 0–10 (val. ≤3) | ~1% | `pip install euclidemu2` | C++, GSL ≥2.5 ⚠ | GPL-3 | **A/B** |

Notes:
- **CosmicEmu unit trap:** k in 1/Mpc, P in Mpc³ — *no h*. Every other entry uses h/Mpc and (Mpc/h)³.
- **BACCO parameter trap:** takes σ8_cold and Ω_cold (cold matter = CDM+baryons), not total-matter quantities.
- **symbolic_pofk** is the standout MCP citizen: zero data files, microsecond evals, numpy-only, MIT. Ideal default tool with emulators layered above it.
- CosmoPower-branded code carries a **non-commercial rider on GPL-3** — flag before redistributing weights.
- "EmulateApost" from the original brief: no such repo exists; nearest matches are `emuPK` and `EmulateLSS`.

---

## 2. Modified gravity & beyond-ΛCDM

Nearly all MG emulators return a **boost relative to ΛCDM** and must be composed with a ΛCDM nonlinear P(k) source (§1.3/1.4).

| Emulator | Model | MG params | k [h/Mpc] | z | Accuracy | Install | Deps | License | MCP |
|---|---|---|---|---|---|---|---|---|---|
| **CubicGalileonEmu** (Ramachandra) | cubic Galileon | f_φ ∈ [0.02, 1.0] + 4 cosmo | 0.02–12.06 | 0–49 (51 snaps) | n/a (paper in prep) | git + pip -e | SEPIA (git dep), **scipy ≤ 1.10.1** ⚠ | Apache-2.0 | **A/B** ⚠ |
| **e-MANTIS** | f(R) HS (+wCDM HMF) | \|f_R0\| ∈ [1e-7, 1e-4], Ωm, σ8 | 0.03–7 | 0–2 | <1–3% | `pip install emantis` | sklearn | GPL-3+ | **A** |
| nDGPemu | nDGP | H₀r_c ∈ [0.2, 20] + 5 cosmo | ≤5 | 0–2 | ~3% | git + pip | sklearn MLP | GPL-3 | **A** |
| FREmu / FREmus | f(R) HS + ν | f_R0, Mν + 5 cosmo | 0.009–0.5 (FREmus →1) | 0–3 | <5% | `pip install fremu` | torch, camb | MIT | B |
| FORGE emulator | f(R) HS | log10\|f_R0\| ∈ [−6.2, −4.5], wide Ωm/σ8 | ≤10 | 0–2 (9 snaps) | ~2.5% | Bitbucket clone | sklearn-era | — | B |
| reactemu-fr | f(R)+ν+baryons (ReACT-trained) | broad | <1 | 0–5 | ~3% | git (cosmopower org) | TF/cosmopower ⚠ | GPL-3+NC | B |
| DS-emulators | Dark Scattering IDE | A_ds ∈ [−30, 30] b/GeV + 9 | ~1e-4–5 | 0–5 | ~ReACT | git | TF/cosmopower ⚠ | GPL-3 | B |
| CosmoPower EDE zoo | early dark energy | f_EDE, log10 z_c, θ_i | (CMB + P(k)) | — | ≪0.1σ ACT DR6 | git (cosmopower org) | TF **or plain-numpy .npz** | unstated ⚠ | B |
| Sesame (pipeline + demo f(R)+ν) | any (COLA-train-your-own) | per model | ≤3–5 | per model | ~1–2% | git | torch | none ⚠ | B/C |
| mgemu (LSST DESC) | f(R) HS, varies exponent n | f_R0 ∈ [1e-8, 1e-4], n ∈ [0, 4] | ≤3.5 (trust ≤1) | 0–49 | 1–5% | git | **TF 1.14 time-capsule** ⚠ | BSD-3 | C |
| ReACT (ACTio-ReACTio) | f(R)/nDGP/w0wa/IDE/K-mouflage/GCCG calculator | — | ~1–3 | ≤2.5 | 1–5% | compile (GSL, SUNDIALS) | C++ / pyreact | MIT | B (seconds/call) |
| Cosmic-Enu | nonlinear **neutrino** P(k) | Mν ≤ 0.93 eV, w0wa | — | — | ~3.5% | compile C + GSL | C/GSL | — | B |
| axionHMcode | mixed axion CDM halo model | m_a ~1e-24.5 eV, f_ax ∈ [0.01, 0.3] | — | 1–8 | — | git (+ axionCAMB Fortran) | Python+Numba | none ⚠ | B/C |
| Axion-Emulator | mixed-axion boost (COLA) | m_a, f_ax + cosmo | — | — | — | git | NN, immature | none ⚠ | C (watch) |

Notes:
- **CubicGalileonEmu specifics** (inspected in depth): 50 pickled SEPIA GP models (~13 MB) + ~78 MB training arrays ship in-repo; outputs boost with GP mean+std; pickles are Python-version-sensitive. The scipy ≤ 1.10.1 pin traces entirely to SEPIA's removed `sym_pos` argument — patched in this repo's environment (see ENVIRONMENT.md), so it coexists with modern scipy. Loading models lazily per snapshot is the right pattern.
- freyja (github.com/chzruan/freyja, MIT) adds f(R)/nDGP **halo statistics** (HMF, bias, correlation functions) at z=0.25.
- No public code yet: MGCAMB+ReACT reconstruction emulator (2607.29683), field-level MG emulators, fuzzy-DM Lyα flux emulator (2606.06969).

---

## 3. CMB Cls, LSS multipoles, 3x2pt, secondary anisotropies

### 3.1 CMB power spectra

| Emulator | Output | ℓ-range | Install | Deps | License | MCP |
|---|---|---|---|---|---|---|
| **CosmoPower zoo (Bolliet/Jense)** | TT/TE/EE/PP (ΛCDM, +Neff, +Σmν, wCDM, EDE), P(k) to k=50 Mpc⁻¹, BAO/background | 2–10⁴⁺ | clone model repos | TF, or **.npz via JAX/numpy** | per-repo | **A/B** |
| cosmopower-jax | differentiable loader for the above | per model | pip | JAX | GPL-3+NC ⚠ | **A** |
| CosmoPower (core) | TT/TE/EE/PP Planck-range shipped models | 2–2508 | pip | TF ⚠ (≥2.14 broke .pkl ⚠) | GPL-3+NC ⚠ | B |
| **Capse.jl / jaxcapse** | TT/TE/EE/PP, ~45 µs/eval | multi-1000s (query model) | Julia / `pip install jaxcapse` | Julia or JAX | MIT | **A** |
| ClassNet (class_public branch) | full CLASS outputs, ~150× perturbation module | any | build branch | C + torch | CLASS | B |
| CONNECT | train-your-own CLASS emulators | per model | git | TF + CLASS | — | C (framework) |
| OLÉ | online GP emulator inside samplers, 30–350× | n/a | pip/git | JAX | MIT | B (stateful) |
| CMBolic | symbolic CMB (lensing, 2026) | — | — | none | — | watch |
| PICO, CosmoNet | historical (2006–08) | — | — | — | — | C (defunct) |

### 3.2 Galaxy power spectrum multipoles (EFTofLSS / RSD)

| Emulator | Output | Coverage | Install | Deps | License | MCP |
|---|---|---|---|---|---|---|
| **PyBird-JAX** | one-loop P0/P2/P4 from *any* input linear P(k) — no parameter box | model-independent ⚠ needs linear P(k) provider | pip/git | JAX | — | **A** |
| **Effort.jl / jaxeffort** | EFTofLSS multipoles, µs eval, DESI-used | k ≲ 0.2–0.3 h/Mpc | Julia / pip | Julia or JAX | MIT | **A** |
| COMET | P0/P2/P4 + tree bispectrum, 9-param incl. Ω_K | z ≤ ~3 | `pip install comet-emu` | TF ⚠ | MIT | B |
| Matryoshka / EFTEMU | multipoles (BOSS-era), transfer function | k ≲ 0.25 | git | TF/Keras | MIT | C (frozen) |
| EmulateLSS / ShapeFit_Velocileptors | velocileptors surrogates; DESI ShapeFit route maintained | k ~0.001–0.5 | git | TF / numpy | — | B/C |
| Bora.jl / jaxbora | BAO correlation function | — | Julia / pip | MIT | watch |

### 3.3 3x2pt / angular statistics (mostly exact differentiable calculators, not emulators)

| Tool | Role | MCP note |
|---|---|---|
| jax-cosmo (`pip install jax-cosmo`, MIT) | Limber Cls in pure JAX | pair emulated P(k,z) + jax-cosmo projection = ready 3x2pt tool recipe |
| Blast.jl (MIT) | beyond-Limber 3x2pt algorithm, differentiable | Julia runtime |
| LimberJack.jl (MIT) | differentiable Cls / 3x2pt data vectors | Julia |
| KiDS CosmoPowerCosmosis | emulated P(k) → CosmoSIS exact projection, KiDS-1000 | design template; CosmoSIS too heavy to wrap |
| CosmoLike/Cocoa emulators_code | whole-data-vector NN emulators (LSST-Y1/DES) | fast but survey-rigid |
| dark_emulator (Dark Quest I) | GP: HMF, ξ_hh/ξ_hm, HOD → wp, ΔΣ | pip; ~100 ms GP evals; few-hundred-MB data |

Notes:
- **Unit trap #1 of the whole catalog:** CLASS-trained CosmoPower models emit *raw dimensionless Cl*; likelihoods want Dl in µK². Bake the (T_CMB·10⁶)²·ℓ(ℓ+1)/2π conversion into the tool layer and expose both.
- A **JAX-first stack** (cosmopower-jax + jaxcapse + jaxeffort + PyBird-JAX + jax-cosmo + classy_szfast) covers CMB, P(k), multipoles, and 3x2pt with one runtime and zero TensorFlow.
- Model files: Jense et al. 2024 packaging standard (self-describing YAML/npz: names, ranges, grids) is the format to mirror in MCP tool schemas.

### 3.4 Secondary anisotropies

| Tool | Output | Install | License | MCP |
|---|---|---|---|---|
| **class_sz / classy_szfast** | tSZ Cl^yy, kSZ, CMB lensing, cluster counts, CIB, cross-Cls; emulator-backed CMB/P(k) in ~1 ms | `pip install classy_sz` | MIT | **A/B** ⚠ first import downloads ~100s MB — pre-bake in deploy |
| candl (+ candl_data) | JAX-differentiable CMB likelihoods (SPT-3G, ACT) — *consumer* of Cl tools | pip | — | companion |

---

## 4. Baryons, hydro simulations, clusters & halos

### 4.1 Baryonic suppression of P(k)

| Emulator | Params | k [h/Mpc] | z | Accuracy | Install | Deps | License | MCP |
|---|---|---|---|---|---|---|---|---|
| **SP(k) / pyspk** | fb(M,z) relation (from theory or X-ray/SZ data) | 0.1–12 | ≤3 | ≤2% (incl. FLAMINGO) | `pip install pyspk` | numpy/scipy | LGPL-3 | **A** |
| **BCemu** | 3–8 BCM params + fb | 0.03–12.5 | ≤2 (2025: ≤3) | ~1% | `pip install BCemu` | ⚠ legacy pin `smt==1.0.0`; 2025 JAX backend | MIT | **A** |
| baccoemu baryons | 7 baryonification params | ≤5 (Burger25: 0.017–17.7) | ≤1.5 | 1–2% | (same pkg as §1.4) | JAX | MIT | **A** |
| syren-baryon | CAMELS feedback params (A_SN1/2, A_AGN1/2), per-suite fits | ~10s | ≲3 | suite-level | git | numpy | MIT | **A** |
| **subgrid_emu** (Ramachandra) | CRK-HACC subgrid params (κ_w, e_w, M_seed, v_kin, ε_kin) | — | z=0 + snaps | GP + quantiles | git + pip -e | sklearn GP ⚠ pickle-version | **none — add license** ⚠ | **A/B** |
| BCMemu | superseded by BCemu | — | — | — | — | — | MIT | C |
| van Daalen powerlib | 100+ hydro/DMO spectra library + fb fitting formula | ≤1 (formula) | — | 2% | data | — | — | validation data |

subgrid_emu outputs beyond P(k)-ratio: galaxy stellar mass function, cosmic SFR density,
cluster gas density/fraction profiles, BH–stellar mass relation — with 5%/95% GP quantiles.
Uncertainty output is a distinguishing feature worth surfacing in the MCP tool schema.

### 4.2 Halo mass function

| Emulator | Space | Mass range | z | Install | Deps | License | MCP |
|---|---|---|---|---|---|---|---|
| **MiraTitanHMFemulator** | 8-param νw0wa | M200c ≥ 1e13 Msun/h | 0–2 | `pip install MiraTitanHMFemulator` | numpy/scipy | MIT | **A** |
| e-MANTIS HMF | f(R) + wCDM | ~1e12–1e15 (FoF, M200c, M500c) | ≤~1.5 | pip (same pkg) | sklearn | GPL-3+ | **A** |
| aemulusnu_hmf | wνCDM | ≥1e13 | 0–2 | git | numpy/scipy + CLASS ⚠ | — | B |
| Aemulus hmf_emulator | wCDM | 1e13–1e15.8 (M200b) | 0–3 | git | george + CLASS ⚠, frozen | GPL-2 | B/C |
| dark_emulator (DQ1) | 6-param wCDM | ≥1e12 (M200b) | ≤1.48 | `pip install dark_emulator` | george, colossus | MIT | B |
| CCToolkit (Euclid HMF+bias) | calibrated model | 1e13–10^15.5 | broad | git + pip | CAMB | MIT | **A/B** |

### 4.3 Clusters / SZ / X-ray / bias / HOD

| Tool | Output | Install | License | MCP |
|---|---|---|---|---|
| **picasso** (Argonne) | cluster gas pressure/density profiles painted onto GO halos (tSZ/X-ray proxies) | git + pip | MIT | **A** |
| class_sz | tSZ/cluster counts engine (see §3.4) | pip | MIT | A/B |
| Aemulus bias_emulator | linear halo bias b(M,z,cosmo), ~1% | git | GPL-3 | B |
| hydro_mc (Magneticum) | c(M,z,cosmo) + hydro↔DMO mass conversion | git | none ⚠ | A/B |
| HODEmu (Magneticum) | satellite HOD from hydro sims | git | — | B |
| CAMELS_emulator (Lau) | X-ray CGM surface brightness profiles vs feedback | git | none ⚠ | B |
| sunbird | galaxy 2PCF / density-split / void CCF (AbacusSummit, w0waCDM+Neff) | git | MIT | B |
| AbacusHOD (abacusutils) | HOD mock generator — needs 10s–100s GB catalogs | pip | — | C for hosted |
| The300 DeepPlanck, Magneticum portal | data products / web services, not emulators | — | — | C |

---

## 5. IGM: Lyman-α & 21cm

Directly relevant to the eBOSS Lyα thread in the existing spectra server.

| Emulator | Output | Ranges | Install | Deps | License | MCP |
|---|---|---|---|---|---|---|
| **LaCE** (igmhub, DESI) | Lyα P1D(k∥) — GP + NN flavors | z ~2–4.5; k∥ ≲ 3–4 Mpc⁻¹; 6-param IGM/cosmo compression (Δ²_p, n_p, mF, sigT, γ, kF) | git + pip | CAMB, torch/GPy | check | **A/B** |
| **ForestFlow** (igmhub) | Lyα P3D(k,µ) via conditional normalizing flow, same parametrization | as LaCE | git | LaCE + flow stack | check | B |
| lya_emulator_full (PRIYA, Bird) | multi-fidelity GP P1D + T0, 9D (4 cosmo + 5 astro/thermal) | z 2.2–5.4; velocity units | git | GPyTorch, cobaya | MIT? check | B |
| **21cmEMU** | 21cmFAST outputs: global Tb, x_HI, Ts, τ_e, Δ²₂₁(k,z), UV LFs | 9–11 astro params; z 5–35 | `pip install py21cmemu` | TF/Keras ⚠ | MIT | **A** |
| globalemu | 21cm global signal | — | `pip install globalemu` | TF | — | A/B |
| 21cmVAE / 21cmLSTM | global signal variants | — | git | — | — | B/C |

---

## 6. Higher-dimensional: fields, maps, generative models

For a CPU-only host most neural field emulators are catalog-tier; the realistic candidates are marked.

| Tool | Output | Deps | Weights public? | License | MCP (CPU) |
|---|---|---|---|---|---|
| **GLASS** | lognormal lightcones: HEALPix matter shells, WL convergence/shear maps, galaxy catalogs | `pip install glass`, numpy stack | n/a (statistical) | MIT | **A** — best production candidate in this category |
| **ForSE / ForSE+** | Galactic dust foreground maps 80′→12′/3′ (HEALPix, µK) | Keras/TF ⚠ | yes (NERSC portal) | MIT | **A/B** |
| CAMELS CMD inference nets | (Ωm, σ8, astro) from 2D fields, small CNNs | PyTorch | yes | — | B |
| scattering_transform / s2scat | field synthesis from scattering statistics (flat / sphere) | torch / JAX | n/a | MIT | B |
| map2map + Jamieson emulators (+ jax_nbody_emulator) | field-level N-body surrogate (ZA → displacements/velocities), ~1% P(k) to k≈1 | PyTorch / JAX | yes | GPL-3 | C (GPU-class; ≤128³ feasible) |
| pmwd / JaxPM | differentiable PM sims (64³–128³ CPU-feasible) | JAX | n/a | — | B (as "fast approx sim" tool) |
| SRS-map2map (super-resolution) | 512× particle SR | PyTorch, bigfile | yes (in repo) | — | C (GPU) |
| LDL | paint hydro observables on N-body | nbodykit/vmad ⚠ aging | yes | MIT? | B/C |
| Diffusion (nmudur), HIGlow, jax-lensing, ICdiffusion | 2D/3D generative + inference | torch/JAX | partial | — | C (GPU-flagged) |
| cosmoGAN, 21cmGAN, 3DcosmoGAN | TF1-era GANs | TF1 ⚠ | yes | — | C (env unbuildable; ONNX-convert if wanted) |
| NECOLA, HIDM, nuGAN, HInet | paper-only | — | **no code** | — | excluded |

---

## 7. Frameworks & infrastructure (build/serve layer)

| Tool | Role | Install | License |
|---|---|---|---|
| SEPIA (LANL) | Bayesian GP emulation + calibration (CubicGalileonEmu's backend) | git | BSD-3 |
| swiftemulator | GP emulation of sim scaling relations | `pip install swiftemulator` | GPL-3 |
| OLÉ | online-learning emulator layer for samplers | git/pip | MIT |
| Aemulator | base-class API for Aemulus emulators | git | — |
| ostrich | generic PCA+GP surrogate | git | — |
| pyccl (LSST DESC) | wraps CosmicEmu, baccoemu, HMF fits behind one API | `pip install pyccl` | BSD |
| Cobaya / CosmoSIS shims | sampler interfaces (cosmopower_cobaya, CosmoPowerCosmosis, SOLikeT) | per repo | — |
| candl | differentiable CMB likelihoods | pip | — |

---

## 8. Cross-cutting design notes for the MCP server

**Composition is the core abstraction.** Most emulators are *ratios*: nonlinear boost × linear;
MG boost × ΛCDM nonlinear; baryon suppression × gravity-only. The server should expose
raw per-emulator tools **and** composed tools (e.g. `get_nonlinear_pk(model=...)`) that
document which baseline was used. This is also the natural skill structure: a skill that knows
which boosts compose with which baselines, over which overlapping (k, z, parameter) ranges.

**Unit traps (encode in the tool layer, return units in every result):**
1. CosmicEmu: k [1/Mpc], P [Mpc³] — no h anywhere. Everything else: h/Mpc, (Mpc/h)³.
2. CosmoPower CMB: raw dimensionless Cl → convert to Dl [µK²] with (T_CMB·10⁶)²·ℓ(ℓ+1)/2π.
3. BACCO: σ8_cold / Ω_cold (CDM+baryons), not total matter.
4. Lyα: comoving Mpc vs velocity (km/s) conventions differ between LaCE and PRIYA.

**Dependency clusters** (a small-RAM host argues for lazy-load + LRU-unload of models;
in practice all clusters below coexist in one environment — see ENVIRONMENT.md):
- *numpy/scipy-only*: symbolic_pofk, csstemu, pyspk, MiraTitanHMFemulator, hydro_mc, CCToolkit(+camb) — near-zero conflict risk, can share.
- *sklearn GP*: e-MANTIS, nDGPemu, subgrid_emu, FORGE — sklearn/joblib pickle version pins.
- *JAX*: baccoemu, cosmopower-jax, jaxcapse/jaxeffort, PyBird-JAX, jax-cosmo, picasso, classy_szfast, OLÉ — the recommended primary runtime.
- *PyTorch*: GokuNEmu, FREmu, sunbird, LaCE-NN, Sesame — CPU wheels fine, ~1 GB install.
- *TensorFlow*: cosmopower core, COMET, 21cmEMU, BCemu-legacy, reactemu-fr, DS-emulators — heaviest, most version-fragile; prefer JAX/plain-npz routes where they exist.
- *Compiled C/C++*: CLASS, CosmicEmu, EuclidEmulator2, ReACT, Cosmic-Enu — build in image, wrap via Python/subprocess.
- *Quarantine*: CubicGalileonEmu (scipy ≤ 1.10.1 + git SEPIA) — own venv; mgemu (TF1) — skip or ONNX.

**Cold-start & data logistics:** pre-bake all weight downloads at deploy time (baccoemu cache,
classy_szfast ~100s MB, 21cmEMU HF weights, CosmoPower model repos); record model versions/SHAs
in tool metadata; TF/JAX first-call JIT costs seconds — warm up at service start, not per request.

**Range validation:** every emulator silently extrapolates (NN) or errors (GP) outside its
Latin hypercube. Enforce parameter boxes at the tool boundary, surface the valid ranges in
tool descriptions, and prefer reading ranges from shipped model metadata over hard-coding.

**Licensing:** MIT/BSD/Apache dominate (safe); GPL-family fine for a hosted service;
the **CosmoPower non-commercial rider** and the **no-license repos** (Sesame, axionHMcode,
Axion-Emulator, EDE zoo, hydro_mc, subgrid_emu) need decisions before
redistributing weights.

---

## 9. Selection (implemented in this server)

The production selection — broad science coverage, all CPU-fast, minimal fragility:

**Wave 1 — core P(k) stack:** symbolic_pofk · baccoemu · EuclidEmulator2 · CSST emulator · GokuNEmu · CAMB/HMcode baseline
**Wave 2 — beyond-ΛCDM (the differentiator):** CubicGalileonEmu · e-MANTIS (P(k)+HMF) · nDGPemu · FREmu · Cosmic-Enu
**Wave 3 — baryons & clusters:** pyspk · BCemu · subgrid_emu · MiraTitanHMFemulator · picasso · CCToolkit
**Wave 4 — CMB & LSS observables:** cosmopower-jax + Bolliet/Jense model zoo · jaxcapse · PyBird-JAX (+ jaxeffort) · classy_szfast · jax-cosmo 3x2pt recipe
**Wave 5 — IGM (ties to eBOSS demo):** LaCE · ForestFlow · 21cmEMU
**Wave 6 — showpiece higher-dimensional:** GLASS lightcones · CAMELS CMD inference nets · (ForSE if TF2-compat verified)

Items to *exclude* for now with reasons documented above: mgemu (TF1), AbacusHOD (data size),
BCMemu (superseded), PICO/CosmoNet (defunct), all GPU-class field emulators, paper-only codes.
