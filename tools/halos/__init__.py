"""Halo and cluster tools: mass function and cluster gas modeling."""

import numpy as np
from typing import Annotated, Literal

from pydantic import Field, validate_call

from ..common import (ArtifactResult, downsample_columns, get_cached,
                      param_slug, quiet, resolve_outdir, summary_stats,
                      varied_label, write_csv)

_HMF_DEFAULTS = {"Ommh2": 0.147, "Ombh2": 0.022, "Omnuh2": 0.0006,
                 "sigma_8": 0.8, "h": 0.67, "n_s": 0.965,
                 "w_0": -1.0, "w_a": 0.0}

__all__ = ["compute_hmf", "predict_cluster_gas_params"]

# colossus model names for the theory backends; press74/sheth99 are
# friends-of-friends multiplicity functions (colossus requires mdef='fof'),
# tinker08 is calibrated for spherical-overdensity definitions.
_THEORY_MODELS = {"tinker08": "tinker08", "sheth_tormen": "sheth99",
                  "press_schechter": "press74"}


def _theory_hmf(backend, mass_def, cosmo, masses, z):
    from colossus.cosmology import cosmology as ccosmo
    from colossus.lss import mass_function

    h = cosmo["h"]
    params = {"flat": True, "H0": h * 100,
              "Om0": cosmo["Ommh2"] / h**2, "Ob0": cosmo["Ombh2"] / h**2,
              "sigma8": cosmo["sigma_8"], "ns": cosmo["n_s"]}
    if cosmo["w_0"] != -1.0 or cosmo["w_a"] != 0.0:
        params.update(de_model="w0wa", w0=cosmo["w_0"], wa=cosmo["w_a"])
    name = f"hmf_{param_slug(params)}"
    # persistence='' stops colossus writing interpolation tables to
    # $HOME/.colossus — read-only under the production systemd sandbox
    # (same failure class as the PyBird cache); in-memory caching still works
    ccosmo.setCosmology(name, params, persistence="")

    model = _THEORY_MODELS[backend]
    if model in ("press74", "sheth99"):
        if mass_def != "fof":
            raise ValueError(
                f"{backend} is a friends-of-friends multiplicity function; "
                "call it with mass_def='fof'. For an apples-to-apples "
                "comparison against the Mira-Titan emulator (M200c), use "
                "backend='tinker08' with mass_def='200c' — evaluating an "
                "FoF-calibrated fit at an SO mass is the classic way to get "
                "a spurious factor-of-a-few discrepancy.")
        mdef = "fof"
    else:
        mdef = mass_def
        if mdef == "fof":
            raise ValueError("tinker08 is SO-calibrated; use mass_def "
                             "'200c', '200m', or '500c'.")
    return mass_function.massFunction(masses, z, mdef=mdef, model=model,
                                      q_out="dndlnM")


@validate_call
def compute_hmf(
    output_dir: Annotated[str, Field(min_length=1)],
    backend: Annotated[Literal["miratitan", "tinker08", "sheth_tormen", "press_schechter"], Field(description="'miratitan' = simulation-calibrated GP emulator (M200c only). The rest are linear-theory/analytic fits via colossus: 'tinker08' (SO-calibrated, comparable to miratitan at mass_def='200c'), 'sheth_tormen' and 'press_schechter' (FoF multiplicity functions, mass_def='fof').")] = "miratitan",
    mass_def: Annotated[Literal["200c", "200m", "500c", "fof"], Field(description="Mass definition. miratitan supports only '200c'; tinker08 supports the SO definitions; press_schechter/sheth_tormen require 'fof'.")] = "200c",
    Ommh2: Annotated[float, Field(ge=0.12, le=0.155, description="Physical total matter density Omega_m h^2")] = 0.147,
    Ombh2: Annotated[float, Field(ge=0.0215, le=0.0235)] = 0.022,
    Omnuh2: Annotated[float, Field(ge=0.0, le=0.01, description="Physical neutrino density (0.0006 ~ 0.06 eV)")] = 0.0006,
    sigma_8: Annotated[float, Field(ge=0.7, le=0.9)] = 0.8,
    h: Annotated[float, Field(ge=0.55, le=0.85)] = 0.67,
    n_s: Annotated[float, Field(ge=0.85, le=1.05)] = 0.965,
    w_0: Annotated[float, Field(ge=-1.3, le=-0.7)] = -1.0,
    w_a: Annotated[float, Field(ge=-1.0, le=1.0, description="Constrained jointly with w_0: (-w0-wa)^(1/4) must lie in [0.3, 1.29]")] = 0.0,
    log10_M_min: Annotated[float, Field(ge=13.0, le=15.5)] = 13.0,
    log10_M_max: Annotated[float, Field(ge=13.5, le=16.0)] = 15.5,
    n_masses: Annotated[int, Field(ge=5, le=200)] = 50,
    z: Annotated[float, Field(ge=0.0, le=2.0)] = 0.0,
    random_seed: Annotated[int, Field(ge=0, description="Seed for the emulator's Monte-Carlo error draws — fixed by default so identical calls are bitwise reproducible.")] = 0,
    return_data: Annotated[bool, Field(description="Include downsampled arrays in metadata.data.")] = False,
) -> ArtifactResult:
    """Compute the halo mass function dn/dlnM — emulator or linear-theory fits.

    Backends: "miratitan" (simulation-calibrated GP emulator, M200c,
    <2% for 1e13-1e14 Msun/h at z<1, ~10% at 1e15, includes emulator_std)
    and three theory baselines via colossus: "tinker08" (SO-calibrated fit
    — the right one to overlay against miratitan, at mass_def='200c'),
    "sheth_tormen" and "press_schechter" (FoF multiplicity functions,
    mass_def='fof'; comparing those against SO masses produces the classic
    factor-of-a-few artifact, so the tool refuses mismatched definitions).
    All backends share the same cosmology arguments (Ommh2/Ombh2 are
    PHYSICAL densities) and the same output convention: M [Msun/h],
    dn/dlnM comoving [(Mpc/h)^-3] — files overlay directly in
    plot_pk_comparison. Emulator-vs-theory comparison is a good validation
    workflow: expect ~5-10% agreement between miratitan and tinker08 at
    200c, and larger, mass-dependent deviations for the older fits.
    """
    cosmo = {"Ommh2": Ommh2, "Ombh2": Ombh2, "Omnuh2": Omnuh2,
             "n_s": n_s, "h": h, "sigma_8": sigma_8, "w_0": w_0, "w_a": w_a}
    masses = np.logspace(log10_M_min, log10_M_max, n_masses)

    if backend == "miratitan":
        import MiraTitanHMFemulator
        if mass_def != "200c":
            raise ValueError("The Mira-Titan emulator provides M200c only; "
                             "use backend='tinker08' for other SO "
                             "definitions ('200m', '500c').")
        emu = get_cached("miratitan_hmf", MiraTitanHMFemulator.Emulator)
        with quiet():
            # the emulator's error estimate uses np.random draws internally;
            # seed for bitwise-reproducible outputs (provenance/caching)
            np.random.seed(random_seed)
            hmf_mean, hmf_err = emu.predict(cosmo, z, masses)
        hmf = np.ravel(np.asarray(hmf_mean))
        err = np.ravel(np.asarray(hmf_err))
        base_label = f"Mira-Titan HMF z={z:g}"
        err_note = "emulator_std is the GP 1-sigma uncertainty"
    else:
        with quiet():
            hmf = np.ravel(np.asarray(_theory_hmf(backend, mass_def, cosmo,
                                                  masses, z)))
        err = np.zeros_like(hmf)
        base_label = f"{backend} ({mass_def}) HMF z={z:g}"
        err_note = ("analytic fit - emulator_std column is 0; typical "
                    "calibration accuracy ~5-10% (tinker08) or worse "
                    "(older fits)")

    label = varied_label(base_label, cosmo, _HMF_DEFAULTS)
    outdir = resolve_outdir(output_dir)
    slug = param_slug(dict(cosmo, z=z, backend=backend, mdef=mass_def))
    path = outdir / f"hmf_{backend}_z{z:g}_{slug}.csv"
    mass_col = f"M{mass_def}_Msun_per_h"
    columns = {mass_col: masses, "dn_dlnM_h3_Mpc3": hmf,
               "emulator_std": err}
    write_csv(path, columns,
              [f"label: {label}", "quantity: hmf",
               f"units: M ({mass_def}) [Msun/h], dn/dlnM [(Mpc/h)^-3]",
               f"backend: {backend}", f"mass_def: {mass_def}",
               f"z: {z:g}", f"cosmo: {cosmo}",
               f"random_seed: {random_seed}"])
    metadata = {"backend": backend, "mass_def": mass_def, "cosmology": cosmo,
                "z": z, "random_seed": random_seed,
                "units": {"M": f"{mass_def}, Msun/h",
                          "dn/dlnM": "(Mpc/h)^-3"},
                "uncertainty_note": err_note,
                "stats": summary_stats(masses, hmf, "M", "dn_dlnM")}
    if return_data:
        metadata["data"] = downsample_columns(columns)
    return ArtifactResult(
        status="success", files=[str(path)],
        message=f"Computed dn/dlnM ({backend}, {mass_def}) for {n_masses} "
                f"masses 1e{log10_M_min:g}..1e{log10_M_max:g} Msun/h at "
                f"z={z:g}.",
        metadata=metadata,
    )


@validate_call
def predict_cluster_gas_params(
    output_dir: Annotated[str, Field(min_length=1)],
    log10_M200c: Annotated[float, Field(ge=13.0, le=15.5, description="Halo mass log10(M200c / Msun)")] = 14.5,
    c200: Annotated[float, Field(ge=2.0, le=12.0, description="NFW concentration")] = 5.0,
    cacc_over_c200: Annotated[float, Field(ge=0.3, le=3.0, description="Accumulated-mass concentration ratio (dynamical state proxy)")] = 1.0,
    cpeak_over_c200: Annotated[float, Field(ge=0.3, le=3.0)] = 1.0,
    log10_dx_R200c: Annotated[float, Field(ge=-4.0, le=0.0, description="log10 center-of-mass offset / R200c (relaxedness proxy)")] = -2.0,
    ellipticity: Annotated[float, Field(ge=0.0, le=0.7)] = 0.1,
    prolateness: Annotated[float, Field(ge=-0.5, le=0.5)] = 0.05,
    a25: Annotated[float, Field(ge=0.05, le=1.0, description="Scale factor when 25% of final mass was assembled")] = 0.3,
    a50: Annotated[float, Field(ge=0.1, le=1.0)] = 0.5,
    a75: Annotated[float, Field(ge=0.15, le=1.0)] = 0.7,
    almm: Annotated[float, Field(ge=0.05, le=1.0, description="Scale factor of last major merger")] = 0.8,
    mdot: Annotated[float, Field(ge=0.0, le=10.0, description="Recent mass accretion rate (dimensionless)")] = 1.0,
) -> ArtifactResult:
    """Predict intracluster gas polytropic-model parameters with picasso.

    The picasso NN (Keruzore et al. 2024, trained on 576 zoom simulations)
    maps gravity-only halo properties to the parameters of a polytropic gas
    model (P0, rho0, polytropic index, non-thermal fraction shape) from
    which pressure/density profiles — and thence tSZ and X-ray proxies —
    follow. Feed halo properties from any N-body simulation or from
    plausible defaults for a population study.
    """
    import jax.numpy as jnp
    from picasso import predictors

    pred = get_cached("picasso_baseline", lambda: predictors.baseline_576)
    x = jnp.array([log10_M200c, c200, cacc_over_c200, cpeak_over_c200,
                   log10_dx_R200c, ellipticity, prolateness,
                   a25, a50, a75, almm, mdot])
    with quiet():
        theta = np.ravel(np.asarray(pred.predict_model_parameters(x)))

    names = ["rho_0", "P_0", "Gamma_0", "c_Gamma", "theta_0", "A_nt", "B_nt", "C_nt"]
    names = names[:len(theta)] + [f"theta_{i}" for i in range(len(names), len(theta))]
    outdir = resolve_outdir(output_dir)
    path = outdir / f"cluster_gas_params_M{log10_M200c:g}.csv"
    write_csv(path, {"param_index": np.arange(len(theta), dtype=float),
                     "value": theta},
              [f"label: picasso gas model log10M={log10_M200c:g}",
               f"param_names: {names}",
               f"halo: c200={c200}, log10_dx={log10_dx_R200c}"])
    return ArtifactResult(
        status="success", files=[str(path)],
        message=f"Predicted {len(theta)} polytropic gas-model parameters for "
                f"a log10(M200c)={log10_M200c:g} halo.",
        metadata={"gas_model_params": dict(zip(names, np.round(theta, 6).tolist())),
                  "model": "picasso baseline_576",
                  "note": "see picasso docs for the polytropic profile formulae"},
    )
