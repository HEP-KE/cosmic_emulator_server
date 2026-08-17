"""Halo and cluster tools: mass function and cluster gas modeling."""

import numpy as np
from typing import Annotated

from pydantic import Field, validate_call

from ..common import (ArtifactResult, get_cached, quiet, resolve_outdir,
                      write_csv)

__all__ = ["compute_hmf", "predict_cluster_gas_params"]


@validate_call
def compute_hmf(
    output_dir: Annotated[str, Field(min_length=1)],
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
) -> ArtifactResult:
    """Compute the halo mass function dn/dlnM with the Mira-Titan GP emulator.

    Masses are M200c in Msun/h; output dn/dlnM is comoving [(Mpc/h)^-3].
    Accuracy <2% for 1e13-1e14 Msun/h at z<1, degrading to ~10% at 1e15.
    The parameter box is the Mira-Titan design (note Ommh2/Ombh2 are
    PHYSICAL densities). Output CSV: M200c, dn/dlnM.
    """
    import MiraTitanHMFemulator

    emu = get_cached("miratitan_hmf", MiraTitanHMFemulator.Emulator)
    cosmo = {"Ommh2": Ommh2, "Ombh2": Ombh2, "Omnuh2": Omnuh2,
             "n_s": n_s, "h": h, "sigma_8": sigma_8, "w_0": w_0, "w_a": w_a}
    masses = np.logspace(log10_M_min, log10_M_max, n_masses)
    with quiet():
        hmf_mean, hmf_err = emu.predict(cosmo, z, masses)
    hmf = np.ravel(np.asarray(hmf_mean))
    err = np.ravel(np.asarray(hmf_err))

    label = f"Mira-Titan HMF z={z:g}"
    outdir = resolve_outdir(output_dir)
    path = outdir / f"hmf_miratitan_z{z:g}.csv"
    write_csv(path, {"M200c_Msun_per_h": masses, "dn_dlnM_h3_Mpc3": hmf,
                     "emulator_std": err},
              [f"label: {label}", f"z: {z:g}", f"cosmo: {cosmo}"])
    return ArtifactResult(
        status="success", files=[str(path)],
        message=f"Computed dn/dlnM for {n_masses} masses "
                f"1e{log10_M_min:g}..1e{log10_M_max:g} Msun/h at z={z:g}.",
        metadata={"cosmology": cosmo, "z": z,
                  "units": {"M": "M200c, Msun/h", "dn/dlnM": "(Mpc/h)^-3"}},
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
