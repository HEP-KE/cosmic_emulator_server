"""Modified-gravity tools: boosts B(k) = P_MG / P_LCDM and composed P(k).

Models: Hu-Sawicki f(R) (e-MANTIS), nDGP (nDGPemu), cubic Galileon
(CubicGalileonEmu, SEPIA GP with uncertainty). Boosts compose with any LCDM
nonlinear backend from tools.pk.
"""

import numpy as np
from typing import Annotated, Literal

from pydantic import Field, validate_call

from ..common import (ArtifactResult, downsample_columns, get_cached, k_grid,
                      param_slug, quiet, resolve_outdir, summary_stats,
                      write_csv)
from ..pk import backends as pk_backends

# Per-model training boxes (cosmology side; the MG parameter itself is
# already exactly bounded in the tool schema). The schema's shared bounds
# are the union; these are the per-model truths.
MG_BOXES = {
    "fofr": {"Om": [0.2365, 0.3941], "sigma8": [0.6, 1.0], "z": [0.0, 2.0]},
    "ndgp": {"Om": [0.28, 0.36], "Ob": [0.04, 0.06], "ns": [0.92, 1.0],
             "As": [1.7e-9, 2.5e-9], "h": [0.61, 0.73], "z": [0.0, 2.0]},
    "cubic_galileon": {"Om": [0.275, 0.331], "ns": [0.85, 1.1],
                       "As": [1.453e-9, 3.291e-9], "h": [0.61, 0.73],
                       "z": [0.0, 49.0]},
}


def _check_mg_box(model: str, values: dict) -> tuple[bool, list[str]]:
    warnings = []
    for name, (lo, hi) in MG_BOXES[model].items():
        v = values.get(name)
        if v is not None and not (lo <= v <= hi):
            warnings.append(f"{model}: {name}={v:g} outside training box "
                            f"[{lo:g}, {hi:g}] — output is an extrapolation")
    return not warnings, warnings

__all__ = ["compute_mg_boost", "compute_mg_pk"]

MGModel = Literal["fofr", "ndgp", "cubic_galileon"]

# CubicGalileonEmu snapshot redshifts are stored in its z_k data file; loaded lazily.


def _fofr_boost(k, z, Om, sigma8, minus_log10_fR0, **_):
    from emantis import FofrBoost
    emu = get_cached("emantis_fofr", FofrBoost)
    with quiet():
        b = emu.predict_boost(Om, sigma8, minus_log10_fR0, 1.0 / (1.0 + z), k=k)
    return np.ravel(np.asarray(b)), {}


def _ndgp_boost(k, z, Om, Ob, h, ns, As, H0rc, **_):
    from nDGPemu import BoostPredictor
    emu = get_cached("ndgpemu", BoostPredictor)
    cosmo = {"Om": Om, "ns": ns, "As": As, "h": h, "Ob": Ob}
    with quiet():
        b = emu.predict(H0rc=H0rc, z=z, cosmo_params=cosmo)
    return np.interp(k, np.ravel(emu.k_vals), np.ravel(b)), {}


def _load_cubic_galileon(z_index: int):
    import os
    import CubicGalileonEmu
    from CubicGalileonEmu import load as cgl
    from CubicGalileonEmu.emu import load_model_multiple

    def factory():
        B, B_sm, k_emu, z_all = cgl.load_boost_data()
        params = cgl.load_params()
        model_dir = os.path.join(os.path.dirname(CubicGalileonEmu.__file__), "model/")
        models, datas = load_model_multiple(
            model_dir=model_dir, p_train_all=params, y_vals_all=B_sm,
            y_ind_all=k_emu, z_index_range=[z_index])
        return models[0], datas[0], k_emu, z_all

    return get_cached(f"cubic_galileon:{z_index}", factory)


def _cubic_galileon_boost(k, z, Om, ns, As, h, f_phi, **_):
    from CubicGalileonEmu.emu import emulate
    # find snapshot nearest to requested z
    z_all = np.ravel(_load_cubic_galileon(0)[3])
    z_index = int(np.argmin(np.abs(z_all - z)))
    model, data, k_emu, _ = _load_cubic_galileon(z_index)
    with quiet():
        mean, std = emulate(sepia_model=model, sepia_data=data,
                            input_params=np.array([[Om, ns, As * 1e9, h, f_phi]]))
    mean, std = np.ravel(np.asarray(mean)), np.ravel(np.asarray(std))
    boost = np.interp(k, k_emu, mean)
    err = np.interp(k, k_emu, std)
    return boost, {"gp_std": err, "snapshot_z": float(z_all[z_index])}


@validate_call
def compute_mg_boost(
    output_dir: Annotated[str, Field(min_length=1)],
    model: MGModel = "fofr",
    Om: Annotated[float, Field(ge=0.2365, le=0.3941, description="Omega_m (intersection of the three models' boxes: [0.2365, 0.3941])")] = 0.31,
    Ob: Annotated[float, Field(ge=0.04, le=0.06)] = 0.049,
    h: Annotated[float, Field(ge=0.61, le=0.73)] = 0.67,
    ns: Annotated[float, Field(ge=0.92, le=1.0)] = 0.965,
    As: Annotated[float, Field(ge=1.7e-9, le=2.5e-9)] = 2.1e-9,
    sigma8: Annotated[float, Field(ge=0.6, le=1.0, description="sigma8 of the LCDM counterpart (used by fofr)")] = 0.82,
    minus_log10_fR0: Annotated[float, Field(ge=4.0, le=7.0, description="-log10|f_R0|, e.g. 5.0 for |f_R0|=1e-5 (fofr only)")] = 5.0,
    H0rc: Annotated[float, Field(ge=0.2, le=20.0, description="nDGP crossover scale H0*rc (ndgp only)")] = 1.0,
    f_phi: Annotated[float, Field(ge=0.02, le=1.0, description="Galileon dark-energy fraction (cubic_galileon only)")] = 0.5,
    k_min: Annotated[float, Field(ge=0.02)] = 0.03,
    k_max: Annotated[float, Field(le=12.0)] = 5.0,
    n_points: Annotated[int, Field(ge=10, le=1000)] = 200,
    z: Annotated[float, Field(ge=0.0, le=2.0)] = 0.0,
    return_data: Annotated[bool, Field(description="Include downsampled arrays in metadata.data.")] = False,
) -> ArtifactResult:
    """Compute a modified-gravity power spectrum boost B(k) = P_MG / P_LCDM.

    Models: "fofr" (Hu-Sawicki f(R), e-MANTIS, k 0.03-7 h/Mpc), "ndgp"
    (nDGPemu, k <= 5), "cubic_galileon" (SEPIA GP, k 0.02-12, returns a
    gp_std uncertainty column; the requested z snaps to the nearest of 51
    training snapshots, recorded in metadata). Only the parameters relevant
    to the chosen model are used; the rest are ignored. B(k) is
    dimensionless; multiply into a LCDM nonlinear P(k) or call compute_mg_pk
    to get the composed spectrum directly.
    """
    k = k_grid(k_min, k_max, n_points)
    kwargs = dict(Om=Om, Ob=Ob, h=h, ns=ns, As=As, sigma8=sigma8,
                  minus_log10_fR0=minus_log10_fR0, H0rc=H0rc, f_phi=f_phi)
    boost_fn = {"fofr": _fofr_boost, "ndgp": _ndgp_boost,
                "cubic_galileon": _cubic_galileon_boost}[model]
    boost, extra = boost_fn(k, z, **kwargs)

    mg_par = {"fofr": f"-log10|fR0|={minus_log10_fR0}",
              "ndgp": f"H0rc={H0rc}",
              "cubic_galileon": f"f_phi={f_phi}"}[model]
    in_box, box_warnings = _check_mg_box(model, dict(kwargs, z=z))
    label = f"{model} boost ({mg_par}) z={z:g}"
    columns = {"k_h_per_Mpc": k, "boost": boost}
    if "gp_std" in extra:
        columns["gp_std"] = extra["gp_std"]
    outdir = resolve_outdir(output_dir)
    slug = param_slug(dict(kwargs, model=model, z=z))
    path = outdir / f"mg_boost_{model}_z{z:g}_{slug}.csv"
    write_csv(path, columns,
              [f"label: {label}", "quantity: boost",
               "units: k [h/Mpc], boost [dimensionless]",
               f"model: {model}", f"z: {z:g}"])

    meta = {"model": model, "mg_parameter": mg_par, "z": z,
            "in_training_box": in_box, "extrapolation_warnings": box_warnings,
            "units": {"k": "h/Mpc", "boost": "dimensionless"},
            "stats": summary_stats(k, boost, "k", "boost")}
    if "snapshot_z" in extra:
        meta["snapshot_z_used"] = extra["snapshot_z"]
    if return_data:
        meta["data"] = downsample_columns(columns)
    message = (f"Computed {label}: max boost {np.max(boost):.3f} at "
               f"k={k[np.argmax(boost)]:.2f} h/Mpc.")
    if box_warnings:
        message += " WARNING: " + "; ".join(box_warnings)
    return ArtifactResult(status="success", files=[str(path)],
                          message=message, metadata=meta)


@validate_call
def compute_mg_pk(
    output_dir: Annotated[str, Field(min_length=1)],
    model: MGModel = "fofr",
    baseline: Literal["baccoemu", "camb_hmcode", "euclidemu2", "csst"] = "baccoemu",
    Om: Annotated[float, Field(ge=0.2365, le=0.3941)] = 0.31,
    Ob: Annotated[float, Field(ge=0.04, le=0.06)] = 0.049,
    h: Annotated[float, Field(ge=0.61, le=0.73)] = 0.67,
    ns: Annotated[float, Field(ge=0.92, le=1.0)] = 0.965,
    As: Annotated[float, Field(ge=1.7e-9, le=2.5e-9)] = 2.1e-9,
    sigma8: Annotated[float, Field(ge=0.6, le=1.0)] = 0.82,
    minus_log10_fR0: Annotated[float, Field(ge=4.0, le=7.0)] = 5.0,
    H0rc: Annotated[float, Field(ge=0.2, le=20.0)] = 1.0,
    f_phi: Annotated[float, Field(ge=0.02, le=1.0)] = 0.5,
    k_min: Annotated[float, Field(ge=0.02)] = 0.03,
    k_max: Annotated[float, Field(le=5.0)] = 3.0,
    n_points: Annotated[int, Field(ge=10, le=1000)] = 200,
    z: Annotated[float, Field(ge=0.0, le=1.5)] = 0.0,
    return_data: Annotated[bool, Field(description="Include downsampled arrays in metadata.data.")] = False,
) -> ArtifactResult:
    """Compute the modified-gravity NONLINEAR P(k): MG boost x LCDM baseline.

    Composes compute_mg_boost with a LCDM nonlinear backend from tools.pk
    (metadata records which baseline was used — quote it when reporting
    results). Output CSV: k [h/Mpc], P_MG [(Mpc/h)^3], boost. For plots, pass
    the file to plot_pk_comparison together with a LCDM spectrum at the same
    cosmology.
    """
    k = k_grid(k_min, k_max, n_points)
    kwargs = dict(Om=Om, Ob=Ob, h=h, ns=ns, As=As, sigma8=sigma8,
                  minus_log10_fR0=minus_log10_fR0, H0rc=H0rc, f_phi=f_phi)
    boost_fn = {"fofr": _fofr_boost, "ndgp": _ndgp_boost,
                "cubic_galileon": _cubic_galileon_boost}[model]
    boost, extra = boost_fn(k, z, **kwargs)

    params = {"Om": Om, "Ob": Ob, "h": h, "ns": ns, "As": As,
              "sigma8": sigma8, "mnu": 0.0, "w0": -1.0, "wa": 0.0}
    pk_lcdm = pk_backends.NONLINEAR_BACKENDS[baseline](params, k, z)
    pk_mg = pk_lcdm * boost

    mg_par = {"fofr": f"-log10|fR0|={minus_log10_fR0}",
              "ndgp": f"H0rc={H0rc}",
              "cubic_galileon": f"f_phi={f_phi}"}[model]
    in_box, box_warnings = _check_mg_box(model, dict(kwargs, z=z))
    label = f"{model} ({mg_par}) x {baseline} z={z:g}"
    outdir = resolve_outdir(output_dir)
    slug = param_slug(dict(kwargs, model=model, baseline=baseline, z=z))
    path = outdir / f"pk_mg_{model}_{baseline}_z{z:g}_{slug}.csv"
    columns = {"k_h_per_Mpc": k, "Pk_Mpc_over_h_cubed": pk_mg, "boost": boost}
    write_csv(path, columns,
              [f"label: {label}", "quantity: power_spectrum",
               "variant: nonlinear_mg",
               "units: k [h/Mpc], Pk [(Mpc/h)^3]",
               f"model: {model}", f"baseline: {baseline}", f"z: {z:g}"])
    meta = {"model": model, "mg_parameter": mg_par, "baseline": baseline,
            "z": z, "in_training_box": in_box,
            "extrapolation_warnings": box_warnings,
            "units": {"k": "h/Mpc", "Pk": "(Mpc/h)^3"},
            "stats": summary_stats(k, pk_mg, "k", "Pk")}
    if "snapshot_z" in extra:
        meta["snapshot_z_used"] = extra["snapshot_z"]
    if return_data:
        meta["data"] = downsample_columns(columns)
    message = f"Computed P_MG(k) = {label}."
    if box_warnings:
        message += " WARNING: " + "; ".join(box_warnings)
    return ArtifactResult(status="success", files=[str(path)],
                          message=message, metadata=meta)
