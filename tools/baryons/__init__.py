"""Baryonic-feedback tools: P(k) suppression and hydro subgrid emulation."""

import numpy as np
from typing import Annotated, Literal

from pydantic import Field, validate_call

import re
from pathlib import Path

from ..common import (ArtifactResult, downsample_columns, get_cached, k_grid,
                      param_slug, quiet, read_csv, resolve_outdir,
                      summary_stats, write_csv)

__all__ = ["compute_baryon_suppression", "baryonify_pk",
           "emulate_subgrid_statistic"]

SuppressionModel = Literal["spk", "bacco", "syren_IllustrisTNG",
                           "syren_Astrid", "syren_SIMBA", "syren_Swift_EAGLE"]

SUBGRID_STATS = {
    "Pk": "baryonic P(k) ratio hydro/gravity-only",
    "GSMF": "galaxy stellar mass function",
    "CGD": "cluster gas density profile",
    "fGas": "cluster gas fraction",
    "BHMSM": "black-hole mass vs stellar mass",
    "CSFR": "cosmic star formation rate density",
}


@validate_call
def compute_baryon_suppression(
    output_dir: Annotated[str, Field(min_length=1)],
    model: SuppressionModel = "spk",
    # --- SP(k) inputs (power-law fb - Mhalo relation) ---
    fb_a: Annotated[float, Field(ge=0.1, le=1.0, description="SP(k): fb normalization at the pivot mass, in units of Omega_b/Omega_m")] = 0.4,
    fb_pow: Annotated[float, Field(ge=-0.5, le=1.0, description="SP(k): fb power-law slope")] = 0.3,
    fb_pivot_log10Msun: Annotated[float, Field(ge=12.0, le=15.0)] = 13.5,
    SO: Literal[200, 500] = 200,
    # --- bacco baryonification inputs ---
    log10_M_c: Annotated[float, Field(ge=9.0, le=15.0, description="bacco: gas retention mass scale")] = 14.0,
    log10_eta: Annotated[float, Field(ge=-0.7, le=0.7)] = -0.3,
    log10_beta: Annotated[float, Field(ge=-1.0, le=0.7)] = -0.22,
    log10_M1_z0_cen: Annotated[float, Field(ge=9.0, le=13.0)] = 10.5,
    log10_theta_out: Annotated[float, Field(ge=0.0, le=0.5)] = 0.25,
    log10_theta_inn: Annotated[float, Field(ge=-2.0, le=-0.5)] = -0.86,
    log10_M_inn: Annotated[float, Field(ge=9.0, le=13.5)] = 13.4,
    # --- syren-baryon (CAMELS feedback) inputs ---
    A_SN1: Annotated[float, Field(ge=0.25, le=4.0)] = 1.0,
    A_SN2: Annotated[float, Field(ge=0.5, le=2.0)] = 1.0,
    A_AGN1: Annotated[float, Field(ge=0.25, le=4.0)] = 1.0,
    A_AGN2: Annotated[float, Field(ge=0.5, le=2.0)] = 1.0,
    # --- shared cosmology ---
    Om: Annotated[float, Field(ge=0.2, le=0.45)] = 0.31,
    Ob: Annotated[float, Field(ge=0.03, le=0.07)] = 0.049,
    h: Annotated[float, Field(ge=0.6, le=0.8)] = 0.67,
    ns: Annotated[float, Field(ge=0.9, le=1.02)] = 0.965,
    sigma8: Annotated[float, Field(ge=0.6, le=1.0)] = 0.8,
    mnu: Annotated[float, Field(ge=0.0, le=0.4)] = 0.0,
    k_min: Annotated[float, Field(ge=0.01)] = 0.1,
    k_max: Annotated[float, Field(le=12.0)] = 8.0,
    n_points: Annotated[int, Field(ge=10, le=1000)] = 200,
    z: Annotated[float, Field(ge=0.0, le=3.0)] = 0.0,
    return_data: Annotated[bool, Field(description="Include downsampled arrays in metadata.data.")] = False,
) -> ArtifactResult:
    """Compute the baryonic suppression S(k) = P_hydro(k) / P_gravity-only(k).

    Models parameterize feedback differently — only the matching parameter
    group is used:
    - "spk": SP(k) from the baryon fraction - halo mass relation (fb_a,
      fb_pow, fb_pivot; fb is in units of Omega_b/Omega_m). Most direct link
      to X-ray/SZ observations.
    - "bacco": baccoemu baryonification boost (7 log10 M_c...M_inn params).
    - "syren_<suite>": closed-form CAMELS fits per hydro code
      (IllustrisTNG / Astrid / SIMBA / Swift_EAGLE) driven by A_SN1/2,
      A_AGN1/2 — good for quantifying inter-suite systematic spread.

    S(k) is dimensionless (=1 means no baryonic effect); apply it to a
    gravity-only nonlinear P(k) with baryonify_pk (or compose_spectra) —
    do not multiply client-side. NOTE on redshift scans with "spk": the
    fb_a/fb_pow inputs are z-INDEPENDENT here, so scanning z at fixed
    defaults holds the baryon-fraction relation fixed while the halo
    population evolves — the resulting S(k,z) trend is an artifact of that
    choice, not a physical prediction. Supply z-appropriate fb parameters
    per call for physical z-evolution. Output CSV: k [h/Mpc], suppression.
    """
    k = k_grid(k_min, k_max, n_points)

    if model == "spk":
        import pyspk.model as spk
        with quiet():
            k_out, sup = spk.sup_model(SO=SO, z=z, fb_a=fb_a, fb_pow=fb_pow,
                                       fb_pivot=10 ** fb_pivot_log10Msun,
                                       k_array=k)
        sup = np.interp(k, np.ravel(k_out), np.ravel(sup))
        detail = f"fb_a={fb_a}, fb_pow={fb_pow}, SO={SO}"
    elif model == "bacco":
        import baccoemu
        emu = get_cached("baccoemu", baccoemu.Matter_powerspectrum)
        bp = dict(omega_cold=Om, omega_baryon=Ob, sigma8_cold=sigma8, ns=ns,
                  hubble=h, neutrino_mass=mnu, w0=-1.0, wa=0.0,
                  expfactor=1.0 / (1.0 + z),
                  M_c=log10_M_c, eta=log10_eta, beta=log10_beta,
                  M1_z0_cen=log10_M1_z0_cen, theta_out=log10_theta_out,
                  theta_inn=log10_theta_inn, M_inn=log10_M_inn)
        with quiet():
            _, sup = emu.get_baryonic_boost(k=k, **bp)
        sup = np.asarray(sup)
        detail = f"log10 M_c={log10_M_c}"
    else:
        from symbolic_pofk import syren_baryon
        suite = model.removeprefix("syren_")
        fn = getattr(syren_baryon, f"S_{suite}")
        sup = fn(k, Om, sigma8, A_SN1, A_SN2, A_AGN1, A_AGN2, 1.0 / (1.0 + z))
        detail = f"{suite}: A_SN=({A_SN1},{A_SN2}), A_AGN=({A_AGN1},{A_AGN2})"

    label = f"baryon suppression {model} ({detail}) z={z:g}"
    outdir = resolve_outdir(output_dir)
    slug = param_slug({"model": model, "detail": detail, "z": z,
                       "k0": k_min, "k1": k_max})
    path = outdir / f"baryon_suppression_{model}_z{z:g}_{slug}.csv"
    columns = {"k_h_per_Mpc": k, "suppression": sup}
    write_csv(path, columns,
              [f"label: {label}", "quantity: suppression",
               "units: k [h/Mpc], suppression [dimensionless]",
               f"model: {model}", f"z: {z:g}", f"detail: {detail}"])
    metadata = {"model": model, "detail": detail, "z": z,
                "units": {"k": "h/Mpc", "suppression": "dimensionless"},
                "stats": summary_stats(k, sup, "k", "S")}
    if model == "spk":
        metadata["z_scan_note"] = ("fb parameters are z-independent inputs; "
                                   "a z-scan at fixed fb is not a physical "
                                   "evolution prediction")
    if return_data:
        metadata["data"] = downsample_columns(columns)
    return ArtifactResult(
        status="success", files=[str(path)],
        message=f"Computed S(k) with {model} at z={z:g}: max suppression "
                f"{float(np.nanmin(sup)):.3f} at k={k[np.nanargmin(sup)]:.2f} h/Mpc.",
        metadata=metadata,
    )


@validate_call
def baryonify_pk(
    pk_file: Annotated[str, Field(min_length=1, description="Gravity-only nonlinear P(k) CSV from compute_nonlinear_pk (or compute_mg_pk).")],
    output_dir: Annotated[str, Field(min_length=1)],
    model: SuppressionModel = "spk",
    fb_a: Annotated[float, Field(ge=0.1, le=1.0)] = 0.4,
    fb_pow: Annotated[float, Field(ge=-0.5, le=1.0)] = 0.3,
    fb_pivot_log10Msun: Annotated[float, Field(ge=12.0, le=15.0)] = 13.5,
    SO: Literal[200, 500] = 200,
    log10_M_c: Annotated[float, Field(ge=9.0, le=15.0)] = 14.0,
    log10_eta: Annotated[float, Field(ge=-0.7, le=0.7)] = -0.3,
    log10_beta: Annotated[float, Field(ge=-1.0, le=0.7)] = -0.22,
    log10_M1_z0_cen: Annotated[float, Field(ge=9.0, le=13.0)] = 10.5,
    log10_theta_out: Annotated[float, Field(ge=0.0, le=0.5)] = 0.25,
    log10_theta_inn: Annotated[float, Field(ge=-2.0, le=-0.5)] = -0.86,
    log10_M_inn: Annotated[float, Field(ge=9.0, le=13.5)] = 13.4,
    A_SN1: Annotated[float, Field(ge=0.25, le=4.0)] = 1.0,
    A_SN2: Annotated[float, Field(ge=0.5, le=2.0)] = 1.0,
    A_AGN1: Annotated[float, Field(ge=0.25, le=4.0)] = 1.0,
    A_AGN2: Annotated[float, Field(ge=0.5, le=2.0)] = 1.0,
    return_data: Annotated[bool, Field(description="Include downsampled arrays in metadata.data.")] = False,
) -> ArtifactResult:
    """Apply baryonic suppression to a gravity-only P(k) file, server-side.

    Mirrors compute_mg_pk for baryons: reads the P(k) CSV, evaluates the
    chosen suppression model on the SAME k-grid and redshift (z is read
    from the input file header — no mismatch possible), and writes the
    baryonified spectrum with full provenance. This answers questions like
    "f(R) universe with feedback": feed it the output of compute_mg_pk.
    Parameter groups per model are as in compute_baryon_suppression.
    """
    header, cols = read_csv(pk_file)
    names = list(cols.keys())
    k = cols[names[0]]
    pk = cols[names[1]]
    if header.get("quantity") not in (None, "power_spectrum"):
        raise ValueError(f"{pk_file} has quantity="
                         f"{header.get('quantity')!r}; expected a power_spectrum file.")
    z_match = re.search(r"([\d.]+)", header.get("z", ""))
    z = float(z_match.group(1)) if z_match else 0.0

    sup_result = compute_baryon_suppression(
        output_dir=output_dir, model=model, fb_a=fb_a, fb_pow=fb_pow,
        fb_pivot_log10Msun=fb_pivot_log10Msun, SO=SO,
        log10_M_c=log10_M_c, log10_eta=log10_eta, log10_beta=log10_beta,
        log10_M1_z0_cen=log10_M1_z0_cen, log10_theta_out=log10_theta_out,
        log10_theta_inn=log10_theta_inn, log10_M_inn=log10_M_inn,
        A_SN1=A_SN1, A_SN2=A_SN2, A_AGN1=A_AGN1, A_AGN2=A_AGN2,
        k_min=float(max(k[0], 0.02)), k_max=float(min(k[-1], 12.0)),
        n_points=min(len(k), 500), z=min(z, 3.0))
    _, sup_cols = read_csv(sup_result.files[0])
    sup = np.interp(k, sup_cols["k_h_per_Mpc"], sup_cols["suppression"],
                    left=1.0, right=np.nan)
    pk_baryon = pk * sup
    keep = np.isfinite(pk_baryon)

    base_label = header.get("label", Path(pk_file).stem)
    label = f"{base_label} x {model} baryons"
    outdir = resolve_outdir(output_dir)
    slug = param_slug({"in": pk_file, "model": model, "z": z})
    path = outdir / f"pk_baryonified_{model}_z{z:g}_{slug}.csv"
    columns = {"k_h_per_Mpc": k[keep], "Pk_Mpc_over_h_cubed": pk_baryon[keep],
               "suppression": sup[keep]}
    write_csv(path, columns,
              [f"label: {label}", "quantity: power_spectrum",
               "variant: nonlinear_baryonified",
               "units: k [h/Mpc], Pk [(Mpc/h)^3]",
               f"model: {model}", f"z: {z:g}", f"input: {pk_file}"])
    metadata = {"model": model, "z": z, "input_file": pk_file,
                "input_label": base_label,
                "units": {"k": "h/Mpc", "Pk": "(Mpc/h)^3"},
                "stats": summary_stats(k[keep], pk_baryon[keep], "k", "Pk"),
                "suppression_stats": summary_stats(k[keep], sup[keep], "k", "S")}
    if return_data:
        metadata["data"] = downsample_columns(columns)
    return ArtifactResult(
        status="success", files=[str(path), sup_result.files[0]],
        message=f"Baryonified {base_label} with {model}: max suppression "
                f"{float(np.nanmin(sup[keep])):.3f}.",
        metadata=metadata,
    )


@validate_call
def emulate_subgrid_statistic(
    output_dir: Annotated[str, Field(min_length=1)],
    statistic: Literal["Pk", "GSMF", "CGD", "fGas", "BHMSM", "CSFR"] = "Pk",
    kappa_w: Annotated[float, Field(ge=2.0, le=4.0, description="Galactic wind efficiency")] = 3.0,
    e_w: Annotated[float, Field(ge=0.2, le=1.0, description="Wind energy fraction")] = 0.6,
    M_seed: Annotated[float, Field(ge=0.6, le=1.2, description="Black-hole seed mass in 1e6 Msun")] = 0.9,
    v_kin: Annotated[float, Field(ge=0.1, le=1.2, description="AGN kinetic wind velocity in 1e4 km/s")] = 0.6,
    e_kin: Annotated[float, Field(ge=0.02, le=1.2, description="AGN kinetic feedback efficiency x10")] = 0.6,
    z_index: Annotated[int, Field(ge=0, le=3, description="Snapshot index (0 = z~0)")] = 0,
    return_data: Annotated[bool, Field(description="Include downsampled arrays in metadata.data.")] = False,
) -> ArtifactResult:
    """Emulate a CRK-HACC hydro-simulation summary statistic vs subgrid parameters.

    GP emulator (Ramachandra et al. 2026) over 5 subgrid-physics parameters,
    trained on 64 CRK-HACC hydro simulations. Statistics: "Pk" (baryonic
    P(k) ratio), "GSMF" (stellar mass function), "CGD" (cluster gas density),
    "fGas" (gas fraction), "BHMSM" (BH-stellar mass relation), "CSFR" (star
    formation history). Returns the GP mean AND standard deviation — the
    uncertainty column is real emulator error, propagate it. The x-axis
    (mass, k, radius, or z depending on the statistic) is the emulator's
    native grid, written as the first CSV column.
    """
    from subgrid_emu.emulator import SubgridEmulator
    from subgrid_emu.model_metadata import TRAINING_GRIDS

    emu = get_cached(f"subgrid:{statistic}:{z_index}",
                     lambda: SubgridEmulator(statistic, z_index=z_index))
    params = np.array([[kappa_w, e_w, M_seed, v_kin, e_kin]])
    with quiet():
        mean, std = emu.predict(params)
    mean, std = np.ravel(mean), np.ravel(std)

    grid_info = TRAINING_GRIDS.get(statistic, {})
    x = None
    for key in ("x", "x_grid", "grid", "bins"):
        if isinstance(grid_info, dict) and key in grid_info:
            x = np.ravel(np.asarray(grid_info[key], dtype=float))
            break
    if x is None or len(x) != len(mean):
        x = np.arange(len(mean), dtype=float)
        x_name = "grid_index"
    else:
        x_name = str(grid_info.get("x_name", "x"))

    label = (f"subgrid {statistic} [kappa_w={kappa_w}, e_w={e_w}, "
             f"M_seed={M_seed}, v_kin={v_kin}, e_kin={e_kin}]")
    outdir = resolve_outdir(output_dir)
    slug = param_slug({"stat": statistic, "z": z_index, "kw": kappa_w,
                       "ew": e_w, "ms": M_seed, "vk": v_kin, "ek": e_kin})
    path = outdir / f"subgrid_{statistic}_z{z_index}_{slug}.csv"
    columns = {x_name: x, "mean": mean, "gp_std": std}
    write_csv(path, columns,
              [f"label: {label}", f"quantity: subgrid_{statistic}",
               f"units: {x_name}, statistic-native",
               f"statistic: {statistic}", f"z_index: {z_index}",
               f"params: kappa_w={kappa_w},e_w={e_w},M_seed={M_seed},"
               f"v_kin={v_kin},e_kin={e_kin}"])
    metadata = {"statistic": statistic, "z_index": z_index,
                "params": {"kappa_w": kappa_w, "e_w": e_w, "M_seed": M_seed,
                           "v_kin": v_kin, "e_kin": e_kin},
                "note": "gp_std column is the emulator 1-sigma uncertainty",
                "stats": summary_stats(x, mean, x_name, "mean")}
    if return_data:
        metadata["data"] = downsample_columns(columns)
    return ArtifactResult(
        status="success", files=[str(path)],
        message=f"Emulated {SUBGRID_STATS[statistic]} ({len(mean)} points) "
                "with GP uncertainty.",
        metadata=metadata,
    )
