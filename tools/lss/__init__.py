"""Large-scale structure observables: galaxy multipoles (EFTofLSS) and
weak-lensing angular spectra."""

import numpy as np
from typing import Annotated

from pydantic import Field, validate_call

from ..common import (ArtifactResult, downsample_columns, get_cached,
                      param_slug, quiet, read_csv, resolve_outdir,
                      summary_stats, write_csv)

__all__ = ["compute_galaxy_multipoles", "compute_lensing_cls"]


@validate_call
def compute_galaxy_multipoles(
    linear_pk_file: Annotated[str, Field(min_length=1, description="CSV from compute_linear_pk at the SAME redshift z (k should reach >= 1 h/Mpc, starting <= 1e-3)")],
    output_dir: Annotated[str, Field(min_length=1)],
    z: Annotated[float, Field(ge=0.0, le=3.0)] = 0.5,
    growth_rate_f: Annotated[float | None, Field(ge=0.0, le=1.5, description="Logarithmic growth rate f at z. If unset, computed as Omega_m(z)^0.55.")] = None,
    Om: Annotated[float, Field(ge=0.15, le=0.5, description="Used only to derive f when growth_rate_f is unset")] = 0.31,
    b1: Annotated[float, Field(ge=0.5, le=4.0, description="Linear galaxy bias")] = 2.0,
    b2: Annotated[float, Field(ge=-4.0, le=4.0)] = 0.8,
    b3: Annotated[float, Field(ge=-4.0, le=4.0)] = 0.2,
    b4: Annotated[float, Field(ge=-4.0, le=4.0)] = 0.8,
    k_max: Annotated[float, Field(ge=0.1, le=0.3)] = 0.25,
    number_density: Annotated[float, Field(gt=0, le=0.01, description="Galaxy number density in (h/Mpc)^3, sets stochastic-term normalization")] = 3e-4,
    return_data: Annotated[bool, Field(description="Include downsampled arrays in metadata.data.")] = False,
) -> ArtifactResult:
    """Compute one-loop EFTofLSS galaxy power spectrum multipoles P0/P2/P4 with PyBird.

    Model-independent: works for ANY cosmology because the input is a linear
    P(k) CSV from compute_linear_pk (use the same z!). EFT counterterms are
    set to conservative defaults (cct=0.2, cr1=cr2=-1, ce0=1) — fine for
    exploration; fit them to data for inference. Output CSV: k [h/Mpc],
    P0, P2, P4 [(Mpc/h)^3], valid to k_max.
    """
    header, cols = read_csv(linear_pk_file)
    kk = cols["k_h_per_Mpc"]
    pk_lin = cols["Pk_Mpc_over_h_cubed"]
    if kk[0] > 1.1e-3 or kk[-1] < 0.9:
        raise ValueError(
            "linear P(k) must span at least k = 1e-3 .. 1 h/Mpc for the "
            f"one-loop computation (got {kk[0]:.2g}..{kk[-1]:.2g}). Re-run "
            "compute_linear_pk with k_min=1e-3 (or smaller) and k_max>=1.")

    f = growth_rate_f
    if f is None:
        Omz = Om * (1 + z) ** 3 / (Om * (1 + z) ** 3 + 1 - Om)
        f = Omz ** 0.55

    from pybird.correlator import Correlator

    def factory():
        c = Correlator()
        with quiet():
            c.set({"output": "bPk", "multipole": 3, "kmax": 0.3,
                   "km": 0.7, "kr": 0.35, "nd": number_density, "z": z,
                   "optiresum": False, "with_resum": True})
        return c

    # Correlator carries z/nd state -> cache key includes them
    c = get_cached(f"pybird:{z}:{number_density}", factory)
    with quiet():
        c.compute({"kk": kk, "pk_lin": pk_lin, "f": f, "D": 1.0})
        bias = {"b1": b1, "b2": b2, "b3": b3, "b4": b4,
                "cct": 0.2, "cr1": -1.0, "cr2": -1.0,
                "ce0": 1.0, "ce1": 0.0, "ce2": -1.0}
        ps = np.asarray(c.get(bias))
    k_out = np.asarray(c.co.k)
    keep = k_out <= k_max
    label = f"EFTofLSS multipoles b1={b1} z={z:g}"
    outdir = resolve_outdir(output_dir)
    slug = param_slug({"z": z, "b1": b1, "b2": b2, "b3": b3, "b4": b4,
                       "f": round(f, 4), "in": linear_pk_file})
    path = outdir / f"galaxy_multipoles_z{z:g}_{slug}.csv"
    columns = {"k_h_per_Mpc": k_out[keep], "P0": ps[0][keep],
               "P2": ps[1][keep], "P4": ps[2][keep]}
    write_csv(path, columns,
              [f"label: {label}", "quantity: multipoles",
               "units: k [h/Mpc], P_l [(Mpc/h)^3]",
               f"z: {z:g}", f"f: {f:.4f}",
               f"bias: b1={b1},b2={b2},b3={b3},b4={b4}",
               f"input: {linear_pk_file}"])
    metadata = {"z": z, "growth_rate_f": round(f, 4), "bias": bias,
                "units": {"k": "h/Mpc", "Pl": "(Mpc/h)^3"},
                "note": "one-loop EFT with default counterterms",
                "stats": summary_stats(k_out[keep], ps[0][keep], "k", "P0")}
    if return_data:
        metadata["data"] = downsample_columns(columns)
    return ArtifactResult(
        status="success", files=[str(path)],
        message=f"Computed P0/P2/P4 to k={k_max} h/Mpc at z={z:g} "
                f"(f={f:.3f}, b1={b1}).",
        metadata=metadata,
    )


@validate_call
def compute_lensing_cls(
    output_dir: Annotated[str, Field(min_length=1)],
    Om: Annotated[float, Field(ge=0.15, le=0.5)] = 0.31,
    Ob: Annotated[float, Field(ge=0.03, le=0.07)] = 0.049,
    h: Annotated[float, Field(ge=0.55, le=0.85)] = 0.67,
    ns: Annotated[float, Field(ge=0.85, le=1.05)] = 0.965,
    sigma8: Annotated[float, Field(ge=0.6, le=1.0)] = 0.81,
    w0: Annotated[float, Field(ge=-1.5, le=-0.5)] = -1.0,
    z_source: Annotated[float, Field(ge=0.3, le=3.0, description="Mean redshift of the Smail source distribution")] = 1.0,
    ell_min: Annotated[int, Field(ge=2)] = 10,
    ell_max: Annotated[int, Field(le=5000)] = 3000,
    n_ell: Annotated[int, Field(ge=5, le=200)] = 50,
    return_data: Annotated[bool, Field(description="Include downsampled arrays in metadata.data.")] = False,
) -> ArtifactResult:
    """Compute weak-lensing convergence angular power spectrum with jax-cosmo.

    Limber-approximation cosmic-shear Cl for a single Smail n(z) source bin
    (a=2, b=2, z0 tuned so <z> ~ z_source). Fully differentiable backend;
    useful as the projection layer over emulated P(k). Output CSV: ell,
    Cl_kappa (dimensionless).
    """
    import jax.numpy as jnp
    import jax_cosmo as jc

    cosmo = jc.Cosmology(Omega_c=Om - Ob, Omega_b=Ob, h=h, n_s=ns,
                         sigma8=sigma8, Omega_k=0.0, w0=w0, wa=0.0)
    nz = jc.redshift.smail_nz(1.0, 2.0, z_source / 2.0)
    probe = jc.probes.WeakLensing([nz])
    ell = np.unique(np.geomspace(ell_min, ell_max, n_ell).astype(int)).astype(float)
    with quiet():
        cl = np.ravel(np.asarray(
            jc.angular_cl.angular_cl(cosmo, jnp.asarray(ell), [probe])))

    label = f"WL kappa Cl (z_s~{z_source:g}, sigma8={sigma8})"
    outdir = resolve_outdir(output_dir)
    slug = param_slug({"Om": Om, "sigma8": sigma8, "w0": w0, "zs": z_source})
    path = outdir / f"lensing_cls_zs{z_source:g}_{slug}.csv"
    columns = {"ell": ell, "Cl_kappa": cl}
    write_csv(path, columns,
              [f"label: {label}", "quantity: cl",
               "units: ell [multipole], Cl [dimensionless]",
               f"params: Om={Om},sigma8={sigma8},w0={w0}"])
    metadata = {"z_source": z_source, "units": {"Cl": "dimensionless"},
                "approximation": "Limber, single Smail source bin",
                "stats": summary_stats(ell, cl, "ell", "Cl")}
    if return_data:
        metadata["data"] = downsample_columns(columns)
    return ArtifactResult(
        status="success", files=[str(path)],
        message=f"Computed lensing Cl for {len(ell)} multipoles "
                f"l = {int(ell[0])}..{int(ell[-1])}.",
        metadata=metadata,
    )
