"""Matter power spectrum tools: linear and nonlinear P(k) from six backends."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, validate_call

from ..common import (ArtifactResult, k_grid, plot_curves, resolve_outdir,
                      write_csv)
from . import backends

__all__ = ["compute_linear_pk", "compute_nonlinear_pk", "plot_pk_comparison"]

LinearBackend = Literal["camb", "syren", "baccoemu"]
NonlinearBackend = Literal["camb_hmcode", "syren_halofit", "baccoemu",
                           "euclidemu2", "csst", "gokunemu"]


def _params(Om, Ob, h, ns, As, sigma8, mnu, w0, wa):
    return {"Om": Om, "Ob": Ob, "h": h, "ns": ns, "As": As,
            "sigma8": sigma8, "mnu": mnu, "w0": w0, "wa": wa}


def _run(kind: str, backend: str, params: dict, k, z, output_dir) -> ArtifactResult:
    table = backends.LINEAR_BACKENDS if kind == "linear" else backends.NONLINEAR_BACKENDS
    pk = table[backend](params, k, z)
    label = f"{backend} {kind} z={z:g}"
    outdir = resolve_outdir(output_dir)
    path = outdir / f"pk_{kind}_{backend}_z{z:g}.csv"
    shown = {p: v for p, v in params.items() if v is not None}
    write_csv(path, {"k_h_per_Mpc": k, "Pk_Mpc_over_h_cubed": pk},
              [f"label: {label}", f"backend: {backend}", f"z: {z:g}",
               f"params: {shown}"])
    return ArtifactResult(
        status="success",
        files=[str(path)],
        message=f"Computed {kind} P(k) with {backend} at z={z:g} "
                f"({len(k)} points, k = {k[0]:.4g}..{k[-1]:.4g} h/Mpc).",
        metadata={"backend": backend, "z": z, "params": shown,
                  "units": {"k": "h/Mpc", "Pk": "(Mpc/h)^3"}},
    )


@validate_call
def compute_linear_pk(
    output_dir: Annotated[str, Field(min_length=1)],
    backend: LinearBackend = "syren",
    Om: Annotated[float, Field(ge=0.15, le=0.5, description="Total matter density Omega_m")] = 0.31,
    Ob: Annotated[float, Field(ge=0.03, le=0.07)] = 0.049,
    h: Annotated[float, Field(ge=0.5, le=0.9)] = 0.67,
    ns: Annotated[float, Field(ge=0.85, le=1.1)] = 0.965,
    As: Annotated[float, Field(ge=1e-9, le=4e-9)] = 2.1e-9,
    sigma8: Annotated[float | None, Field(ge=0.5, le=1.2, description="If set, overrides As for backends parameterized by sigma8 (syren, baccoemu). If unset it is derived from As with CAMB.")] = None,
    mnu: Annotated[float, Field(ge=0.0, le=0.5, description="Sum of neutrino masses in eV")] = 0.0,
    w0: Annotated[float, Field(ge=-1.5, le=-0.5)] = -1.0,
    wa: Annotated[float, Field(ge=-0.7, le=0.5)] = 0.0,
    k_min: Annotated[float, Field(ge=1e-4)] = 0.001,
    k_max: Annotated[float, Field(le=50.0)] = 5.0,
    n_points: Annotated[int, Field(ge=10, le=2000)] = 300,
    z: Annotated[float, Field(ge=0.0, le=10.0)] = 0.0,
) -> ArtifactResult:
    """Compute a LINEAR matter power spectrum P(k) and write it to CSV.

    Backends: "syren" (closed-form Bartlett et al. fit, microseconds, ~0.2%),
    "camb" (exact Boltzmann solve, ~1 s), "baccoemu" (NN emulator, note it
    treats the given Om/sigma8 as cold-matter quantities — exact for mnu=0).
    Output columns: k [h/Mpc], P(k) [(Mpc/h)^3]. Pass the returned file path
    to other tools (plot_pk_comparison, compute_galaxy_multipoles) — never
    copy the numbers. Check describe_emulator for each backend's valid
    parameter ranges; out-of-range inputs raise errors or extrapolate.
    """
    return _run("linear", backend,
                _params(Om, Ob, h, ns, As, sigma8, mnu, w0, wa),
                k_grid(k_min, k_max, n_points), z, output_dir)


@validate_call
def compute_nonlinear_pk(
    output_dir: Annotated[str, Field(min_length=1)],
    backend: NonlinearBackend = "baccoemu",
    Om: Annotated[float, Field(ge=0.15, le=0.5)] = 0.31,
    Ob: Annotated[float, Field(ge=0.03, le=0.07)] = 0.049,
    h: Annotated[float, Field(ge=0.5, le=0.9)] = 0.67,
    ns: Annotated[float, Field(ge=0.85, le=1.1)] = 0.965,
    As: Annotated[float, Field(ge=1e-9, le=4e-9)] = 2.1e-9,
    sigma8: Annotated[float | None, Field(ge=0.5, le=1.2)] = None,
    mnu: Annotated[float, Field(ge=0.0, le=0.5)] = 0.0,
    w0: Annotated[float, Field(ge=-1.5, le=-0.5)] = -1.0,
    wa: Annotated[float, Field(ge=-0.7, le=0.5)] = 0.0,
    k_min: Annotated[float, Field(ge=1e-3)] = 0.01,
    k_max: Annotated[float, Field(le=10.0)] = 5.0,
    n_points: Annotated[int, Field(ge=10, le=2000)] = 300,
    z: Annotated[float, Field(ge=0.0, le=3.0)] = 0.0,
) -> ArtifactResult:
    """Compute a NONLINEAR matter power spectrum P(k) and write it to CSV.

    Backends and their niches:
    - "baccoemu": NN boost x linear, 1-2% to k=5, z<=1.5 (default)
    - "euclidemu2": Euclid boost x CAMB linear, ~1% to k=10, nu-w0waCDM
    - "csst": Kun-suite GP, 1% to k=10, z<=2
    - "gokunemu": widest 10-parameter space (w0waCDM + mnu)
    - "camb_hmcode": HMcode-2020 halo model (~2.5%, any cosmology)
    - "syren_halofit": closed-form, fastest, ~1%

    Running several backends at the same cosmology and comparing with
    plot_pk_comparison is the recommended cross-check for production results.
    Output: k [h/Mpc], P(k) [(Mpc/h)^3]. Each backend has its own valid
    parameter box — see describe_emulator; errors name the violated range.
    """
    return _run("nonlinear", backend,
                _params(Om, Ob, h, ns, As, sigma8, mnu, w0, wa),
                k_grid(k_min, k_max, n_points), z, output_dir)


@validate_call
def plot_pk_comparison(
    spectrum_files: Annotated[list[str], Field(min_length=1, max_length=8)],
    output_dir: Annotated[str, Field(min_length=1)],
    reference_index: Annotated[int, Field(ge=0)] = 0,
    title: str = "Matter power spectrum comparison",
) -> ArtifactResult:
    """Plot P(k) curves from CSVs written by the compute tools, with a ratio panel.

    Use after compute_linear_pk / compute_nonlinear_pk / compute_mg_pk to
    compare backends, cosmologies, or gravity models. reference_index selects
    which file the lower panel divides by.
    """
    if reference_index >= len(spectrum_files):
        raise ValueError("reference_index is out of range for spectrum_files.")
    outdir = resolve_outdir(output_dir)
    path = outdir / "pk_comparison.png"
    labels = plot_curves(spectrum_files, path, title=title,
                         ylabel=r"$P(k)\ [(\mathrm{Mpc}/h)^3]$",
                         ratio_reference=reference_index)
    return ArtifactResult(
        status="success", files=[str(path)],
        message=f"Plotted {len(labels)} spectra (ratio vs {labels[reference_index]}).",
        metadata={"labels": labels},
    )
