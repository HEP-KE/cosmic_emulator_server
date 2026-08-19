"""Matter power spectrum tools: linear and nonlinear P(k) from six backends,
plus spectrum composition and comparison plotting."""

import numpy as np
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, validate_call

from ..common import (ArtifactResult, downsample_columns, k_grid, param_slug,
                      plot_curves, read_csv, resolve_outdir, summary_stats,
                      varied_label, write_csv)
from . import backends

__all__ = ["compute_linear_pk", "compute_nonlinear_pk", "compose_spectra",
           "plot_pk_comparison"]

LinearBackend = Literal["camb", "syren", "baccoemu"]
NonlinearBackend = Literal["camb_hmcode", "syren_halofit", "baccoemu",
                           "euclidemu2", "csst", "gokunemu", "miratitan"]

_DEFAULTS = {"Om": 0.31, "Ob": 0.049, "h": 0.67, "ns": 0.965, "As": 2.1e-9,
             "sigma8": None, "mnu": 0.0, "w0": -1.0, "wa": 0.0}

_SIGMA8_BACKENDS = {"syren", "syren_halofit", "baccoemu", "miratitan"}


def _params(Om, Ob, h, ns, As, sigma8, mnu, w0, wa):
    return {"Om": Om, "Ob": Ob, "h": h, "ns": ns, "As": As,
            "sigma8": sigma8, "mnu": mnu, "w0": w0, "wa": wa}


def _run(kind, backend, params, k, z, output_dir, return_data) -> ArtifactResult:
    table = backends.LINEAR_BACKENDS if kind == "linear" else backends.NONLINEAR_BACKENDS
    pk = table[backend](params, k, z)

    box_params = dict(params)
    if backend in _SIGMA8_BACKENDS and box_params.get("sigma8") is None:
        box_params["sigma8"] = backends._resolve_sigma8(params)
    in_box, warnings = backends.check_box(backend, box_params, z)

    shown = {p: v for p, v in params.items() if v is not None}
    label = varied_label(f"{backend} {kind} z={z:g}", shown, _DEFAULTS)
    columns = {"k_h_per_Mpc": k, "Pk_Mpc_over_h_cubed": pk}
    outdir = resolve_outdir(output_dir)
    path = outdir / f"pk_{kind}_{backend}_z{z:g}_{param_slug(shown)}.csv"
    write_csv(path, columns,
              [f"label: {label}", "quantity: power_spectrum",
               f"variant: {kind}", "units: k [h/Mpc], Pk [(Mpc/h)^3]",
               f"backend: {backend}", f"z: {z:g}", f"params: {shown}"])

    metadata = {
        "backend": backend, "z": z, "params": shown,
        "in_training_box": in_box, "extrapolation_warnings": warnings,
        "units": {"k": "h/Mpc", "Pk": "(Mpc/h)^3"},
        "sigma8_convention": "sigma8 is sigma8(z=0); the spectrum is evolved to z",
        "stats": summary_stats(k, pk, "k", "Pk"),
    }
    if return_data:
        metadata["data"] = downsample_columns(columns)
    message = (f"Computed {kind} P(k) with {backend} at z={z:g} "
               f"({len(k)} points, k = {k[0]:.4g}..{k[-1]:.4g} h/Mpc).")
    if warnings:
        message += " WARNING: outside training box — " + "; ".join(warnings)
    return ArtifactResult(status="success", files=[str(path)],
                          message=message, metadata=metadata)


@validate_call
def compute_linear_pk(
    output_dir: Annotated[str, Field(min_length=1)],
    backend: LinearBackend = "syren",
    Om: Annotated[float, Field(ge=0.15, le=0.5, description="Total matter density Omega_m")] = 0.31,
    Ob: Annotated[float, Field(ge=0.03, le=0.07)] = 0.049,
    h: Annotated[float, Field(ge=0.5, le=0.9)] = 0.67,
    ns: Annotated[float, Field(ge=0.85, le=1.1)] = 0.965,
    As: Annotated[float, Field(ge=1e-9, le=4e-9)] = 2.1e-9,
    sigma8: Annotated[float | None, Field(ge=0.5, le=1.2, description="sigma8 at z=0 (the spectrum is evolved to the requested z). If set, overrides As for backends parameterized by sigma8 (syren, baccoemu); if unset it is derived from As with CAMB.")] = None,
    mnu: Annotated[float, Field(ge=0.0, le=0.5, description="Sum of neutrino masses in eV")] = 0.0,
    w0: Annotated[float, Field(ge=-1.5, le=-0.5)] = -1.0,
    wa: Annotated[float, Field(ge=-0.7, le=0.5)] = 0.0,
    k_min: Annotated[float, Field(ge=1e-4)] = 0.001,
    k_max: Annotated[float, Field(le=50.0)] = 5.0,
    n_points: Annotated[int, Field(ge=10, le=2000)] = 300,
    z: Annotated[float, Field(ge=0.0, le=10.0)] = 0.0,
    return_data: Annotated[bool, Field(description="Include a downsampled (<=80 point) copy of the arrays in metadata.data, for clients that cannot fetch the CSV.")] = False,
) -> ArtifactResult:
    """Compute a LINEAR matter power spectrum P(k) and write it to CSV.

    Backends: "syren" (closed-form Bartlett et al. fit, microseconds, ~0.2%),
    "camb" (exact Boltzmann solve, ~1 s), "baccoemu" (NN emulator, note it
    treats the given Om/sigma8 as cold-matter quantities — exact for mnu=0).
    Convention: sigma8 always means sigma8(z=0); the returned spectrum is
    evolved to the requested z. Each response reports in_training_box and
    quotable summary stats; set return_data=true to also get the arrays
    inline. Output columns: k [h/Mpc], P(k) [(Mpc/h)^3]. Pass the returned
    file path to other tools — never copy full arrays between tools.
    """
    return _run("linear", backend,
                _params(Om, Ob, h, ns, As, sigma8, mnu, w0, wa),
                k_grid(k_min, k_max, n_points), z, output_dir, return_data)


@validate_call
def compute_nonlinear_pk(
    output_dir: Annotated[str, Field(min_length=1)],
    backend: NonlinearBackend = "baccoemu",
    Om: Annotated[float, Field(ge=0.15, le=0.5)] = 0.31,
    Ob: Annotated[float, Field(ge=0.03, le=0.07)] = 0.049,
    h: Annotated[float, Field(ge=0.5, le=0.9)] = 0.67,
    ns: Annotated[float, Field(ge=0.85, le=1.1)] = 0.965,
    As: Annotated[float, Field(ge=1e-9, le=4e-9)] = 2.1e-9,
    sigma8: Annotated[float | None, Field(ge=0.5, le=1.2, description="sigma8 at z=0; see compute_linear_pk for the convention.")] = None,
    mnu: Annotated[float, Field(ge=0.0, le=0.5)] = 0.0,
    w0: Annotated[float, Field(ge=-1.5, le=-0.5)] = -1.0,
    wa: Annotated[float, Field(ge=-0.7, le=0.5)] = 0.0,
    k_min: Annotated[float, Field(ge=1e-3)] = 0.01,
    k_max: Annotated[float, Field(le=10.0, description="Backend k-limits: baccoemu 4.9, euclidemu2 9.4, csst/gokunemu 10; the default works everywhere.")] = 4.5,
    n_points: Annotated[int, Field(ge=10, le=2000)] = 300,
    z: Annotated[float, Field(ge=0.0, le=3.0)] = 0.0,
    return_data: Annotated[bool, Field(description="Include downsampled arrays in metadata.data.")] = False,
) -> ArtifactResult:
    """Compute a NONLINEAR matter power spectrum P(k) and write it to CSV.

    Backends and their niches:
    - "baccoemu": NN boost x linear, 1-2% to k=5, z<=1.5 (default)
    - "euclidemu2": Euclid boost x CAMB linear, ~1% to k=10, nu-w0waCDM
    - "csst": Kun-suite GP, 1% to k=10, z<=2
    - "gokunemu": widest 10-parameter space (w0waCDM + mnu)
    - "miratitan": Mira-Titan IV / CosmicEmu (via pyccl), 2-3%, the
      Moran et al. 2022 HACC suite — pairs with the miratitan HMF backend;
      narrowest parameter box (sigma8 0.7-0.9, z <= 2), check the response's
      in_training_box
    - "camb_hmcode": HMcode-2020 halo model (~2.5%, any cosmology)
    - "syren_halofit": closed-form, fastest, ~1%

    The schema accepts the UNION of all backend ranges so backends can be
    compared; each response then reports in_training_box for the specific
    backend and lists any extrapolation warnings — check them before using
    the numbers. Running several backends at one cosmology and comparing
    with plot_pk_comparison is the recommended production cross-check.
    Output: k [h/Mpc], P(k) [(Mpc/h)^3].
    """
    return _run("nonlinear", backend,
                _params(Om, Ob, h, ns, As, sigma8, mnu, w0, wa),
                k_grid(k_min, k_max, n_points), z, output_dir, return_data)


@validate_call
def compose_spectra(
    spectrum_files: Annotated[list[str], Field(min_length=2, max_length=6, description="CSV files written by this server's tools. The first file defines the output k-grid.")],
    output_dir: Annotated[str, Field(min_length=1)],
    op: Literal["multiply", "divide", "ratio"] = "multiply",
    output_name: Annotated[str | None, Field(description="Optional output file stem (default derives from inputs).")] = None,
    return_data: Annotated[bool, Field(description="Include downsampled arrays in metadata.data.")] = False,
) -> ArtifactResult:
    """Combine spectra/boost/suppression CSVs arithmetically on a common grid.

    The server-side way to build composed quantities: e.g. multiply a
    gravity-only nonlinear P(k) by a baryonic suppression, or divide two
    P(k) files to get their ratio ("divide" and "ratio" are synonyms).
    Later files are interpolated onto the FIRST file's k-grid; the overlap
    must cover at least half of that grid, otherwise the call errors.
    Provenance (input files, their quantities and labels) is written into
    the output header and metadata. Use this instead of ever multiplying
    numbers client-side.
    """
    curves = []
    for path_str in spectrum_files:
        header, cols = read_csv(path_str)
        names = list(cols.keys())
        curves.append({"path": path_str, "header": header,
                       "x": cols[names[0]], "y": cols[names[1]],
                       "quantity": header.get("quantity", "unknown"),
                       "label": header.get("label", Path(path_str).stem)})

    x = curves[0]["x"]
    result = np.array(curves[0]["y"], dtype=float)
    for c in curves[1:]:
        lo, hi = max(x.min(), c["x"].min()), min(x.max(), c["x"].max())
        overlap = np.mean((x >= lo) & (x <= hi))
        if overlap < 0.5:
            raise ValueError(
                f"k-grids barely overlap ({overlap:.0%} of the base grid): "
                f"{c['path']} spans {c['x'].min():.3g}..{c['x'].max():.3g} vs "
                f"base {x.min():.3g}..{x.max():.3g}. Recompute on a matching grid.")
        y = np.interp(x, c["x"], c["y"], left=np.nan, right=np.nan)
        result = result * y if op == "multiply" else result / y
    keep = np.isfinite(result)

    quantities = [c["quantity"] for c in curves]
    opname = {"multiply": "x", "divide": "/", "ratio": "/"}[op]
    label = f" {opname} ".join(c["label"] for c in curves)
    out_q = ("ratio" if op in ("divide", "ratio")
             else quantities[0] if len(set(quantities)) == 1 else "composed")

    stem = output_name or f"composed_{param_slug({'f': tuple(spectrum_files), 'op': op})}"
    outdir = resolve_outdir(output_dir)
    path = outdir / f"{stem}.csv"
    columns = {"k_h_per_Mpc": x[keep], "value": result[keep]}
    write_csv(path, columns,
              [f"label: {label}", f"quantity: {out_q}",
               "units: k [h/Mpc], value [product of input units]",
               f"op: {op}"] +
              [f"input{i}: {c['path']} ({c['quantity']})"
               for i, c in enumerate(curves)])
    metadata = {"op": op, "inputs": [{"file": c["path"],
                                      "quantity": c["quantity"],
                                      "label": c["label"]} for c in curves],
                "output_quantity": out_q,
                "stats": summary_stats(x[keep], result[keep], "k", "value")}
    if return_data:
        metadata["data"] = downsample_columns(columns)
    return ArtifactResult(
        status="success", files=[str(path)],
        message=f"Composed {len(curves)} files with '{op}' on "
                f"{int(keep.sum())} shared k-points.",
        metadata=metadata)


@validate_call
def plot_pk_comparison(
    spectrum_files: Annotated[list[str], Field(min_length=1, max_length=8)],
    output_dir: Annotated[str, Field(min_length=1)],
    reference_index: Annotated[int, Field(ge=0)] = 0,
    title: str = "Comparison",
    allow_mixed_quantities: Annotated[bool, Field(description="Set true to deliberately overlay different quantities (e.g. a boost on top of spectra). Default false: mixing errors out.")] = False,
    output_name: Annotated[str | None, Field(description="Optional output file stem (default derives from the inputs).")] = None,
) -> ArtifactResult:
    """Plot curves from CSVs written by this server's tools, with a ratio panel.

    Refuses to overlay files of different `quantity` (power_spectrum vs
    suppression vs hmf vs cl ...) unless allow_mixed_quantities=true —
    mixed-axis plots are usually a bug, and axis labels come from the
    files' own unit headers. reference_index selects the ratio-panel
    denominator.
    """
    if reference_index >= len(spectrum_files):
        raise ValueError("reference_index is out of range for spectrum_files.")

    headers = [read_csv(f)[0] for f in spectrum_files]
    quantities = [h.get("quantity", "unknown") for h in headers]
    variants = {h.get("variant") for h in headers if h.get("variant")}
    if len(set(quantities)) > 1 and not allow_mixed_quantities:
        listing = "; ".join(f"{Path(f).name}: {q}"
                            for f, q in zip(spectrum_files, quantities))
        raise ValueError(
            f"Refusing to overlay different quantities ({listing}). "
            "Pass allow_mixed_quantities=true only if this is intentional.")

    units = headers[0].get("units", "")
    ylabel = units.split(",")[-1].strip() if "," in units else quantities[0]
    xlabel = units.split(",")[0].strip() if "," in units else "x"
    logy = quantities[0] in ("power_spectrum", "hmf", "cl", "multipoles", "p1d",
                             "unknown", "composed")

    outdir = resolve_outdir(output_dir)
    stem = output_name or f"comparison_{param_slug({'f': tuple(spectrum_files)})}"
    path = outdir / f"{stem}.png"
    labels = plot_curves(spectrum_files, path, title=title, ylabel=ylabel,
                         xlabel=xlabel, logy=logy,
                         ratio_reference=reference_index)
    message = f"Plotted {len(labels)} curves (ratio vs {labels[reference_index]})."
    metadata = {"labels": labels, "quantities": quantities}
    if len(variants) > 1:
        note = (f"inputs mix spectrum variants {sorted(variants)} — "
                "e.g. linear vs nonlinear; ensure this comparison is intended")
        message += f" NOTE: {note}."
        metadata["variant_warning"] = note
    return ArtifactResult(status="success", files=[str(path)],
                          message=message, metadata=metadata)
