"""Cross-cutting tools that operate on other tools' outputs: publication
figures from server CSVs, and batched parameter scans over any compute tool.

Both exist because of how agents actually use this server: research runs
issued 10-40 single-point calls per scan (one z, one feedback strength, one
Fisher step at a time) and then re-plotted the files many times.
"""

import importlib
import inspect
import itertools
import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, validate_call

from ..common import ArtifactResult, param_slug, read_csv, resolve_outdir
from ..plotting import DIMENSIONLESS, param_symbol, plot_files

__all__ = ["plot_emulator_curves", "scan_emulator_parameters"]

# tool name -> module that defines it (imported lazily: avoids import cycles)
SCANNABLE_TOOLS = {
    "compute_linear_pk": "tools.pk",
    "compute_nonlinear_pk": "tools.pk",
    "compute_mg_boost": "tools.gravity",
    "compute_mg_pk": "tools.gravity",
    "compute_baryon_suppression": "tools.baryons",
    "baryonify_pk": "tools.baryons",
    "emulate_subgrid_statistic": "tools.baryons",
    "compute_cmb_cls": "tools.cmb",
    "compute_lensing_cls": "tools.lss",
    "compute_galaxy_multipoles": "tools.lss",
    "compute_hmf": "tools.halos",
    "emulate_lya_p1d": "tools.igm",
}
ScannableTool = Literal[
    "compute_linear_pk", "compute_nonlinear_pk", "compute_mg_boost",
    "compute_mg_pk", "compute_baryon_suppression", "baryonify_pk",
    "emulate_subgrid_statistic", "compute_cmb_cls", "compute_lensing_cls",
    "compute_galaxy_multipoles", "compute_hmf", "emulate_lya_p1d"]

MAX_SCAN_RUNS = 60
MAX_REMOTE_RUNS = 6
_RESERVED = {"output_dir", "return_data"}


@validate_call
def plot_emulator_curves(
    curve_files: Annotated[list[str], Field(min_length=1, max_length=40, description="CSV files written by this server's compute tools (any family: P(k), B(k), S(k), HMF, Cl, P1D, composed products).")],
    output_dir: Annotated[str, Field(min_length=1)],
    title: Annotated[str | None, Field(description="Optional figure title (wrapped automatically). Leave unset for a paper-ready figure; captions go in the text.")] = None,
    labels: Annotated[list[str] | None, Field(description="Short legend entries, one per file, e.g. ['F5', 'F6', 'N1']. Matplotlib mathtext is allowed ('$|f_{R0}|=10^{-5}$'). Default: derived from the files, with words shared by every curve moved into the legend title.")] = None,
    ratio_panel: Annotated[Literal["auto", "on", "off"], Field(description="Lower panel with each curve divided by the reference curve. 'auto' = on for spectra/HMF/Cl with >= 2 curves, off for quantities that are already ratios (boost, suppression, composed B*S).")] = "auto",
    reference_index: Annotated[int, Field(ge=0, description="Which file is the ratio-panel denominator.")] = 0,
    color_values: Annotated[list[float] | None, Field(description="One number per file (e.g. the z or log10 M_c each curve was computed at). Switches to a sequential colormap with a colorbar — use for parameter sweeps with more than ~6 curves.")] = None,
    colorbar_label: Annotated[str | None, Field(description="Colorbar label when color_values is set, e.g. '$z$' or '$\\log_{10} M_c$'.")] = None,
    xlabel: Annotated[str | None, Field(description="Override the automatic x-axis label (keep it short).")] = None,
    ylabel: Annotated[str | None, Field(description="Override the automatic y-axis label (keep it short).")] = None,
    logx: Annotated[bool | None, Field(description="Force log/linear x. Default: automatic.")] = None,
    logy: Annotated[bool | None, Field(description="Force log/linear y. Default: automatic (log for P(k), HMF, lensing Cl; linear for ratios and CMB D_l).")] = None,
    allow_mixed_quantities: Annotated[bool, Field(description="Allow overlaying dimensional quantities of different kinds (e.g. P(k) with HMF). Dimensionless ones (boost, suppression, ratio, composed) always mix freely.")] = False,
    save_pdf: Annotated[bool, Field(description="Also write a vector PDF next to the PNG (for manuscripts).")] = False,
    output_name: Annotated[str | None, Field(description="Optional output file stem.")] = None,
) -> ArtifactResult:
    """Render a publication-quality figure from this server's output CSVs.

    The one plotting tool for every curve this server produces: P(k)
    backends, MG boosts B(k), baryonic suppressions S(k), composed B*S
    products, halo mass functions, lensing/CMB spectra, Lyman-alpha P1D.
    Axis labels come from each file's quantity and units (journal-style
    mathtext, no LaTeX install needed); uncertainty columns (gp_std,
    emulator_rel_std) are drawn as shaded 1-sigma bands; ratio quantities get a y=1
    reference line and linear axes. Pass short `labels` for the cleanest
    legend. Returns the PNG (plus PDF if save_pdf). Only plots files —
    to compute new curves call the compute_* / emulate_* tools first, or
    scan_emulator_parameters for a whole sweep.
    """
    headers = [read_csv(f)[0] for f in curve_files]
    quantities = [h.get("quantity", "unknown") for h in headers]
    dimensional = {q for q in quantities if q not in DIMENSIONLESS}
    mixed = len(dimensional) > 1 or (dimensional and len(set(quantities)) > 1)
    if mixed and not allow_mixed_quantities:
        listing = "; ".join(f"{Path(f).name}: {q}"
                            for f, q in zip(curve_files, quantities))
        raise ValueError(
            f"Refusing to overlay different quantities ({listing}) on one "
            "axis. Pass allow_mixed_quantities=true only if intentional.")

    outdir = resolve_outdir(output_dir)
    stem = output_name or f"figure_{param_slug({'f': tuple(curve_files)})}"
    info = plot_files(
        curve_files, outdir / f"{stem}.png", title=title, labels=labels,
        xlabel=xlabel, ylabel=ylabel, logx=logx, logy=logy,
        ratio_panel=ratio_panel, reference_index=reference_index,
        color_values=color_values, colorbar_label=colorbar_label,
        save_pdf=save_pdf)

    variants = {h.get("variant") for h in headers if h.get("variant")}
    message = f"Plotted {len(curve_files)} curves"
    if info["ratio_panel"]:
        message += f" with a ratio panel vs {Path(curve_files[reference_index]).name}"
    message += "."
    metadata = {"legend_labels": info["labels"], "quantities": quantities,
                "shared_caption": info["shared_caption"],
                "axes": {"logx": info["logx"], "logy": info["logy"]}}
    if len(variants) > 1:
        note = (f"inputs mix spectrum variants {sorted(variants)} — "
                "e.g. linear vs nonlinear; ensure this comparison is intended")
        message += f" NOTE: {note}."
        metadata["variant_warning"] = note
    return ArtifactResult(status="success", files=info["files"],
                          message=message, metadata=metadata)


def _tool_function(name: str):
    return getattr(importlib.import_module(SCANNABLE_TOOLS[name]), name)


def _fmt(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


@validate_call
def scan_emulator_parameters(
    tool: Annotated[ScannableTool, Field(description="The compute tool to run at every grid point.")],
    scan: Annotated[dict[str, list[float | int | str]], Field(description="Parameter name -> list of values, using that tool's own argument names, e.g. {'z': [0, 0.5, 1, 2]} or {'log10_M_c': [13.5, 14, 14.5], 'z': [0.3, 0.9]}. 1-3 parameters.")],
    output_dir: Annotated[str, Field(min_length=1)],
    fixed_args: Annotated[dict[str, Any] | None, Field(description="Arguments held fixed for every run (same names as the tool's own arguments), e.g. {'model': 'bacco', 'k_max': 5}.")] = None,
    combine: Annotated[Literal["product", "zip"], Field(description="'product' = every combination of the scan lists; 'zip' = pair the lists element-wise (all must have equal length), e.g. finite-difference steps.")] = "product",
    plot: Annotated[bool, Field(description="Also render figure(s): one curve per run, colored sequentially by the first scanned parameter, one figure per combination of the others (max 6 figures).")] = True,
    title: Annotated[str | None, Field(description="Optional title prefix for the figures.")] = None,
) -> ArtifactResult:
    """Run one compute tool over a grid of parameter values in a single call.

    Replaces long chains of single-point calls: redshift sweeps, feedback-
    strength sweeps, finite-difference steps for Fisher derivatives
    (combine='zip'), model grids. Every run goes through the target tool
    itself, so its validation, training-box warnings and file naming are
    unchanged; each output CSV is a normal file usable by compose_spectra
    and plot_emulator_curves. Returns all files, a JSON manifest mapping
    parameter values -> file, per-run warnings and unity crossings, and
    (by default) figures. Runs that fail are reported, not fatal. Limit:
    60 runs (6 when dispatch targets an HPC site — each run is one job).
    """
    fixed = dict(fixed_args or {})
    fn = _tool_function(tool)
    accepted = set(inspect.signature(fn).parameters) - _RESERVED
    if not 1 <= len(scan) <= 3:
        raise ValueError("scan must name 1-3 parameters.")
    unknown = sorted((set(scan) | set(fixed)) - accepted)
    if unknown:
        raise ValueError(f"{tool} has no argument(s) {unknown}. "
                         f"Valid: {', '.join(sorted(accepted))}")
    overlap = set(scan) & set(fixed)
    if overlap:
        raise ValueError(f"{sorted(overlap)} appear in both scan and fixed_args.")
    if any(len(v) == 0 for v in scan.values()):
        raise ValueError("Every scan list needs at least one value.")

    names = list(scan)
    if combine == "zip":
        lengths = {len(v) for v in scan.values()}
        if len(lengths) > 1:
            raise ValueError("combine='zip' needs equal-length scan lists.")
        points = list(zip(*scan.values()))
    else:
        points = list(itertools.product(*scan.values()))
    if len(points) > MAX_SCAN_RUNS:
        raise ValueError(f"{len(points)} runs requested; the limit is "
                         f"{MAX_SCAN_RUNS}. Coarsen the grid or split the scan.")

    from mcp_server.dispatch import remote_site  # lazy: server-only

    if remote_site() and len(points) > MAX_REMOTE_RUNS:
        raise ValueError(
            f"Dispatch targets {remote_site()}: each run is one facility job. "
            f"Scans are limited to {MAX_REMOTE_RUNS} runs there; switch to "
            "local execution for larger scans.")

    runs, files, errors = [], [], []
    for point in points:
        values = dict(zip(names, point))
        try:
            result = fn(output_dir=output_dir, **fixed, **values)
        except Exception as exc:  # report and keep scanning
            errors.append({"params": values, "error": str(exc).splitlines()[0][:300]})
            continue
        entry = {"params": values, "file": result.files[0],
                 "message": result.message}
        for key in ("in_training_box", "unity_crossings_k", "snapshot_z_used"):
            if key in result.metadata:
                entry[key] = result.metadata[key]
        runs.append(entry)
        files.append(result.files[0])
    if not runs:
        raise ValueError(f"All {len(points)} runs failed; first error: "
                         f"{errors[0]['error']}")

    outdir = resolve_outdir(output_dir)
    slug = param_slug({"tool": tool, "scan": json.dumps(scan, sort_keys=True),
                       "fixed": json.dumps(fixed, sort_keys=True, default=str),
                       "combine": combine})
    manifest = outdir / f"scan_{tool}_{slug}.json"
    manifest.write_text(json.dumps(
        {"tool": tool, "scan": scan, "fixed_args": fixed, "combine": combine,
         "runs": runs, "errors": errors}, indent=2, default=str),
        encoding="utf-8")

    figures = _scan_figures(runs, names, tool, outdir, slug, title) if plot else []

    out_of_box = sum(1 for r in runs if r.get("in_training_box") is False)
    message = (f"Scanned {tool} over {', '.join(names)}: {len(runs)}/"
               f"{len(points)} runs succeeded, {len(figures)} figure(s).")
    if out_of_box:
        message += f" WARNING: {out_of_box} run(s) outside the training box (see runs)."
    if errors:
        message += f" {len(errors)} run(s) failed (see errors)."
    return ArtifactResult(
        status="success", files=figures + [str(manifest)] + files,
        message=message,
        metadata={"tool": tool, "scanned": names, "fixed_args": fixed,
                  "runs": runs, "errors": errors, "figures": figures,
                  "manifest": str(manifest)})


def _scan_figures(runs, names, tool, outdir, slug, title) -> list[str]:
    """One figure per combination of the 2nd/3rd scanned parameters; curves
    within a figure are colored by the first scanned parameter."""
    first, rest = names[0], names[1:]
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault(tuple(r["params"][n] for n in rest), []).append(r)
    figures = []
    for gi, (key, members) in enumerate(list(groups.items())[:6]):
        values = [m["params"][first] for m in members]
        numeric = all(isinstance(v, (int, float)) for v in values)
        sequential = numeric and len(members) > 1
        if not sequential and len(members) > 8:
            members = members[:8]
        group_text = ", ".join(f"{param_symbol(n)} = {_fmt(v)}"
                               for n, v in zip(rest, key))
        fig_title = " — ".join(t for t in (title, group_text) if t) or None
        stem = f"scan_{tool}_{slug}" + (f"_{gi}" if len(groups) > 1 else "")
        info = plot_files(
            [m["file"] for m in members], outdir / f"{stem}.png",
            title=fig_title,
            labels=None if sequential else
            [f"{param_symbol(first)} = {_fmt(m['params'][first])}"
             for m in members],
            ratio_panel="off",
            color_values=[float(v) for v in values] if sequential else None,
            colorbar_label=param_symbol(first))
        figures.extend(info["files"])
    return figures
