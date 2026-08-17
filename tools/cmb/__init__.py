"""CMB angular power spectrum tools.

Two emulator backends with DIFFERENT native conventions, unified here:
- cosmopower_jax: raw dimensionless Cl (CLASS convention), l = 2..2508
- capse (jaxcapse): Dl in muK^2 directly, l = 2..5000, trained on CAMB

Every output CSV contains Dl = l(l+1)Cl/2pi in muK^2; the conversion for
cosmopower_jax [Dl = Cl * (T_CMB*1e6)^2 * l(l+1)/2pi] was verified against
CAMB 1.6.6 at the Planck point (agreement ~0.01%).
"""

import numpy as np
from typing import Annotated, Literal

from pydantic import Field, validate_call

from ..common import (ArtifactResult, T_CMB_UK, get_cached, quiet,
                      resolve_outdir, write_csv)

__all__ = ["compute_cmb_cls", "plot_cmb_spectra"]

Spectrum = Literal["TT", "TE", "EE", "PP"]
Backend = Literal["capse", "cosmopower_jax"]

_CP_PROBES = {"TT": "cmb_tt", "TE": "cmb_te", "EE": "cmb_ee"}


def _capse_dl(spectrum: str, params: dict) -> tuple[np.ndarray, np.ndarray]:
    import jaxcapse

    def factory():
        return jaxcapse.load_emulator(str(jaxcapse.get_emulator_path(spectrum)))

    emu = get_cached(f"capse:{spectrum}", factory)
    # Capse parameter order: ln10As, ns, H0, omega_b, omega_cdm, tau
    x = np.array([params["ln10As"], params["ns"], params["h"] * 100,
                  params["omega_b"], params["omega_cdm"], params["tau"]])
    with quiet():
        dl = np.asarray(emu.get_Cl(x))
    ell = np.arange(2, 2 + len(dl))
    return ell, dl


def _cosmopower_dl(spectrum: str, params: dict) -> tuple[np.ndarray, np.ndarray]:
    from cosmopower_jax.cosmopower_jax import CosmoPowerJAX
    if spectrum not in _CP_PROBES:
        raise ValueError("cosmopower_jax backend supports TT, TE, EE "
                         "(PP requires the capse backend).")

    emu = get_cached(f"cosmopower_jax:{spectrum}",
                     lambda: CosmoPowerJAX(probe=_CP_PROBES[spectrum]))
    # cosmopower parameter order: omega_b, omega_cdm, h, tau, ns, ln10As
    x = np.array([params["omega_b"], params["omega_cdm"], params["h"],
                  params["tau"], params["ns"], params["ln10As"]])
    with quiet():
        cl_raw = np.asarray(emu.predict(x))
    ell = np.arange(2, 2 + len(cl_raw))
    dl = cl_raw * T_CMB_UK**2 * ell * (ell + 1) / (2 * np.pi)
    return ell, dl


@validate_call
def compute_cmb_cls(
    output_dir: Annotated[str, Field(min_length=1)],
    spectrum: Spectrum = "TT",
    backend: Backend = "capse",
    omega_b: Annotated[float, Field(ge=0.019, le=0.026, description="Physical baryon density omega_b h^2")] = 0.022,
    omega_cdm: Annotated[float, Field(ge=0.09, le=0.15, description="Physical CDM density omega_c h^2")] = 0.12,
    h: Annotated[float, Field(ge=0.6, le=0.76)] = 0.67,
    tau: Annotated[float, Field(ge=0.02, le=0.12, description="Optical depth to reionization")] = 0.055,
    ns: Annotated[float, Field(ge=0.92, le=1.01)] = 0.965,
    ln10As: Annotated[float, Field(ge=2.7, le=3.3, description="ln(10^10 As)")] = 3.045,
    ell_max: Annotated[int, Field(ge=100, le=5000)] = 2500,
) -> ArtifactResult:
    """Compute a CMB angular power spectrum Dl = l(l+1)Cl/2pi in muK^2.

    Backends: "capse" (Capse.jl weights, l up to 5000, matches CAMB
    high-precision to ~0.01%; supports TT/TE/EE/PP) and "cosmopower_jax"
    (CosmoPower weights, l up to 2508, TT/TE/EE). Both are millisecond
    evaluations trained on Planck-neighborhood parameter boxes — stay within
    the field ranges above. For PP (lensing potential) the output is the raw
    Capse convention, recorded in metadata. Output CSV: ell, Dl_muK2.
    """
    params = {"omega_b": omega_b, "omega_cdm": omega_cdm, "h": h,
              "tau": tau, "ns": ns, "ln10As": ln10As}
    ell, dl = (_capse_dl if backend == "capse" else _cosmopower_dl)(spectrum, params)
    keep = ell <= ell_max
    ell, dl = ell[keep], dl[keep]

    label = f"{spectrum} {backend}"
    outdir = resolve_outdir(output_dir)
    path = outdir / f"cmb_{spectrum.lower()}_{backend}.csv"
    unit = "muK^2 (Dl)" if spectrum != "PP" else "Capse native convention"
    write_csv(path, {"ell": ell, "Dl_muK2": dl},
              [f"label: {label}", f"spectrum: {spectrum}",
               f"backend: {backend}", f"unit: {unit}", f"params: {params}"])
    return ArtifactResult(
        status="success", files=[str(path)],
        message=f"Computed {spectrum} D_l with {backend} for l = 2..{int(ell[-1])}.",
        metadata={"spectrum": spectrum, "backend": backend, "params": params,
                  "unit": unit, "ell_range": [int(ell[0]), int(ell[-1])]},
    )


@validate_call
def plot_cmb_spectra(
    spectrum_files: Annotated[list[str], Field(min_length=1, max_length=6)],
    output_dir: Annotated[str, Field(min_length=1)],
    logx: bool = False,
) -> ArtifactResult:
    """Plot CMB Dl spectra from CSVs written by compute_cmb_cls."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..common import read_csv

    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = []
    for i, path_str in enumerate(spectrum_files):
        header, cols = read_csv(path_str)
        label = header.get("label", f"curve {i}")
        ax.plot(cols["ell"], cols["Dl_muK2"], color=f"C{i}", linewidth=1.6,
                label=label)
        labels.append(label)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell\ [\mu K^2]$")
    ax.set_title("CMB angular power spectra")
    ax.legend(fontsize="small")
    outdir = resolve_outdir(output_dir)
    path = outdir / "cmb_spectra.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ArtifactResult(
        status="success", files=[str(path)],
        message=f"Plotted {len(labels)} CMB spectra.",
        metadata={"labels": labels},
    )
