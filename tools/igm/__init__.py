"""Intergalactic-medium tools: Lyman-alpha forest 1D flux power spectrum."""

import numpy as np
from typing import Annotated

from pydantic import Field, validate_call

from ..common import (ArtifactResult, downsample_columns, get_cached,
                      param_slug, quiet, resolve_outdir, summary_stats,
                      write_csv)

__all__ = ["emulate_lya_p1d"]


@validate_call
def emulate_lya_p1d(
    output_dir: Annotated[str, Field(min_length=1)],
    Delta2_p: Annotated[float, Field(ge=0.2, le=0.6, description="Amplitude of the linear power at the pivot k_p = 0.7 1/Mpc")] = 0.35,
    n_p: Annotated[float, Field(ge=-2.4, le=-2.2, description="Slope of the linear power at the pivot")] = -2.3,
    mF: Annotated[float, Field(ge=0.5, le=0.9, description="Mean transmitted flux fraction (encodes redshift)")] = 0.66,
    sigT_Mpc: Annotated[float, Field(ge=0.09, le=0.17, description="Thermal broadening scale in comoving Mpc")] = 0.13,
    gamma: Annotated[float, Field(ge=1.0, le=1.9, description="Slope of the temperature-density relation")] = 1.5,
    kF_Mpc: Annotated[float, Field(ge=9.0, le=15.0, description="Pressure smoothing scale in 1/Mpc")] = 10.5,
    k_min_Mpc: Annotated[float, Field(ge=0.05)] = 0.1,
    k_max_Mpc: Annotated[float, Field(le=4.0)] = 3.0,
    n_points: Annotated[int, Field(ge=10, le=500)] = 100,
    return_data: Annotated[bool, Field(description="Include downsampled arrays in metadata.data.")] = False,
) -> ArtifactResult:
    """Emulate the Lyman-alpha forest 1D flux power spectrum P1D(k_par) with LaCE.

    LaCE (DESI/igmhub) works in a 6-parameter IGM/cosmology compression:
    (Delta2_p, n_p) describe the linear power at k_p = 0.7 1/Mpc and
    (mF, sigT_Mpc, gamma, kF_Mpc) the IGM state — redshift enters only
    through these, so there is no explicit z argument. UNITS ARE COMOVING
    Mpc (no h!): k_par in 1/Mpc, P1D in Mpc. Output CSV: k_par_Mpc,
    P1D_Mpc, and the dimensionless k*P/pi.
    """
    from lace.emulator.gp_emulator import GPEmulator

    emu = get_cached(
        "lace_gp",
        lambda: GPEmulator(training_set="Pedersen21", emulator_label="Pedersen21"))
    model = {"Delta2_p": Delta2_p, "n_p": n_p, "mF": mF,
             "sigT_Mpc": sigT_Mpc, "gamma": gamma, "kF_Mpc": kF_Mpc}
    k = np.linspace(k_min_Mpc, k_max_Mpc, n_points)
    with quiet():
        p1d = np.ravel(np.asarray(emu.emulate_p1d_Mpc(model, k), dtype=float))

    label = f"LaCE P1D (mF={mF}, Delta2_p={Delta2_p})"
    outdir = resolve_outdir(output_dir)
    path = outdir / f"lya_p1d_{param_slug(model)}.csv"
    columns = {"k_par_1_per_Mpc": k, "P1D_Mpc": p1d,
               "k_P_over_pi": k * p1d / np.pi}
    write_csv(path, columns,
              [f"label: {label}", "quantity: p1d",
               "units: k_par [1/Mpc comoving, no h], P1D [Mpc]",
               f"igm_params: {model}"])
    metadata = {"igm_params": model,
                "units": {"k_par": "1/Mpc (comoving, no h)", "P1D": "Mpc"},
                "emulator": "LaCE GP, Pedersen21 training set",
                "stats": summary_stats(k, p1d, "k_par", "P1D")}
    if return_data:
        metadata["data"] = downsample_columns(columns)
    return ArtifactResult(
        status="success", files=[str(path)],
        message=f"Emulated P1D for {n_points} k bins "
                f"{k_min_Mpc}..{k_max_Mpc} 1/Mpc.",
        metadata=metadata,
    )
