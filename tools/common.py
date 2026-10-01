"""Shared infrastructure for all emulator tool modules.

Conventions enforced here and documented in every tool schema:
- wavenumbers k in h/Mpc, power spectra in (Mpc/h)^3
- CMB spectra returned as Dl = l(l+1)Cl/2pi in muK^2
- data flows between tools as CSV file paths, never raw arrays
"""

import contextlib
import io
import threading
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from pydantic import BaseModel

# Some upstream emulators (pybird 0.3.x) still call np.trapz, removed in
# numpy 2. Restore the alias before any of them import.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

T_CMB_UK = 2.7255e6  # CMB monopole temperature in microkelvin


class ArtifactResult(BaseModel):
    """Uniform result contract returned by every tool."""

    status: Literal["success"]
    files: list[str]
    message: str
    metadata: dict[str, Any]


_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


def get_cached(key: str, factory: Callable[[], Any]) -> Any:
    """Load-once cache for emulator objects (model files, GP fits, JIT warmup).

    Emulator construction can take seconds and download data; every tool goes
    through here so repeated calls are milliseconds.
    """
    with _CACHE_LOCK:
        if key not in _CACHE:
            with contextlib.redirect_stdout(io.StringIO()):
                _CACHE[key] = factory()
        return _CACHE[key]


@contextlib.contextmanager
def quiet():
    """Silence emulators that print progress to stdout (SEPIA, jaxcapse, ...)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def resolve_outdir(output_dir: str) -> Path:
    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def param_slug(params: dict) -> str:
    """Short stable hash of parameter values, used to make filenames unique.

    Prevents parameter scans from silently overwriting each other's outputs
    (every call with different inputs gets a different file name; identical
    calls reuse the same name, which is idempotent).
    """
    import hashlib
    blob = ",".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.sha1(blob.encode()).hexdigest()[:6]


def varied_label(base: str, params: dict, defaults: dict) -> str:
    """Label = base + the parameters that differ from the tool's defaults.

    Three HMF runs at different sigma_8 must not all be labeled
    'Mira-Titan HMF z=0'; this appends e.g. ' [sigma_8=0.75, w_0=-0.8]'.
    """
    diffs = [f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
             for k, v in params.items()
             if k in defaults and v != defaults[k]]
    return f"{base} [{', '.join(diffs)}]" if diffs else base


def downsample_columns(columns: dict, n: int = 80) -> dict:
    """Downsample all columns to <= n points (same indices for every column)."""
    length = len(next(iter(columns.values())))
    if length <= n:
        idx = np.arange(length)
    else:
        idx = np.unique(np.linspace(0, length - 1, n).astype(int))
    return {k: np.round(np.asarray(v, dtype=float).ravel()[idx], 8).tolist()
            for k, v in columns.items()}


def summary_stats(x: np.ndarray, y: np.ndarray, x_name: str, y_name: str) -> dict:
    """Always-on quotable numbers: extrema, median, and a few sampled points."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    finite = np.isfinite(y)
    if not finite.any():
        return {"note": "no finite values"}
    xs, ys = x[finite], y[finite]
    i_min, i_max = int(np.argmin(ys)), int(np.argmax(ys))
    probes = np.unique(np.linspace(0, len(xs) - 1, 5).astype(int))
    return {
        f"min_{y_name}": float(ys[i_min]), f"min_at_{x_name}": float(xs[i_min]),
        f"max_{y_name}": float(ys[i_max]), f"max_at_{x_name}": float(xs[i_max]),
        f"median_{y_name}": float(np.median(ys)),
        "samples": {f"{x_name}={xs[i]:.4g}": float(ys[i]) for i in probes},
    }


def write_csv(path: Path, columns: dict[str, np.ndarray], header_lines: list[str]) -> None:
    """Write named columns with '# key: value' header comments."""
    arrays = [np.asarray(v, dtype=float).ravel() for v in columns.values()]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("write_csv: all columns must have equal length")
    with path.open("w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        f.write(",".join(columns.keys()) + "\n")
        for row in zip(*arrays):
            f.write(",".join(f"{x:.8g}" for x in row) + "\n")


def read_csv(path_str: str) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    """Read a CSV written by write_csv: returns (header dict, column dict)."""
    path = Path(path_str).expanduser().resolve()
    header: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            break
        key, _, value = line.lstrip("# ").partition(":")
        header[key.strip()] = value.strip()
    names = lines[i].split(",")
    data = np.loadtxt(lines[i + 1:], delimiter=",", ndmin=2)
    return header, {name: data[:, j] for j, name in enumerate(names)}


def k_grid(k_min: float, k_max: float, n_points: int) -> np.ndarray:
    return np.logspace(np.log10(k_min), np.log10(k_max), n_points)


def unity_crossings(x: np.ndarray, y: np.ndarray, max_points: int = 6) -> list[float]:
    """x-values where a dimensionless ratio crosses 1 (linear interpolation).

    For B(k), S(k) and composed B*S products these are the scales where the
    effects cancel / switch sign — quoted directly so nobody has to load the
    CSV just to find them.
    """
    x = np.asarray(x, dtype=float).ravel()
    d = np.asarray(y, dtype=float).ravel() - 1.0
    ok = np.isfinite(d)
    x, d = x[ok], d[ok]
    idx = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    out = [float(x[i] - d[i] * (x[i + 1] - x[i]) / (d[i + 1] - d[i])) for i in idx]
    return [round(v, 5) for v in out[:max_points]]
