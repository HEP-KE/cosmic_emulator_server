"""Optional HPC dispatch: run this server's compute kernels on DOE facilities.

Local execution is the default and is untouched by this module. Two modes:

CLIENT-SIDE (hosted deployments — recommended): this server holds NO
credentials. export_dispatch_pack hands the tools/ kernels to the client,
and the client's hep-genesis harness — which owns the facility tokens,
Globus endpoint, project and workdir — stages and submits them itself
(facility-server run_pack_kernel tool / sidecar POST /jobs pack=). A VM
deployment needs nothing beyond this server.

SERVER-SIDE (running the server on your own machine): calling
set_dispatch("polaris"|"perlmutter") routes the compute-heavy tools through
the hep-genesis dispatch engine in THIS process: the tools/ package is
staged to the facility, the kernel runs on a compute node via IRI, and
results come back to this machine. Requirements on this host:
- the hep-genesis backend importable in this environment
  (pip install -e <hep-genesis-agent>/backend[iri])
- facility sign-in (hep-genesis-alcf-auth / -transfer-auth CLIs, or the
  desktop app's HPC panel) — check with the auth_status tool
- a Globus endpoint (Globus Connect Personal) for ALCF staging/fetch-back.
  NERSC needs NO Globus endpoint: with no GCP on this host (or with
  DISPATCH_STAGING=iri) the engine stages up and fetches results back over
  the IRI filesystem API.
Do not use server-side mode on shared/hosted deployments — one identity's
credentials and allocation would serve every client.

State is process-local and read at call time, so an agent can flip sites
per message. Nothing here imports hep_genesis until a remote site is chosen.
"""

import os
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"

_state = {"site": "local"}

__all__ = ["set_dispatch", "get_dispatch", "auth_status", "export_dispatch_pack"]

# Mirror the hep-genesis engine's pack limits so an exported pack is always
# stageable by the client that receives it.
_PACK_MAX_FILES = 200
_PACK_MAX_BYTES = 10 * 1024 * 1024


def export_dispatch_pack() -> dict:
    """Hand this server's compute kernels to the CLIENT for HPC dispatch.

    Returns the ``tools/`` kernels package as in-band text files plus a
    manifest (kernel entry points, per-backend pip requirements for the
    compute node, walltime hints). The CLIENT — which holds the facility
    tokens, Globus endpoint, project and workdir — stages and submits the
    kernels with ITS OWN credentials (e.g. the hep-genesis facility servers'
    run_pack_kernel tool). Nothing credential-shaped is needed on, or
    returned by, this server.

    Clients that recognise the ``dispatch_pack`` key should save the files
    locally and NOT echo their contents into the conversation.
    """
    files: dict[str, str] = {}
    total = 0
    for path in sorted(TOOLS_DIR.rglob("*.py")):
        rel = path.relative_to(TOOLS_DIR)
        if any(part.startswith((".", "__pycache__")) for part in rel.parts):
            continue
        text = path.read_text()
        total += len(text.encode())
        if len(files) >= _PACK_MAX_FILES or total > _PACK_MAX_BYTES:
            raise RuntimeError(
                f"tools/ package exceeds pack limits ({_PACK_MAX_FILES} files / "
                f"{_PACK_MAX_BYTES} bytes) — prune before exporting."
            )
        files[f"tools/{rel}"] = text

    # Node-side pip requirements live WITH the kernels (single source of truth).
    try:
        from tools.pk.backends import DISPATCH_PIP_DEPS
        pip_deps = {b: list(d) for b, d in DISPATCH_PIP_DEPS.items()}
    except Exception:  # noqa: BLE001 — manifest stays useful without it
        pip_deps = {}

    return {
        "dispatch_pack": {
            "name": "tools",
            "server": "cosmic-emulator-server",
            "file_count": len(files),
            "bytes": total,
            "files": files,
            "kernels": {
                "pk": {
                    "function": "pk.backends.compute_pk",
                    "args": {"kind": "linear|nonlinear", "backend": "<name>",
                             "params": "{Om, Ob, h, ns, As|sigma8, mnu, w0, wa}",
                             "k": "[list of k in h/Mpc]", "z": "float"},
                    "pip_deps_by_backend": pip_deps,
                    "base_pip_deps": ["numpy", "pydantic"],
                    "duration_hint_s": {"default": 600, "gokunemu": 1200},
                    "returns": "P(k) as a list, (Mpc/h)^3",
                },
            },
            "usage": (
                "Save these files on the CLIENT machine, then dispatch with the "
                "hep-genesis facility server: run_pack_kernel(pack=<saved dir>/tools, "
                "function='pk.backends.compute_pk', args={...}, "
                "pip_deps=base_pip_deps + pip_deps_by_backend[backend])."
            ),
        }
    }


def _engine():
    """Import the dispatch engine, with install instructions on failure."""
    try:
        from hep_genesis.iri.alcf.dispatch import run_codes_on_polaris
        from hep_genesis.iri.nersc.dispatch import run_codes_on_perlmutter
    except ImportError as exc:
        raise RuntimeError(
            "HPC dispatch needs the hep-genesis backend in this server's "
            "environment. Install it with: pip install -e "
            "<hep-genesis-agent>/backend[iri]  (import failed: " + str(exc) + ")"
        ) from exc
    return {"polaris": run_codes_on_polaris, "perlmutter": run_codes_on_perlmutter}


def remote_site() -> str | None:
    """The active remote site ('polaris'/'perlmutter'), or None for local.

    Not an MCP tool — compute tools call this to branch at call time.
    """
    site = _state["site"]
    return None if site == "local" else site


def run_kernel(function: str, args: dict, pip_deps: list[str] | None = None,
               duration: int = 600) -> dict:
    """Run a kernel from this server's tools/ package on the active site.

    Not an MCP tool. ``function`` is a dotted path within tools/ (e.g.
    "pk.backends.compute_pk"); ``args`` must be JSON-safe. Returns the
    engine dict: result, host, and artifact_files (local paths of files the
    kernel wrote in its job directory, fetched back automatically).

    Raises RuntimeError with remediation text on any dispatch failure —
    callers should surface it, and agents must NOT blind-retry.
    """
    site = _state["site"]
    if site == "local":
        raise RuntimeError("run_kernel called with local dispatch — use the kernel directly.")
    run = _engine()[site]
    try:
        result = run(function=function, args=args, codes=str(TOOLS_DIR),
                     pip_deps=pip_deps, duration=duration)
    except Exception as exc:
        body = ""
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                body = (resp.text or "")[:600]
            except Exception:
                pass
        from hep_genesis.iri.dispatch.hints import dispatch_auth_hint
        hint = dispatch_auth_hint(site, body or str(exc))
        raise RuntimeError(
            f"Remote dispatch to {site} failed: {exc}\n"
            f"{('Response body: ' + body) if body else ''}{hint}\n"
            "Do NOT retry this call — the failure is in the dispatch layer "
            "and will recur. Surface this error to the user."
        ) from exc
    if result.get("status") != "success":
        raise RuntimeError(
            f"Remote job on {site} failed: {result.get('error', 'unknown error')}\n"
            f"{('Stderr: ' + result['stderr']) if result.get('stderr') else ''}\n"
            "Do NOT retry this call — surface this error to the user."
        )
    return result


def set_dispatch(site: str, artifact_dir: str | None = None) -> str:
    """Set where compute-heavy tools execute: 'local' (this machine),
    'polaris' (ALCF) or 'perlmutter' (NERSC).

    Remote sites run each compute call as one facility job (staging + queue +
    walltime: minutes, not seconds) and need facility sign-in — check with
    auth_status. Light tools (plots, filters on existing files) always run
    locally. Optional artifact_dir sets where produced files are fetched back
    to (e.g. a research project's jobs/ directory). Returns the active
    configuration.
    """
    if artifact_dir:
        # Explicit set beats the setdefault below, so a project-scoped root
        # sticks for this server process until replaced.
        os.environ["DISPATCH_ARTIFACT_DIR"] = str(Path(artifact_dir).expanduser())
    site = (site or "").strip().lower()
    if site in ("local", "off", "none", ""):
        _state["site"] = "local"
        return "Dispatch: local execution."
    if site in ("polaris", "alcf"):
        site = "polaris"
    elif site in ("perlmutter", "nersc"):
        site = "perlmutter"
    else:
        return f"Unknown site {site!r}. Use 'local', 'polaris', or 'perlmutter'."
    # Import the engine NOW: a missing install fails here with instructions,
    # and hep_genesis's .env load (override=True at import) happens before we
    # touch the environment below, so it cannot clobber what we set.
    _engine()
    # Transfer-token selection inside hep_genesis.iri reads DISPATCH_TARGET.
    os.environ["DISPATCH_TARGET"] = site
    # Fetch produced files back to a GCP-visible folder (never a dot-folder).
    os.environ.setdefault(
        "DISPATCH_ARTIFACT_DIR", str(Path.home() / "hep-genesis" / "artifacts")
    )
    _state["site"] = site
    facility = "ALCF" if site == "polaris" else "NERSC"
    return (
        f"Dispatch: remote on {site} ({facility}) — compute-heavy tools will "
        "stage this server's kernels, submit via IRI, and fetch results back. "
        "Each call is one facility job (expect minutes)."
    )


def get_dispatch() -> str:
    """Report where compute-heavy tools currently execute."""
    site = _state["site"]
    if site == "local":
        return "Dispatch: local execution."
    return f"Dispatch: remote on {site}."


def auth_status() -> str:
    """Report facility sign-in state for HPC dispatch (ALCF and NERSC).

    Call this before dispatching remotely, or when a remote call fails with
    an auth error. Sign-in happens outside this server (hep-genesis auth
    CLIs or the desktop app's HPC panel); tokens are re-read on every call,
    so a fresh sign-in is picked up without restarting this server.
    """
    try:
        from hep_genesis.iri.alcf.client import (
            _live_iri_token as alcf_iri, _live_transfer_token as alcf_tx,
        )
        from hep_genesis.iri.nersc.client import (
            _live_iri_token as nersc_iri, _live_transfer_token as nersc_tx,
        )
    except ImportError:
        return (
            "hep-genesis backend not installed in this server's environment — "
            "remote dispatch unavailable. Install: pip install -e "
            "<hep-genesis-agent>/backend[iri]"
        )
    lines = []
    for name, iri, tx in (("ALCF", alcf_iri, alcf_tx), ("NERSC", nersc_iri, nersc_tx)):
        lines.append(
            f"{name}: IRI token {'available' if iri() else 'MISSING'}, "
            f"transfer token {'available' if tx() else 'MISSING'}"
        )
    lines.append(f"{get_dispatch()} Sign in via the hep-genesis auth CLIs or the app's HPC panel.")
    return "\n".join(lines)
