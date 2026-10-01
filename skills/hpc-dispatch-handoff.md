---
name: hpc-dispatch-handoff
description: Run this server's kernels (P(k), HMF) on ALCF/NERSC when the server itself cannot dispatch — export the pack, let the CLIENT's hep-genesis facility tools run it under the user's credentials
---

# HPC dispatch handoff (client-side execution)

Use this when the user wants a computation from THIS server executed on a
DOE facility and `auth_status` here reports the dispatch backend is not
installed. That is normal for hosted deployments: this server holds no
facility tokens by design. The kernels travel to the client; the client's
hep-genesis facility servers (which hold the user's tokens, project, and
workdir) run them.

1. Call `export_dispatch_pack` (this server). The client saves the kernel
   files locally and shows you `pack_path` plus a `kernels` manifest.
2. On the CLIENT's facility server (hep-genesis-alcf / hep-genesis-nersc),
   call `run_pack_kernel` once per run:
   - `pack` = the exact `pack_path` (ends in `/tools`) — never a run/output
     directory.
   - `function`, `args` (JSON object string), and `pip_deps`
     (`base_pip_deps` + the chosen backend's `pip_deps_by_backend` entry)
     all come from the manifest.
   - `artifact_dir` = the client run directory; the returned arrays are
     saved there as `<function>_result.json/.csv` next to the job logs.
3. Verify the result's `host` field names a compute node before telling the
   user it ran on the facility.

Kernels currently in the pack: `pk.backends.compute_pk` (all pip-clean P(k)
backends) and `halos.kernels.compute_hmf` (miratitan via
MiraTitanHMFemulator; tinker08 / sheth_tormen / press_schechter via
colossus). The `csst` P(k) backend is local-only (needs vendored patches).

Plotting facility results: this server cannot read the client's files, so
never accept client-side CSV paths into `plot_emulator_curves`. Because the
emulator kernels are seeded and deterministic, recomputing the same call
LOCALLY here reproduces the facility numbers exactly — do that to make the
CSVs/figure, and state clearly that the figure was rendered from an
identical local recompute of the facility-verified result.

`set_dispatch` on this server is only for deployments running on the user's
own machine with the hep-genesis backend installed; it never affects, and is
never affected by, other servers' dispatch state.
