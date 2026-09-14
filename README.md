# cosmic-emulator-server

Production MCP server exposing public cosmological emulators as agent tools:
matter power spectra (6 cross-validating backends), modified gravity (f(R),
nDGP, cubic Galileon), CMB Cls (2 backends), EFTofLSS galaxy multipoles,
weak-lensing Cls, baryonic feedback (4 models), halo mass function, cluster
gas modeling, and Lyman-alpha P1D — 21 tools over ~20 emulators, all CPU,
millisecond-to-second evaluations.

Companion documents:
- [`EMULATOR_CATALOG.md`](EMULATOR_CATALOG.md) — survey of the public
  cosmological emulators this server draws from, with parameter ranges,
  unit conventions, licenses, and deployment notes
- [`ENVIRONMENT.md`](ENVIRONMENT.md) — how the single Python environment is
  assembled, incl. three vendored one-line patches to upstream codes

## Quick start

```bash
bash scripts/setup_env.sh                 # creates conda env "cosmic-emu"
conda activate cosmic-emu
python tests/smoke_env.py                 # 18 emulator evaluations
python -m pytest tests/ -q                # 24 tool integration tests
python tests/smoke_server.py              # every tool through a live MCP session
python -m mcp_server                      # stdio transport
python -m mcp_server --transport streamable-http --port 8000   # HTTP
```

Register with Claude Code (stdio, local):

```bash
claude mcp add cosmic-emu -- $(conda info --base)/envs/cosmic-emu/bin/python -m mcp_server
```

or point any MCP client at the HTTP endpoint (`http://host:8000/mcp`).

## Tool families

| Family | Tools | Backends |
|---|---|---|
| meta | list_emulators, describe_emulator, convert_cosmology, list_skills, load_skill | registry + cosmology-convention converter + skills |
| pk | compute_linear_pk, compute_nonlinear_pk, compose_spectra, plot_pk_comparison | camb, syren, baccoemu, euclidemu2, csst, gokunemu |
| gravity | compute_mg_boost, compute_mg_pk | e-MANTIS f(R), nDGPemu, CubicGalileonEmu |
| cmb | compute_cmb_cls, plot_cmb_spectra | capse (l<=5000), cosmopower-jax |
| lss | compute_galaxy_multipoles, compute_lensing_cls | PyBird one-loop EFT, jax-cosmo |
| baryons | compute_baryon_suppression, baryonify_pk, emulate_subgrid_statistic | SP(k), bacco, syren-baryon x4 suites, subgrid_emu |
| halos | compute_hmf, predict_cluster_gas_params | Mira-Titan HMF, picasso |
| igm | emulate_lya_p1d | LaCE (DESI) |

## Skills (server-side, client-agnostic)

The server carries its own skills — named multi-tool recipes in
[`skills/`](skills/) (markdown with a small frontmatter header). Any MCP
client can use them through two routes:

- **Tools**: `list_skills` returns a name + description index;
  `load_skill` returns the full instructions to follow. Works with every
  MCP client, including plain agent loops.
- **MCP prompts**: each skill is also registered as a native prompt, so
  prompt-capable clients surface the same recipes in their UI directly.

| Skill | What it does |
|---|---|
| `cosmo-tour` | one representative computation per family; the demo path |
| `pk-crosscheck` | all six nonlinear P(k) backends at one cosmology; quantifies the inter-emulator spread (the honest error bar) |
| `mg-explore` | modified-gravity boosts, parameter sweeps, composed P(k) vs the ΛCDM baseline |
| `baryon-budget` | baryonic-suppression envelope across models + CRK-HACC subgrid predictions with GP uncertainties |

The file format matches common client-side skill loaders (frontmatter
`name:`/`description:` + markdown body), so the same recipes can also be
vendored directly into an agent framework's own skills directory.

## Design principles

- **Discovery first**: agents call `list_emulators` / `describe_emulator` to
  learn capabilities and valid parameter boxes before computing.
- **Composition**: boost-type emulators (MG, baryons, EE2) are exposed both
  raw and composed with a named baseline; metadata always records which.
- **Units normalized at the tool layer**: k [h/Mpc], P [(Mpc/h)^3], CMB Dl
  [muK^2] — with the two deliberate exceptions (Lya in comoving Mpc; Capse
  PP native) flagged in tool schemas.
- **Uncertainty surfaced**: GP emulators (cubic Galileon, subgrid, HMF)
  return their std as a CSV column, not just the mean.
- **Files, not arrays**: results flow between tools as CSV paths; only
  paths and summaries pass through the LLM context.
- **Range validation, two layers**: pydantic schemas take the union of
  backend ranges (so backends stay comparable); every response then
  reports `in_training_box` and per-backend extrapolation warnings.
- **Numbers retrievable inline**: every compute tool takes
  `return_data=true` (downsampled arrays in metadata) and always returns
  quotable summary stats — for clients that cannot fetch the artifact URL.
- **Server-side composition**: `compose_spectra` and `baryonify_pk` do the
  arithmetic between outputs with provenance; clients never multiply CSVs.

## Hosting

The server is designed for a plain Linux box behind a reverse proxy:

1. Clone the repo and run `scripts/setup_env.sh`.
2. Run `python tests/smoke_env.py --all` once to pre-download the deferred
   emulators' model files (~GB total) before first request.
3. Run under systemd with the HTTP transport bound to localhost and these
   environment variables (also allow writes to the service HOME's `.cache`
   — some libraries write runtime caches there):
   - `MCP_PUBLIC=1` — disable DNS-rebinding protection (only behind a
     proxy/auth)
   - `MCP_OUTPUT_ROOT=/srv/artifacts` — every tool `output_dir` is remapped
     under this directory
   - `MCP_ARTIFACT_URL=https://files.example.org` — tool results then carry
     browsable URLs for every file written
4. Put a TLS reverse proxy (e.g. Caddy) in front: one route to the server
   port, one static file server on `MCP_OUTPUT_ROOT`.
5. After every deploy, run `python tests/smoke_server.py <url>` — it calls
   every tool through the served path (transport + sandbox + filesystem),
   which catches failures the in-process tests cannot.

## Run on HPC (optional)

The P(k) backends can execute on DOE facilities (ALCF Polaris, NERSC
Perlmutter) through the
[hep-genesis](https://github.com/HEP-KE/hep-genesis-agent) dispatch engine.
Local execution is the default and needs none of this. Two modes, split by
where the facility credentials live:

**Client-side dispatch (hosted deployments — recommended).** The server
never holds tokens, Globus endpoints, or facility config; it only hands its
kernels over. The `export_dispatch_pack` tool returns the `tools/` package
(text files + a manifest: kernel entry point, per-backend node pip deps
from `DISPATCH_PIP_DEPS`, walltime hints). The CLIENT — the user's
hep-genesis harness, wherever it runs — saves the pack and dispatches it
under its own credentials via its facility servers' `run_pack_kernel` tool
(or the hep-genesis sidecar's `POST /jobs` with `pack=`); the P(k) list
comes back client-side in `results.json`. Deploying this server to a VM
therefore requires NOTHING beyond the server itself: no hep-genesis
install, no facility sign-in, no Globus.

**Server-side dispatch (running the server on your own machine).**

```bash
pip install -e <hep-genesis-agent>/backend[iri]   # into this server's env
```

Then, in a session: `set_dispatch("polaris")` (or `"perlmutter"`) makes each
`compute_linear_pk` / `compute_nonlinear_pk` call one facility job (the
`tools/` package is staged, the backend runs on a compute node with its pip
deps installed per `DISPATCH_PIP_DEPS`, the P(k) array comes back and the
CSV is still written by this server, so downstream tools work unchanged).
`get_dispatch` reports the site; `auth_status` reports facility sign-in
(hep-genesis auth CLIs or desktop app). ALCF needs Globus Connect Personal
on this host; NERSC is Globus-free (pure IRI). `set_dispatch("local")`
switches back. Do NOT use this mode on shared/hosted deployments — one
identity's credentials and allocation would serve every client.

In both modes the `csst` backend stays local-only (it needs the vendored
patches in `external/`, which pip cannot reproduce on a compute node), and
all other tool families (gravity, CMB, LSS, baryons, halos, IGM) currently
run locally regardless of the dispatch setting.
