# cosmic-emulator-server

Production MCP server exposing public cosmological emulators as agent tools:
matter power spectra (6 cross-validating backends), modified gravity (f(R),
nDGP, cubic Galileon), CMB Cls (2 backends), EFTofLSS galaxy multipoles,
weak-lensing Cls, baryonic feedback (4 models), halo mass function, cluster
gas modeling, and Lyman-alpha P1D — 16 tools over ~20 emulators, all CPU,
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
python -m pytest tests/ -q                # 15 tool integration tests
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
| meta | list_emulators, describe_emulator | registry with ranges/units/citations |
| pk | compute_linear_pk, compute_nonlinear_pk, plot_pk_comparison | camb, syren, baccoemu, euclidemu2, csst, gokunemu |
| gravity | compute_mg_boost, compute_mg_pk | e-MANTIS f(R), nDGPemu, CubicGalileonEmu |
| cmb | compute_cmb_cls, plot_cmb_spectra | capse (l<=5000), cosmopower-jax |
| lss | compute_galaxy_multipoles, compute_lensing_cls | PyBird one-loop EFT, jax-cosmo |
| baryons | compute_baryon_suppression, emulate_subgrid_statistic | SP(k), bacco, syren-baryon x4 suites, subgrid_emu |
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
- **Range validation**: pydantic field bounds mirror each emulator's
  training box; out-of-range errors name the violated range.

## Hosting

The server is designed for a plain Linux box behind a reverse proxy:

1. Clone the repo and run `scripts/setup_env.sh`.
2. Run `python tests/smoke_env.py --all` once to pre-download the deferred
   emulators' model files (~GB total) before first request.
3. Run under systemd with the HTTP transport bound to localhost and these
   environment variables:
   - `MCP_PUBLIC=1` — disable DNS-rebinding protection (only behind a
     proxy/auth)
   - `MCP_OUTPUT_ROOT=/srv/artifacts` — every tool `output_dir` is remapped
     under this directory
   - `MCP_ARTIFACT_URL=https://files.example.org` — tool results then carry
     browsable URLs for every file written
4. Put a TLS reverse proxy (e.g. Caddy) in front: one route to the server
   port, one static file server on `MCP_OUTPUT_ROOT`.
