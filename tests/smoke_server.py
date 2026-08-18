"""Live-server smoke test: call every active tool through a real MCP session.

Unlike smoke_env.py (which validates the Python environment) and
test_tools.py (which calls tool functions in-process), this exercises the
SERVED path — transport, schema, sandboxing, filesystem permissions. Run it
after every deployment; a tool that works in-process can still fail under
the systemd sandbox (e.g. a library writing caches to a read-only $HOME).

Usage:
    python tests/smoke_server.py                          # spawns stdio server
    python tests/smoke_server.py http://127.0.0.1:8002/mcp
    python tests/smoke_server.py https://cosmic.example.org/mcp

Exit code 0 only if every tool call (including the chained ones) succeeds.
"""

import json
import sys

import anyio
from mcp.client.session import ClientSession

OUT = "/tmp/smoke_server" if len(sys.argv) < 2 else "/srv/artifacts/smoke-server"

RESULTS = []


async def call(s, name, args, expect_error=False):
    try:
        res = await s.call_tool(name, args)
        if res.isError:
            text = res.content[0].text if res.content else ""
            if expect_error:
                RESULTS.append((name, "OK", "expected error raised"))
                return None
            RESULTS.append((name, "FAIL", text[:140]))
            return None
        payload = json.loads(res.content[0].text)
        if expect_error:
            RESULTS.append((name, "FAIL", "expected an error, got success"))
            return payload
        RESULTS.append((name, "OK", payload.get("message", "")[:110]))
        return payload
    except Exception as e:  # transport-level failure
        RESULTS.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:120]}"))
        return None


async def run(session: ClientSession):
    await session.initialize()
    tools = await session.list_tools()
    prompts = await session.list_prompts()
    print(f"server: {len(tools.tools)} tools, {len(prompts.prompts)} prompts")

    s = session
    # --- discovery
    await call(s, "list_emulators", {})
    await call(s, "describe_emulator", {"name": "gokunemu"})
    await call(s, "list_skills", {})
    await call(s, "load_skill", {"name": "baryon-budget"})
    await call(s, "convert_cosmology", {"sigma8": 0.81})

    # --- pk family + chaining
    lin = await call(s, "compute_linear_pk",
                     {"output_dir": OUT, "backend": "camb", "z": 0.5,
                      "k_min": 1e-4, "k_max": 1.0, "n_points": 300})
    nl1 = await call(s, "compute_nonlinear_pk",
                     {"output_dir": OUT, "backend": "baccoemu",
                      "return_data": True})
    nl2 = await call(s, "compute_nonlinear_pk",
                     {"output_dir": OUT, "backend": "csst"})
    if nl1 and nl2:
        await call(s, "plot_pk_comparison",
                   {"spectrum_files": [nl1["files"][0], nl2["files"][0]],
                    "output_dir": OUT})

    # --- gravity
    await call(s, "compute_mg_boost",
               {"output_dir": OUT, "model": "cubic_galileon", "f_phi": 0.8})
    mg = await call(s, "compute_mg_pk",
                    {"output_dir": OUT, "model": "fofr", "k_max": 2.0})

    # --- cmb
    c1 = await call(s, "compute_cmb_cls",
                    {"output_dir": OUT, "spectrum": "TT", "backend": "capse"})
    if c1:
        await call(s, "plot_cmb_spectra",
                   {"spectrum_files": [c1["files"][0]], "output_dir": OUT})

    # --- lss (the chain that once broke under the sandbox)
    if lin:
        await call(s, "compute_galaxy_multipoles",
                   {"linear_pk_file": lin["files"][0], "output_dir": OUT,
                    "z": 0.5})
    await call(s, "compute_lensing_cls", {"output_dir": OUT, "n_ell": 15})

    # --- baryons, incl. server-side composition
    sup = await call(s, "compute_baryon_suppression",
                     {"output_dir": OUT, "model": "syren_IllustrisTNG"})
    if nl1:
        await call(s, "baryonify_pk",
                   {"pk_file": nl1["files"][0], "output_dir": OUT,
                    "model": "spk"})
    if nl1 and sup:
        await call(s, "compose_spectra",
                   {"spectrum_files": [nl1["files"][0], sup["files"][0]],
                    "output_dir": OUT, "op": "multiply"})
    await call(s, "emulate_subgrid_statistic",
               {"output_dir": OUT, "statistic": "Pk"})

    # --- halos + igm (theory backend included: colossus must not need
    # writable caches under the service sandbox)
    await call(s, "compute_hmf", {"output_dir": OUT, "return_data": True})
    await call(s, "compute_hmf", {"output_dir": OUT, "backend": "tinker08",
                                  "mass_def": "200c"})
    await call(s, "predict_cluster_gas_params", {"output_dir": OUT})
    await call(s, "emulate_lya_p1d", {"output_dir": OUT})

    # --- guardrails must guard
    if nl1 and sup:
        await call(s, "plot_pk_comparison",
                   {"spectrum_files": [nl1["files"][0], sup["files"][0]],
                    "output_dir": OUT}, expect_error=True)


async def main():
    if len(sys.argv) > 1:
        from mcp.client.streamable_http import streamablehttp_client
        async with streamablehttp_client(sys.argv[1]) as (r, w, _):
            async with ClientSession(r, w) as session:
                await run(session)
    else:
        from mcp.client.stdio import StdioServerParameters, stdio_client
        params = StdioServerParameters(command=sys.executable,
                                       args=["-m", "mcp_server"])
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as session:
                await run(session)


if __name__ == "__main__":
    anyio.run(main)
    width = max(len(n) for n, *_ in RESULTS) + 2
    fails = 0
    for name, status, detail in RESULTS:
        print(f"{status:6s}{name:{width}s}{detail}")
        fails += status == "FAIL"
    print(f"\n{len(RESULTS)} calls: {len(RESULTS) - fails} OK, {fails} FAIL")
    sys.exit(1 if fails else 0)
