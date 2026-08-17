import functools
import importlib
import inspect
import os
from pathlib import Path
import tomllib
from typing import Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

Transport = Literal["stdio", "streamable-http"]

# Hosted deployments set these (same convention as spectra/gaia servers):
#   MCP_OUTPUT_ROOT   e.g. /srv/artifacts — every output_dir an agent passes
#                     is remapped under this directory
#   MCP_ARTIFACT_URL  e.g. https://files.example.org — returned messages then
#                     include a browsable URL for each file written there
OUTPUT_ROOT = os.environ.get("MCP_OUTPUT_ROOT")
ARTIFACT_URL = (os.environ.get("MCP_ARTIFACT_URL") or "").rstrip("/")


def publish_outputs(func):
    """Confine a tool's output_dir to OUTPUT_ROOT and add URLs to its result."""
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        requested = bound.arguments.get("output_dir")
        if requested is not None:
            root = Path(OUTPUT_ROOT)
            path = Path(str(requested))
            if not path.is_relative_to(root):
                safe = "-".join(part for part in path.parts if part != "/")
                bound.arguments["output_dir"] = str(root / (safe or "output"))
        result = func(*bound.args, **bound.kwargs)
        if ARTIFACT_URL and getattr(result, "files", None):
            urls = [f.replace(OUTPUT_ROOT, ARTIFACT_URL, 1)
                    for f in result.files if f.startswith(OUTPUT_ROOT)]
            if urls:
                result.message += " View: " + "  ".join(urls)
        return result

    return wrapper


def pyproject_toml() -> Path:
    for directory in Path(__file__).resolve().parents:
        path = directory / "pyproject.toml"
        if path.exists():
            return path
    raise FileNotFoundError("Could not find pyproject.toml")


def configured_tool_module_names() -> list[str]:
    path = pyproject_toml()
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        return list(config["tool"]["mcp-server"]["tool_modules"])
    except KeyError as exc:
        raise RuntimeError(
            f"{path} must contain a [tool.mcp-server] section with a "
            "tool_modules list."
        ) from exc


def load_tool_modules():
    return [importlib.import_module(name)
            for name in configured_tool_module_names()]


def create_server(*, host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    transport_security = None
    if os.environ.get("MCP_PUBLIC"):
        transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False)

    instructions = (
        "Production cosmological-emulator server: matter power spectra "
        "(6 backends), modified gravity (f(R), nDGP, cubic Galileon), CMB "
        "Cls, EFTofLSS galaxy multipoles, weak-lensing Cls, baryonic "
        "feedback, halo mass function, cluster gas, and Lyman-alpha P1D. "
        "Start with list_emulators to discover capabilities and "
        "describe_emulator for valid parameter ranges — every emulator has "
        "a hard training box. The server also carries skills (named "
        "multi-tool recipes): call list_skills, and when a task matches "
        "one, load_skill and follow it. Units: k in h/Mpc and P(k) in (Mpc/h)^3 "
        "everywhere except Lyman-alpha tools (comoving Mpc, no h); CMB "
        "spectra are Dl in muK^2. Tools that write files accept an "
        "output_dir argument and return structured artifact metadata; pass "
        "file paths between tools, never raw arrays."
    )
    if OUTPUT_ROOT:
        instructions += (
            f" This is a hosted server: all output files are stored under "
            f"{OUTPUT_ROOT} on the server (any other output_dir is remapped "
            "there), and results include browsable URLs — share those URLs "
            "with the user instead of trying to read or recreate the files."
        )

    mcp = FastMCP(
        "Cosmic Emulator MCP Server",
        instructions=instructions,
        host=host,
        port=port,
        transport_security=transport_security,
    )

    for tool_module in load_tool_modules():
        if not hasattr(tool_module, "__all__"):
            raise RuntimeError(
                f"Tool module '{tool_module.__name__}' must define __all__.")
        for name in tool_module.__all__:
            tool_function = getattr(tool_module, name)
            if OUTPUT_ROOT:
                tool_function = publish_outputs(tool_function)
            mcp.tool()(tool_function)

    register_skill_prompts(mcp)
    return mcp


def register_skill_prompts(mcp: FastMCP) -> None:
    """Expose every skills/*.md file as a native MCP prompt.

    The skills are primarily served through the list_skills / load_skill
    tools (which every MCP client supports); registering them as prompts as
    well lets prompt-capable clients surface the same recipes in their UI
    with zero tool calls.
    """
    from tools.meta.skills import skill_index, skill_text

    for name, description in skill_index().items():
        def make_prompt(skill_name: str):
            def prompt() -> str:
                return skill_text(skill_name)
            return prompt

        mcp.prompt(name=name, description=description)(make_prompt(name))


def run_server(*, transport: Transport = "stdio",
               host: str = "127.0.0.1", port: int = 8000) -> None:
    mcp = create_server(host=host, port=port)
    mcp.run(transport=transport)
