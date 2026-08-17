"""Discovery tools: what emulators exist, what they cover, how to use them well."""

from typing import Annotated

from pydantic import Field, validate_call

from ..common import ArtifactResult
from .registry import EMULATORS
from .skills import skill_index, skill_text

__all__ = ["list_emulators", "describe_emulator", "list_skills", "load_skill"]


@validate_call
def list_emulators(
    family: Annotated[str, Field(description="Filter by family: pk, gravity, cmb, lss, baryons, halos, igm. Empty string = all.")] = "",
) -> ArtifactResult:
    """List the cosmological emulators available on this server.

    Call this first to discover what the server can compute. Each entry gives
    the emulator family, its role, which tools expose it, and its status
    ("active" = callable now; "deferred" = installed but not yet exposed).
    Use describe_emulator for parameter ranges and unit conventions.
    """
    entries = {
        name: {"family": e["family"], "role": e["role"],
               "tools": e["tools"], "status": e["status"]}
        for name, e in EMULATORS.items()
        if not family or e["family"] == family
    }
    active = sum(1 for e in entries.values() if e["status"] == "active")
    return ArtifactResult(
        status="success",
        files=[],
        message=f"{len(entries)} emulators ({active} active). "
                "Families: pk, gravity, cmb, lss, baryons, halos, igm.",
        metadata={"emulators": entries},
    )


@validate_call
def describe_emulator(
    name: Annotated[str, Field(min_length=1, description="Emulator key from list_emulators, e.g. 'baccoemu'.")],
) -> ArtifactResult:
    """Full metadata for one emulator: parameter ranges, units, accuracy, citation.

    Always check the parameter ranges here before calling a compute tool —
    emulators silently extrapolate (neural networks) or error (Gaussian
    processes) outside their training box.
    """
    if name not in EMULATORS:
        raise ValueError(
            f"Unknown emulator '{name}'. Valid: {', '.join(sorted(EMULATORS))}")
    return ArtifactResult(
        status="success",
        files=[],
        message=f"{name}: {EMULATORS[name]['role']}",
        metadata=EMULATORS[name],
    )


@validate_call
def list_skills() -> ArtifactResult:
    """List this server's skills: named recipes for multi-tool workflows.

    A skill is procedural know-how — which tools to combine, in what order,
    with what parameters, and how to interpret the results. Returns one
    name + description per skill; when a task matches one, call load_skill
    and follow the loaded instructions.
    """
    index = skill_index()
    return ArtifactResult(
        status="success",
        files=[],
        message=f"{len(index)} skills available. Load one with load_skill "
                "when a task matches its description.",
        metadata={"skills": index},
    )


@validate_call
def load_skill(
    name: Annotated[str, Field(min_length=1, description="Skill name from list_skills, e.g. 'pk-crosscheck'.")],
) -> ArtifactResult:
    """Load the full instructions of a named skill; then follow them.

    Skills encode validated workflows over this server's tools (correct
    tool ordering, parameter choices, physical sanity checks, and how to
    report the results).
    """
    text = skill_text(name)
    if text is None:
        raise ValueError(
            f"No skill named '{name}'. Available: "
            f"{', '.join(sorted(skill_index()))}")
    return ArtifactResult(
        status="success",
        files=[],
        message=f"Loaded skill '{name}'. Follow these instructions.",
        metadata={"instructions": text},
    )
