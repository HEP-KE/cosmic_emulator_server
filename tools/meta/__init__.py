"""Discovery tools: what emulators exist, what they cover, how to use them well."""

from typing import Annotated

from pydantic import Field, validate_call

from ..common import ArtifactResult, get_cached
from .registry import EMULATORS
from .skills import skill_index, skill_text

__all__ = ["list_emulators", "describe_emulator", "convert_cosmology",
           "list_emulator_skills", "load_emulator_skill"]

# Names agents reach for (the tools' backend/model arguments, package names,
# simulation-suite names) -> registry keys. Seen in practice: 'spk',
# 'crk_hacc_subgrid'. Keys are normalized (lowercase, '-' -> '_').
ALIASES = {
    "spk": "pyspk", "sp(k)": "pyspk",
    "bacco": "baccoemu", "bacco_baryons": "baccoemu",
    "syren_illustristng": "syren_baryon", "syren_astrid": "syren_baryon",
    "syren_simba": "syren_baryon", "syren_swift_eagle": "syren_baryon",
    "syren_halofit": "syren", "symbolic_pofk": "syren",
    "camb_hmcode": "camb", "hmcode": "camb",
    "fofr": "emantis", "f(r)": "emantis", "hu_sawicki": "emantis",
    "ndgp": "ndgpemu", "galileon": "cubic_galileon",
    "cubicgalileonemu": "cubic_galileon",
    "subgrid": "subgrid_emu", "crk_hacc": "subgrid_emu",
    "crk_hacc_subgrid": "subgrid_emu", "crkhacc": "subgrid_emu",
    "tinker08": "colossus_hmf", "sheth_tormen": "colossus_hmf",
    "press_schechter": "colossus_hmf", "colossus": "colossus_hmf",
    "cosmopower": "cosmopower_jax", "jaxcapse": "capse",
    "euclidemu": "euclidemu2", "goku": "gokunemu", "cemulator": "csst",
    "cosmicemu": "cosmicemu_mt4", "mira_titan_pk": "cosmicemu_mt4",
    "mira_titan_hmf": "miratitan_hmf", "lace_gp": "lace",
}
AMBIGUOUS = {"miratitan": ["cosmicemu_mt4 (nonlinear P(k))",
                           "miratitan_hmf (halo mass function)"],
             "mira_titan": ["cosmicemu_mt4 (nonlinear P(k))",
                            "miratitan_hmf (halo mass function)"],
             "syren_baryons": ["syren_baryon"]}


def resolve_emulator_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in EMULATORS:
        return key
    if key in ALIASES:
        return ALIASES[key]
    if key in AMBIGUOUS:
        raise ValueError(f"'{name}' is ambiguous — did you mean "
                         f"{' or '.join(AMBIGUOUS[key])}?")
    raise ValueError(
        f"Unknown emulator '{name}'. Valid: {', '.join(sorted(EMULATORS))} "
        "(tool backend/model names such as 'spk', 'bacco', 'fofr', "
        "'tinker08' are accepted too).")


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
    name: Annotated[str, Field(min_length=1, description="Emulator key from list_emulators (e.g. 'baccoemu', 'pyspk'), or the backend/model name a compute tool takes (e.g. 'spk', 'bacco', 'fofr', 'tinker08').")],
) -> ArtifactResult:
    """Full metadata for one emulator: parameter ranges, units, accuracy, citation.

    Always check the parameter ranges here before calling a compute tool —
    emulators silently extrapolate (neural networks) or error (Gaussian
    processes) outside their training box.
    """
    key = resolve_emulator_name(name)
    message = f"{key}: {EMULATORS[key]['role']}"
    if key != name:
        message = f"('{name}' resolves to '{key}') " + message
    return ArtifactResult(
        status="success",
        files=[],
        message=message,
        metadata=dict(EMULATORS[key], registry_key=key),
    )


@validate_call
def convert_cosmology(
    Om: Annotated[float, Field(ge=0.1, le=0.6, description="Total matter density Omega_m")] = 0.31,
    Ob: Annotated[float, Field(ge=0.02, le=0.08)] = 0.049,
    h: Annotated[float, Field(ge=0.5, le=0.9)] = 0.67,
    ns: Annotated[float, Field(ge=0.8, le=1.1)] = 0.965,
    As: Annotated[float | None, Field(ge=5e-10, le=5e-9, description="Give As OR sigma8 (one required). If both, As wins and sigma8 is recomputed.")] = None,
    sigma8: Annotated[float | None, Field(ge=0.4, le=1.3, description="sigma8 at z=0")] = None,
    mnu: Annotated[float, Field(ge=0.0, le=1.0, description="Sum of neutrino masses in eV")] = 0.0,
    w0: Annotated[float, Field(ge=-2.0, le=-0.3)] = -1.0,
    wa: Annotated[float, Field(ge=-3.0, le=1.0)] = 0.0,
    tau: Annotated[float, Field(ge=0.01, le=0.15, description="Optical depth (only enters the CMB parameterization)")] = 0.055,
) -> ArtifactResult:
    """Translate one cosmology into every parameterization this server's tools use.

    Different families take different conventions (P(k)/lensing: Om, Ob, h,
    sigma8; HMF: physical densities Ommh2/Ombh2/Omnuh2; CMB: omega_b,
    omega_cdm, ln10As — no sigma8 or w0). This tool does the exact
    CAMB-backed mapping in one call so nothing is hand-converted: give
    (As OR sigma8) plus the rest, get back the argument dict for each tool
    family, ready to paste into the corresponding compute call. The
    As<->sigma8 conversion solves the actual Boltzmann amplitude relation
    (not a fit). NOTE: w0/wa cannot be forwarded to the CMB emulators
    (their training space is LCDM) — flagged in the response when w0 != -1.
    """
    import numpy as np

    if As is None and sigma8 is None:
        raise ValueError("Provide As or sigma8 (one of the two amplitudes).")

    from ..pk import backends as pk_backends

    if As is not None:
        params = {"Om": Om, "Ob": Ob, "h": h, "ns": ns, "As": As,
                  "sigma8": None, "mnu": mnu, "w0": w0, "wa": wa}
        sigma8_out = pk_backends.camb_sigma8(params)
        As_out = As
    else:
        # invert sigma8 -> As: amplitude scales linearly with As
        As_ref = 2.1e-9
        params = {"Om": Om, "Ob": Ob, "h": h, "ns": ns, "As": As_ref,
                  "sigma8": None, "mnu": mnu, "w0": w0, "wa": wa}
        s8_ref = pk_backends.camb_sigma8(params)
        As_out = As_ref * (sigma8 / s8_ref) ** 2
        sigma8_out = sigma8

    omega_nu = mnu / 93.14
    conversions = {
        "pk_and_lensing": {"Om": Om, "Ob": Ob, "h": h, "ns": ns,
                           "As": round(As_out, 15), "sigma8": round(sigma8_out, 5),
                           "mnu": mnu, "w0": w0, "wa": wa},
        "hmf_miratitan": {"Ommh2": round(Om * h**2, 6),
                          "Ombh2": round(Ob * h**2, 6),
                          "Omnuh2": round(omega_nu, 6),
                          "sigma_8": round(sigma8_out, 5), "h": h,
                          "n_s": ns, "w_0": w0, "w_a": wa},
        "cmb": {"omega_b": round(Ob * h**2, 6),
                "omega_cdm": round((Om - Ob) * h**2 - omega_nu, 6),
                "h": h, "tau": tau, "ns": ns,
                "ln10As": round(float(np.log(As_out * 1e10)), 5)},
    }
    notes = ["sigma8 is sigma8(z=0)",
             "cmb.omega_cdm excludes the neutrino density (physical CDM only)"]
    if w0 != -1.0 or wa != 0.0:
        notes.append("WARNING: w0/wa CANNOT be forwarded to the CMB emulators "
                     "(LCDM training space) — the cmb block above ignores them")
    return ArtifactResult(
        status="success", files=[],
        message=f"Converted: sigma8(z=0)={sigma8_out:.4f} <-> As={As_out:.4e} "
                f"(ln10As={conversions['cmb']['ln10As']}).",
        metadata={"conversions": conversions, "notes": notes},
    )


@validate_call
def list_emulator_skills() -> ArtifactResult:
    """List this emulator server's skills: recipes for multi-tool workflows.

    A skill is procedural know-how — which emulator tools to combine, in
    what order, with what parameters, and how to interpret the results.
    Returns one name + description per skill; when a task matches one, call
    load_emulator_skill and follow the loaded instructions. (These cover
    this server's tools only; other servers' skills are separate.)
    """
    index = skill_index()
    return ArtifactResult(
        status="success",
        files=[],
        message=f"{len(index)} skills available. Load one with "
                "load_emulator_skill when a task matches its description.",
        metadata={"skills": index},
    )


@validate_call
def load_emulator_skill(
    name: Annotated[str, Field(min_length=1, description="Skill name from list_emulator_skills, e.g. 'pk-crosscheck'.")],
) -> ArtifactResult:
    """Load the full instructions of one of this emulator server's skills.

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
