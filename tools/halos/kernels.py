"""HPC-dispatch kernels for the halo tools (see the dispatch contract R1).

JSON-safe in and out: masses arrive as a list, curves go back as lists; the
wrapper in ``__init__.py`` handles files, labels, and metadata wherever the
server runs. Imports stay inside functions / light at module scope so the
package copies cleanly onto a compute node.
"""

import numpy as np

from ..common import get_cached, quiet

# tinker08 is calibrated for spherical-overdensity definitions.
THEORY_MODELS = {"tinker08": "tinker08", "sheth_tormen": "sheth99",
                 "press_schechter": "press74"}

# Pip requirements per backend when this kernel runs on an HPC compute node
# (numpy/pydantic come from the tools-package import chain and are added by
# the wrapper / listed as base deps in the export manifest).
HMF_DISPATCH_PIP_DEPS = {
    "miratitan": ["MiraTitanHMFemulator"],
    "tinker08": ["colossus"],
    "sheth_tormen": ["colossus"],
    "press_schechter": ["colossus"],
}


def _param_slug_local(params: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in sorted(params.items()))


def theory_hmf(backend, mass_def, cosmo, masses, z):
    from colossus.cosmology import cosmology as ccosmo
    from colossus.lss import mass_function

    h = cosmo["h"]
    params = {"flat": True, "H0": h * 100,
              "Om0": cosmo["Ommh2"] / h**2, "Ob0": cosmo["Ombh2"] / h**2,
              "sigma8": cosmo["sigma_8"], "ns": cosmo["n_s"]}
    if cosmo["w_0"] != -1.0 or cosmo["w_a"] != 0.0:
        params.update(de_model="w0wa", w0=cosmo["w_0"], wa=cosmo["w_a"])
    name = f"hmf_{_param_slug_local(params)}"
    # persistence='' stops colossus writing interpolation tables to
    # $HOME/.colossus — read-only under the production systemd sandbox, and
    # equally unwelcome in an HPC job dir; in-memory caching still works.
    ccosmo.setCosmology(name, params, persistence="")

    model = THEORY_MODELS[backend]
    if model in ("press74", "sheth99"):
        if mass_def != "fof":
            raise ValueError(
                f"{backend} is a friends-of-friends multiplicity function; "
                "call it with mass_def='fof'. For an apples-to-apples "
                "comparison against the Mira-Titan emulator (M200c), use "
                "backend='tinker08' with mass_def='200c' — evaluating an "
                "FoF-calibrated fit at an SO mass is the classic way to get "
                "a spurious factor-of-a-few discrepancy.")
        mdef = "fof"
    else:
        mdef = mass_def
        if mdef == "fof":
            raise ValueError("tinker08 is SO-calibrated; use mass_def "
                             "'200c', '200m', or '500c'.")
    return mass_function.massFunction(np.asarray(masses, dtype=float), z,
                                      mdef=mdef, model=model, q_out="dndlnM")


def compute_hmf(backend: str, mass_def: str, cosmo: dict, masses,
                z: float, random_seed: int = 0) -> dict:
    """dn/dlnM for one backend — the dispatchable core of the compute_hmf tool.

    Returns {"dn_dlnM": [...], "emulator_std": [...]} (std is zeros for the
    analytic fits). ``cosmo`` uses the tool's physical-density convention:
    {Ommh2, Ombh2, Omnuh2, n_s, h, sigma_8, w_0, w_a}.
    """
    masses = np.asarray(masses, dtype=float)
    if backend == "miratitan":
        import MiraTitanHMFemulator
        if mass_def != "200c":
            raise ValueError("The Mira-Titan emulator provides M200c only; "
                             "use backend='tinker08' for other SO "
                             "definitions ('200m', '500c').")
        emu = get_cached("miratitan_hmf", MiraTitanHMFemulator.Emulator)
        with quiet():
            # the emulator's error estimate uses np.random draws internally;
            # seed for bitwise-reproducible outputs (provenance/caching)
            np.random.seed(random_seed)
            hmf_mean, hmf_err = emu.predict(cosmo, z, masses)
        hmf = np.ravel(np.asarray(hmf_mean))
        err = np.ravel(np.asarray(hmf_err))
    else:
        with quiet():
            hmf = np.ravel(np.asarray(theory_hmf(backend, mass_def, cosmo,
                                                 masses, z)))
        err = np.zeros_like(hmf)
    return {"dn_dlnM": hmf.tolist(), "emulator_std": err.tolist()}
