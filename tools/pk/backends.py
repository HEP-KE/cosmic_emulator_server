"""Backend adapters for matter power spectrum emulators.

Each adapter maps the server's uniform cosmology parameters onto one
emulator's native API and returns P(k) on the requested k grid
(k in h/Mpc, P in (Mpc/h)^3). Range violations raise ValueError with the
offending emulator named.
"""

import numpy as np

from ..common import get_cached, quiet


def _goku_box():
    """GokuNEmu training bounds, read from the file shipped in the package."""
    import os
    import gokunemu
    limits = np.loadtxt(os.path.join(os.path.dirname(gokunemu.__file__),
                                     "input_limits-W.txt"))
    names = ["Om", "Ob", "h", "As", "ns", "w0", "wa", "mnu", "Neff", "alphas"]
    return {n: [float(lo), float(hi)] for n, (lo, hi) in zip(names, limits)}


# Canonical-parameter training boxes per backend (None = no hard box).
# Sources: emulator docs/papers, verified in EMULATOR_CATALOG.md; gokunemu's
# is read from its own limits file at first use.
TRAINING_BOXES: dict[str, dict | None] = {
    "camb": None,
    "camb_hmcode": None,
    "syren": {"Om": [0.2, 0.45], "Ob": [0.03, 0.07], "h": [0.55, 0.85],
              "ns": [0.85, 1.05], "sigma8": [0.6, 1.0], "z": [0.0, 3.0]},
    "syren_halofit": {"Om": [0.2, 0.45], "Ob": [0.03, 0.07], "h": [0.55, 0.85],
                      "ns": [0.85, 1.05], "sigma8": [0.6, 1.0], "z": [0.0, 3.0]},
    "baccoemu": {"Om": [0.23, 0.40], "Ob": [0.04, 0.06], "h": [0.6, 0.8],
                 "ns": [0.92, 1.01], "sigma8": [0.73, 0.90], "mnu": [0.0, 0.4],
                 "w0": [-1.15, -0.85], "wa": [-0.3, 0.3], "z": [0.0, 1.5]},
    "euclidemu2": {"Om": [0.24, 0.40], "Ob": [0.04, 0.06], "h": [0.61, 0.73],
                   "ns": [0.92, 1.00], "As": [1.7e-9, 2.5e-9],
                   "w0": [-1.3, -0.7], "wa": [-0.7, 0.5], "mnu": [0.0, 0.15],
                   "z": [0.0, 3.0]},
    "csst": {"Om": [0.24, 0.40], "Ob": [0.04, 0.06], "h": [0.6, 0.8],
             "ns": [0.92, 1.00], "As": [1.7e-9, 2.5e-9], "w0": [-1.3, -0.7],
             "wa": [-0.5, 0.5], "mnu": [0.0, 0.3], "z": [0.0, 3.0]},
    "gokunemu": "FROM_PACKAGE",  # resolved lazily via _goku_box()
    "miratitan": "CUSTOM_MT4",   # box is in physical densities; see check_box
}


def _miratitan_box_warnings(values: dict) -> list[str]:
    """Mira-Titan IV design box (Moran et al. 2022), in its native
    parameterization: physical densities + a joint (w0, wa) constraint."""
    h = values["h"]
    derived = {
        "omega_m = Om*h^2": (values["Om"] * h**2, 0.12, 0.155),
        "omega_b = Ob*h^2": (values["Ob"] * h**2, 0.0215, 0.0235),
        "omega_nu = mnu/93.14": (values.get("mnu", 0.0) / 93.14, 0.0, 0.01),
        "sigma8": (values.get("sigma8"), 0.7, 0.9),
        "h": (h, 0.55, 0.85),
        "ns": (values["ns"], 0.85, 1.05),
        "w0": (values["w0"], -1.3, -0.7),
        "(-w0-wa)^(1/4)": ((-values["w0"] - values["wa"]) ** 0.25
                           if -values["w0"] - values["wa"] > 0 else -1.0,
                           0.3, 1.29),
        "z": (values["z"], 0.0, 2.02),
    }
    warnings = []
    for name, (v, lo, hi) in derived.items():
        if v is not None and not (lo <= v <= hi):
            warnings.append(
                f"miratitan: {name}={v:g} outside training box [{lo:g}, {hi:g}]"
                " — output is an extrapolation")
    return warnings


def check_box(backend: str, params: dict, z: float) -> tuple[bool, list[str]]:
    """Compare parameters against the backend's training box.

    Returns (in_training_box, warnings). Schema-level validation stays a
    union across backends so multi-backend comparisons remain expressible;
    this per-backend check makes extrapolation VISIBLE in every response
    instead of silent (NN backends do not guard themselves).
    """
    box = TRAINING_BOXES.get(backend)
    if box == "FROM_PACKAGE":
        box = get_cached("goku_box", _goku_box)
    if box == "CUSTOM_MT4":
        warnings = _miratitan_box_warnings(dict(params, z=z))
        return not warnings, warnings
    if not box:
        return True, []
    values = dict(params, z=z)
    warnings = []
    for name, (lo, hi) in box.items():
        v = values.get(name)
        if v is None:
            continue
        if not (lo <= v <= hi):
            warnings.append(
                f"{backend}: {name}={v:g} outside training box [{lo:g}, {hi:g}]"
                " — output is an extrapolation")
    return not warnings, warnings


def _camb_results(params: dict, z: float, kmax: float, nonlinear: bool):
    import camb
    kwargs = dict(
        H0=params["h"] * 100, ombh2=params["Ob"] * params["h"] ** 2,
        omch2=(params["Om"] - params["Ob"]) * params["h"] ** 2 - params["mnu"] / 93.14,
        ns=params["ns"], As=params["As"], redshifts=[z],
        kmax=max(kmax * 1.2, 1.0), mnu=params["mnu"],
    )
    if nonlinear:
        kwargs["halofit_version"] = "mead2020"
    if params["w0"] != -1.0 or params["wa"] != 0.0:
        kwargs.update(w=params["w0"], wa=params["wa"],
                      dark_energy_model="ppf")
    pars = camb.set_params(**kwargs)
    # set_params(halofit_version=...) alone does NOT switch the matter power
    # to nonlinear — NonLinear must be set explicitly or CAMB returns linear.
    pars.NonLinear = camb.model.NonLinear_pk if nonlinear else camb.model.NonLinear_none
    return camb.get_results(pars)


def camb_pk(params: dict, k: np.ndarray, z: float, nonlinear: bool) -> np.ndarray:
    res = _camb_results(params, z, float(k[-1]), nonlinear)
    kh, _, pk = res.get_matter_power_spectrum(
        minkh=float(k[0]), maxkh=float(k[-1]), npoints=max(len(k), 200))
    return np.interp(k, kh, pk[0])


def camb_sigma8(params: dict) -> float:
    res = _camb_results(params, 0.0, 1.0, False)
    return float(res.get_sigma8_0())


def _resolve_sigma8(params: dict) -> float:
    """sigma8 for backends that need it: given value, or derived from As via CAMB."""
    if params.get("sigma8") is not None:
        return params["sigma8"]
    return get_cached(
        f"sigma8:{params['Om']}:{params['Ob']}:{params['h']}:{params['ns']}:{params['As']}:{params['mnu']}:{params['w0']}:{params['wa']}",
        lambda: camb_sigma8(params))


def syren_linear(params: dict, k: np.ndarray, z: float) -> np.ndarray:
    from symbolic_pofk import linear
    sigma8 = _resolve_sigma8(params)
    pk0 = linear.plin_emulated(k, sigma8, params["Om"], params["Ob"],
                               params["h"], params["ns"])
    if z == 0.0:
        return pk0
    # scale-independent growth via CAMB linear ratio at one anchor scale
    ratio = camb_pk(params, k, z, False) / camb_pk(params, k, 0.0, False)
    return pk0 * ratio


def syren_nonlinear(params: dict, k: np.ndarray, z: float) -> np.ndarray:
    from symbolic_pofk import syrenhalofit
    sigma8 = _resolve_sigma8(params)
    a = 1.0 / (1.0 + z)
    return syrenhalofit.run_halofit(
        k, sigma8, params["Om"], params["Ob"], params["h"], params["ns"], a,
        emulator="fiducial", extrapolate=True)


def _bacco():
    import baccoemu
    return baccoemu.Matter_powerspectrum()


def _bacco_params(params: dict, z: float) -> dict:
    return dict(
        omega_cold=params["Om"], omega_baryon=params["Ob"],
        sigma8_cold=_resolve_sigma8(params), ns=params["ns"],
        hubble=params["h"], neutrino_mass=params["mnu"],
        w0=params["w0"], wa=params["wa"], expfactor=1.0 / (1.0 + z),
    )
    # NOTE: we feed total-matter Om/sigma8 as cold quantities; exact only for
    # mnu = 0. The tool docstring flags this and metadata records it.


def bacco_pk(params: dict, k: np.ndarray, z: float, nonlinear: bool) -> np.ndarray:
    emu = get_cached("baccoemu", _bacco)
    bp = _bacco_params(params, z)
    with quiet():
        if nonlinear:
            _, pk = emu.get_nonlinear_pk(k=k, baryonic_boost=False, **bp)
        else:
            _, pk = emu.get_linear_pk(k=k, **bp)
    return np.asarray(pk)


def ee2_nonlinear(params: dict, k: np.ndarray, z: float) -> np.ndarray:
    """EuclidEmulator2 boost x CAMB linear P(k)."""
    import euclidemu2
    emu = get_cached("euclidemu2", euclidemu2.PyEuclidEmulator)
    cosmo = dict(As=params["As"], ns=params["ns"], Omb=params["Ob"],
                 Omm=params["Om"], h=params["h"], mnu=params["mnu"],
                 w=params["w0"], wa=params["wa"])
    with quiet():
        k_emu, boosts = emu.get_boost(cosmo, [z])
    boost = np.interp(k, np.asarray(k_emu), np.asarray(boosts[0]))
    return camb_pk(params, k, z, False) * boost


def csst_nonlinear(params: dict, k: np.ndarray, z: float) -> np.ndarray:
    from CEmulator.Emulator import Pkmm_CEmulator
    emu = get_cached("csst", lambda: Pkmm_CEmulator(neutrino_mass_split="single"))
    with quiet():
        emu.set_cosmos(Omegab=params["Ob"], Omegac=params["Om"] - params["Ob"],
                       H0=params["h"] * 100, As=params["As"], ns=params["ns"],
                       w=params["w0"], wa=params["wa"], mnu=params["mnu"])
        pk = emu.get_pknl(z=z, k=k, Pcb=False, lintype="Emulator",
                          nltype="hmcode2020")
    return np.ravel(np.asarray(pk))


def miratitan_nonlinear(params: dict, k: np.ndarray, z: float) -> np.ndarray:
    """Mira-Titan IV (CosmicEmu) via pyccl's native reimplementation.

    CCL works in h-free units (k [1/Mpc], P [Mpc^3]) — converted here to
    the server convention (k [h/Mpc], P [(Mpc/h)^3]). sigma8 is required by
    the emulator; derived from As via CAMB when not given.
    """
    import pyccl as ccl

    h = params["h"]
    sigma8 = _resolve_sigma8(params)
    omega_nu_frac = params["mnu"] / 93.14 / h**2
    cosmo = ccl.Cosmology(
        Omega_c=params["Om"] - params["Ob"] - omega_nu_frac,
        Omega_b=params["Ob"], h=h, n_s=params["ns"], sigma8=sigma8,
        w0=params["w0"], wa=params["wa"], m_nu=params["mnu"],
        matter_power_spectrum="linear")
    emu = get_cached("ccl_miratitan_mt4", lambda: ccl.CosmicemuMTIVPk("tot"))
    with quiet():
        pk2d = emu.get_pk2d(cosmo)
        pk = pk2d(k * h, 1.0 / (1.0 + z), cosmo) * h**3
    return np.ravel(np.asarray(pk))


def goku_nonlinear(params: dict, k: np.ndarray, z: float) -> np.ndarray:
    from gokunemu import MatterPowerEmulator
    emu = get_cached("gokunemu", MatterPowerEmulator)
    with quiet():
        k_emu, pk = emu.get_matter_power(
            Om=params["Om"], Ob=params["Ob"], hubble=params["h"],
            As=params["As"], ns=params["ns"], w0=params["w0"],
            wa=params["wa"], mnu=params["mnu"], Neff=3.044, alphas=0.0,
            redshifts=np.array([z]))
    return np.interp(k, np.ravel(k_emu), np.ravel(pk[0]))


LINEAR_BACKENDS = {
    "camb": lambda p, k, z: camb_pk(p, k, z, False),
    "syren": syren_linear,
    "baccoemu": lambda p, k, z: bacco_pk(p, k, z, False),
}

NONLINEAR_BACKENDS = {
    "camb_hmcode": lambda p, k, z: camb_pk(p, k, z, True),
    "syren_halofit": syren_nonlinear,
    "baccoemu": lambda p, k, z: bacco_pk(p, k, z, True),
    "euclidemu2": ee2_nonlinear,
    "csst": csst_nonlinear,
    "gokunemu": goku_nonlinear,
    "miratitan": miratitan_nonlinear,
}
