"""Backend adapters for matter power spectrum emulators.

Each adapter maps the server's uniform cosmology parameters onto one
emulator's native API and returns P(k) on the requested k grid
(k in h/Mpc, P in (Mpc/h)^3). Range violations raise ValueError with the
offending emulator named.
"""

import numpy as np

from ..common import get_cached, quiet


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
}
