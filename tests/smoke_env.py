"""Functional smoke test for the cosmic-emu environment.

Runs one real evaluation per wrapped emulator and reports OK/FAIL with timing.
Emulators that download large model files on first use are marked DEFERRED
unless --all is passed.

Usage:  python tests/smoke_env.py [--all]
"""

import argparse
import contextlib
import io
import sys
import time
import traceback

import numpy as np

# pybird (and some older codes) still call np.trapz, removed in numpy 2.x.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

RESULTS = []


def check(name, deferred=False):
    def deco(fn):
        def wrapper(run_all):
            if deferred and not run_all:
                RESULTS.append((name, "DEFERRED", 0.0, "large first-use download; run with --all"))
                return
            t0 = time.time()
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    detail = fn()
                RESULTS.append((name, "OK", time.time() - t0, detail or ""))
            except Exception:
                tb = traceback.format_exc().strip().splitlines()[-1]
                RESULTS.append((name, "FAIL", time.time() - t0, tb[:160]))
        wrapper._is_check = True
        return wrapper
    return deco


@check("camb (linear + HMcode2020)")
def _camb():
    import camb
    pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.12, ns=0.965, As=2e-9,
                           halofit_version="mead2020", redshifts=[0.0], kmax=10.0)
    res = camb.get_results(pars)
    kh, _, pk = res.get_matter_power_spectrum(minkh=1e-3, maxkh=10, npoints=50)
    return f"P(k=0.1)~{np.interp(0.1, kh, pk[0]):.1f} (Mpc/h)^3"


@check("symbolic_pofk (syren linear)")
def _syren():
    from symbolic_pofk import linear
    k = np.logspace(-2, 0, 10)
    pk = linear.plin_emulated(k, 0.8, 0.31, 0.049, 0.67, 0.965)
    return f"P(k=0.1)~{pk[4]:.1f}"


@check("baccoemu (linear + NL boost)")
def _bacco():
    import baccoemu
    emu = baccoemu.Matter_powerspectrum()
    params = dict(omega_cold=0.31, sigma8_cold=0.80, omega_baryon=0.05,
                  ns=0.965, hubble=0.67, neutrino_mass=0.0,
                  w0=-1.0, wa=0.0, expfactor=1.0)
    k = np.logspace(-2, 0.5, 20)
    _, pk_lin = emu.get_linear_pk(k=k, **params)
    _, pk_nl = emu.get_nonlinear_pk(k=k, baryonic_boost=False, **params)
    return f"boost(k=1)~{(pk_nl / pk_lin)[np.argmin(np.abs(k - 1))]:.2f}"


@check("euclidemu2 (NL boost)")
def _ee2():
    import euclidemu2
    emu = euclidemu2.PyEuclidEmulator()
    cosmo = dict(As=2.1e-9, ns=0.966, Omb=0.04897, Omm=0.3158,
                 h=0.6732, mnu=0.06, w=-1.0, wa=0.0)
    k, b = emu.get_boost(cosmo, [0.0])
    return f"boost(z=0, kmax)~{np.asarray(b[0])[-1]:.2f}"


@check("CEmulator / CSST (NL P(k))")
def _csst():
    from CEmulator.Emulator import Pkmm_CEmulator
    emu = Pkmm_CEmulator(neutrino_mass_split="single")
    emu.set_cosmos(Omegac=0.25, As=2e-9, mnu=0.06)
    k = np.logspace(-2, 0.5, 20)
    # NOTE: vendored numpy-2 patch in external/csstemu (scalar z; GP predict[0])
    pk = emu.get_pknl(z=0.0, k=k, Pcb=False,
                      lintype="Emulator", nltype="hmcode2020")
    return f"Pnl(k=0.1)~{pk[0][np.argmin(np.abs(k - 0.1))]:.1f}"


@check("gokunemu (10-param NL P(k))")
def _goku():
    from gokunemu import MatterPowerEmulator
    emu = MatterPowerEmulator()
    k, pk = emu.get_matter_power(Om=0.31, Ob=0.049, hubble=0.67, As=2.1e-9,
                                 ns=0.965, w0=-1.0, wa=0.0, mnu=0.06,
                                 Neff=3.044, alphas=0.0,
                                 redshifts=np.array([0.0]))
    return f"Pnl(k~0.1)~{np.interp(0.1, np.ravel(k), np.ravel(pk[0])):.1f}"


@check("emantis (f(R) boost)")
def _emantis():
    from emantis import FofrBoost
    emu = FofrBoost()
    # logfR0 is -log10|f_R0|, i.e. positive in [4, 7]; time is aexp
    b = emu.predict_boost(0.31, 0.82, 5.0, 1.0 / (1 + 0.5), k=np.array([0.1, 1.0]))
    return f"B(fR0=1e-5, k=1, z=0.5)~{float(np.atleast_1d(np.squeeze(b))[-1]):.3f}"


@check("nDGPemu (nDGP boost)")
def _ndgp():
    from nDGPemu import BoostPredictor
    emu = BoostPredictor()
    cosmo = {"Om": 0.31, "ns": 0.965, "As": 2.1e-9, "h": 0.67, "Ob": 0.049}
    b = emu.predict(H0rc=1.0, z=0.0, cosmo_params=cosmo)
    return f"B(H0rc=1, z=0) max~{np.max(b):.3f}"


@check("CubicGalileonEmu (Galileon boost)")
def _cubgal():
    # Needs the patched SEPIA in external/SEPIA (scipy>=1.11: sym_pos -> assume_a)
    import os
    import CubicGalileonEmu
    from CubicGalileonEmu import load as cgl
    from CubicGalileonEmu.emu import load_model_multiple, emulate
    B, B_sm, k, z_all = cgl.load_boost_data()
    params = cgl.load_params()
    model_dir = os.path.join(os.path.dirname(CubicGalileonEmu.__file__), "model/")
    models, datas = load_model_multiple(model_dir=model_dir, p_train_all=params,
                                        y_vals_all=B_sm, y_ind_all=k,
                                        z_index_range=[0])
    mean, std = emulate(sepia_model=models[0], sepia_data=datas[0],
                        input_params=np.array([[0.31, 0.965, 2.1, 0.67, 0.5]]))
    boost = np.ravel(np.asarray(mean))
    return f"boost(z=0) mean~{float(boost.mean()):.3f} max~{float(boost.max()):.3f}"


@check("cosmopower_jax (CMB TT)")
def _cpjax():
    from cosmopower_jax.cosmopower_jax import CosmoPowerJAX
    emu = CosmoPowerJAX(probe="cmb_tt")
    params = np.array([0.022, 0.12, 0.67, 0.06, 0.965, 3.05])
    cl = emu.predict(params)
    return f"Cl_TT len {len(cl)}"


@check("jaxcapse (CMB TT to l=5000)")
def _capse():
    import jaxcapse
    emu = jaxcapse.load_emulator(str(jaxcapse.get_emulator_path("TT")))
    cl = emu.get_Cl(np.array([0.022, 0.12, 0.67, 0.06, 0.965, 3.05]))
    return f"Cl len {len(cl)}"


@check("pybird (EFTofLSS, numpy2 trapz shim)")
def _pybird():
    import pybird
    from pybird.correlator import Correlator
    Correlator()
    return f"pybird {pybird.__version__} Correlator ready"


@check("jax_cosmo (Limber Cls)")
def _jaxcosmo():
    import jax_cosmo as jc
    import jax.numpy as jnp
    cosmo = jc.Planck15()
    nz = jc.redshift.smail_nz(1.0, 2.0, 1.0)
    probe = jc.probes.WeakLensing([nz])
    ell = jnp.array([100.0, 300.0])
    cl = jc.angular_cl.angular_cl(cosmo, ell, [probe])
    return f"Cl_kk(l=100)~{float(cl[0][0]):.2e}"


@check("pyspk (baryon suppression)")
def _pyspk():
    import pyspk.model as spk
    # fb here is normalised by Omega_b/Omega_m (power-law form, README quickstart)
    k, sup = spk.sup_model(SO=200, z=0.125, fb_a=0.4, fb_pow=0.3, fb_pivot=10**13.5)
    val = float(np.nanmin(sup))
    if np.isnan(val):
        raise ValueError("all-NaN suppression — fb inputs outside fitted range")
    return f"suppression min~{val:.3f}"


@check("MiraTitanHMFemulator (HMF)")
def _mthmf():
    import MiraTitanHMFemulator
    emu = MiraTitanHMFemulator.Emulator()
    cosmo = {"Ommh2": 0.147, "Ombh2": 0.022, "Omnuh2": 0.0006, "n_s": 0.965,
             "h": 0.67, "sigma_8": 0.8, "w_0": -1.0, "w_a": 0.0}
    hmf = emu.predict(cosmo, 0.0, np.array([1e14]))
    return f"dn/dlnM(1e14)~{float(np.ravel(hmf)[0]):.2e}"


@check("subgrid_emu (CRK-HACC subgrid GP)")
def _subgrid():
    import subgrid_emu
    return "import OK (SEPIA GP; models load from package data)"


@check("picasso (cluster gas painting)")
def _picasso():
    import jax.numpy as jnp
    from picasso import predictors
    pred = predictors.baseline_576
    # 12 halo properties: log M200, c200, cacc/c200, cpeak/c200,
    # log dx/R200c, e, p, a25, a50, a75, almm, mdot
    x = jnp.array([14.0, 5.0, 1.0, 1.0, -2.0, 0.1, 0.05, 0.3, 0.5, 0.7, 0.8, 1.0])
    theta = pred.predict_model_parameters(x)
    return f"gas-model params len {len(np.ravel(theta))}"


@check("LaCE (Lya P1D GP emulator)")
def _lace():
    from lace.emulator.gp_emulator import GPEmulator
    # Needs the LaCE repo clone in external/LaCE (pip wheel omits training data;
    # module-level GPy import patched there)
    emu = GPEmulator(training_set="Pedersen21", emulator_label="Pedersen21")
    p1d = emu.emulate_p1d_Mpc(
        {"Delta2_p": 0.35, "n_p": -2.3, "mF": 0.66,
         "sigT_Mpc": 0.13, "gamma": 1.5, "kF_Mpc": 10.5},
        np.array([0.5, 1.0]))
    return f"P1D(k=0.5)~{float(np.ravel(p1d)[0]):.3f}"


@check("forestflow (Lya P3D)", deferred=True)
def _forestflow():
    import forestflow
    return "import OK"


@check("py21cmemu (21cmFAST emulator)", deferred=True)
def _21cm():
    from py21cmemu import Emulator
    Emulator()  # downloads ~100s MB from HuggingFace on first use
    return "emulator instantiated"


@check("classy_szfast (emulated CLASS + tSZ)", deferred=True)
def _classysz():
    from classy_sz import Class  # first use downloads ~GB of emulator data
    return "import OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also run tests with large first-use downloads")
    args = ap.parse_args()

    checks = [v for v in globals().values() if getattr(v, "_is_check", False)]
    for fn in checks:
        fn(args.all)

    width = max(len(n) for n, *_ in RESULTS) + 2
    n_fail = 0
    for name, status, dt, detail in RESULTS:
        print(f"{status:9s}{name:{width}s}{dt:6.1f}s  {detail}")
        n_fail += status == "FAIL"
    print(f"\n{len(RESULTS)} checks: {sum(1 for r in RESULTS if r[1]=='OK')} OK, "
          f"{n_fail} FAIL, {sum(1 for r in RESULTS if r[1]=='DEFERRED')} deferred")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
