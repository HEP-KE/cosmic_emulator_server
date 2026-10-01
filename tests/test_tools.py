"""Integration tests: call every MCP tool with real emulator evaluations.

Run:  python -m pytest tests/test_tools.py -x -q     (or python tests/test_tools.py)

Uses a temp directory for artifacts; each test asserts the ArtifactResult
contract and basic physical sanity of the numbers.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.common import read_csv  # noqa: E402
from tools import analysis, baryons, cmb, gravity, halos, igm, lss, meta, pk  # noqa: E402

OUT = tempfile.mkdtemp(prefix="cosmic_emu_test_")


def _ok(result):
    assert result.status == "success"
    for f in result.files:
        assert Path(f).exists(), f"missing artifact {f}"
    return result


def test_meta():
    r = _ok(meta.list_emulators())
    assert "baccoemu" in r.metadata["emulators"]
    r = _ok(meta.describe_emulator(name="cubic_galileon"))
    assert r.metadata["ranges"]["f_phi"] == [0.02, 1.0]
    with pytest.raises(ValueError):
        meta.describe_emulator(name="nope")
    # names agents actually tried (tool backend/model names, suite names)
    for alias, key in [("spk", "pyspk"), ("crk_hacc_subgrid", "subgrid_emu"),
                       ("crk-hacc-subgrid", "subgrid_emu"), ("fofr", "emantis"),
                       ("tinker08", "colossus_hmf"), ("BACCO", "baccoemu")]:
        assert meta.describe_emulator(name=alias).metadata["registry_key"] == key
    with pytest.raises(ValueError, match="ambiguous"):
        meta.describe_emulator(name="miratitan")


def test_skills():
    r = _ok(meta.list_emulator_skills())
    skills = r.metadata["skills"]
    assert {"cosmo-tour", "pk-crosscheck", "mg-explore",
            "baryon-budget"} <= set(skills)
    assert all(desc for desc in skills.values()), "every skill needs a description"
    r = _ok(meta.load_emulator_skill(name="pk-crosscheck"))
    text = r.metadata["instructions"]
    assert "compute_nonlinear_pk" in text and "plot_emulator_curves" in text
    with pytest.raises(ValueError):
        meta.load_emulator_skill(name="nope")


def test_linear_pk_backends():
    results = {}
    for backend in ("syren", "camb", "baccoemu"):
        r = _ok(pk.compute_linear_pk(output_dir=OUT, backend=backend,
                                     k_min=0.01, k_max=1.0, n_points=50))
        _, cols = read_csv(r.files[0])
        results[backend] = cols["Pk_Mpc_over_h_cubed"]
    # backends must agree at the few-percent level on linear scales
    for backend, pk_vals in results.items():
        ratio = pk_vals / results["camb"]
        assert np.all(np.abs(ratio - 1) < 0.05), f"{backend} deviates >5% from CAMB"


def test_nonlinear_pk_backends():
    results = {}
    for backend in ("baccoemu", "euclidemu2", "csst", "gokunemu",
                    "miratitan", "camb_hmcode", "syren_halofit"):
        r = _ok(pk.compute_nonlinear_pk(output_dir=OUT, backend=backend,
                                        k_min=0.01, k_max=3.0, n_points=60))
        _, cols = read_csv(r.files[0])
        results[backend] = cols["Pk_Mpc_over_h_cubed"]
    for backend, pk_vals in results.items():
        ratio = pk_vals / results["baccoemu"]
        assert np.all(np.abs(ratio - 1) < 0.12), \
            f"{backend} deviates >12% from baccoemu"


def test_plot_pk():
    f1 = pk.compute_nonlinear_pk(output_dir=OUT, backend="syren_halofit").files[0]
    f2 = pk.compute_nonlinear_pk(output_dir=OUT, backend="camb_hmcode").files[0]
    r = _ok(analysis.plot_emulator_curves(curve_files=[f1, f2], output_dir=OUT,
                                          save_pdf=True))
    assert [Path(f).suffix for f in r.files] == [".png", ".pdf"]


def test_mg_boosts():
    for model, kwargs in [("fofr", {"minus_log10_fR0": 5.0}),
                          ("ndgp", {"H0rc": 1.0}),
                          ("cubic_galileon", {"f_phi": 0.5})]:
        r = _ok(gravity.compute_mg_boost(output_dir=OUT, model=model,
                                         k_max=3.0, **kwargs))
        _, cols = read_csv(r.files[0])
        boost = cols["boost"]
        assert np.all(boost > 0.9) and np.all(boost < 1.6), \
            f"{model} boost outside physical range"
        assert np.max(boost) > 1.005, f"{model} boost suspiciously flat"
    # GP uncertainty column present for the SEPIA emulator
    assert "gp_std" in cols


def test_mg_pk_composed():
    r = _ok(gravity.compute_mg_pk(output_dir=OUT, model="fofr",
                                  baseline="baccoemu", k_max=2.0))
    _, cols = read_csv(r.files[0])
    assert np.all(cols["Pk_Mpc_over_h_cubed"] > 0)
    assert r.metadata["baseline"] == "baccoemu"


def test_cmb_backends_agree():
    r1 = _ok(cmb.compute_cmb_cls(output_dir=OUT, spectrum="TT",
                                 backend="capse"))
    r2 = _ok(cmb.compute_cmb_cls(output_dir=OUT, spectrum="TT",
                                 backend="cosmopower_jax"))
    _, c1 = read_csv(r1.files[0])
    _, c2 = read_csv(r2.files[0])
    n = min(len(c1["ell"]), len(c2["ell"]))
    # sub-percent cross-emulator agreement on the acoustic peaks
    sel = (c1["ell"][:n] > 100) & (c1["ell"][:n] < 2000)
    ratio = c1["Dl_muK2"][:n][sel] / c2["Dl_muK2"][:n][sel]
    assert np.all(np.abs(ratio - 1) < 0.02)
    r = _ok(analysis.plot_emulator_curves(curve_files=[r1.files[0], r2.files[0]],
                                          output_dir=OUT))
    assert r.metadata["axes"] == {"logx": False, "logy": False}


def test_cmb_pp_capse_only():
    _ok(cmb.compute_cmb_cls(output_dir=OUT, spectrum="PP", backend="capse"))
    with pytest.raises(Exception):
        cmb.compute_cmb_cls(output_dir=OUT, spectrum="PP",
                            backend="cosmopower_jax")


def test_galaxy_multipoles():
    lin = pk.compute_linear_pk(output_dir=OUT, backend="camb", z=0.5,
                               k_min=1e-4, k_max=1.0, n_points=300).files[0]
    r = _ok(lss.compute_galaxy_multipoles(linear_pk_file=lin, output_dir=OUT,
                                          z=0.5, b1=2.0))
    _, cols = read_csv(r.files[0])
    assert np.all(cols["P0"] > 0)
    assert cols["P0"][0] > cols["P4"][0]


def test_lensing_cls():
    r = _ok(lss.compute_lensing_cls(output_dir=OUT, n_ell=20))
    _, cols = read_csv(r.files[0])
    assert np.all(cols["Cl_kappa"] > 0)
    assert cols["Cl_kappa"][0] > cols["Cl_kappa"][-1]  # decreasing spectrum


def test_baryon_suppression_models():
    for model in ("spk", "bacco", "syren_IllustrisTNG", "syren_SIMBA"):
        r = _ok(baryons.compute_baryon_suppression(output_dir=OUT, model=model))
        _, cols = read_csv(r.files[0])
        sup = cols["suppression"][~np.isnan(cols["suppression"])]
        assert len(sup) > 10
        assert np.nanmin(sup) > 0.5 and np.nanmax(sup) < 1.3, \
            f"{model} suppression unphysical"


def test_subgrid_statistic():
    for stat in ("Pk", "GSMF"):
        r = _ok(baryons.emulate_subgrid_statistic(output_dir=OUT, statistic=stat))
        _, cols = read_csv(r.files[0])
        assert np.all(cols["gp_std"] >= 0)


def test_hmf():
    r = _ok(halos.compute_hmf(output_dir=OUT))
    _, cols = read_csv(r.files[0])
    hmf = cols["dn_dlnM_h3_Mpc3"]
    assert np.all(np.diff(hmf) < 0), "HMF must decrease with mass"


def test_hmf_theory_backends():
    r_emu = _ok(halos.compute_hmf(output_dir=OUT))
    r_t08 = _ok(halos.compute_hmf(output_dir=OUT, backend="tinker08",
                                  mass_def="200c"))
    _, c_emu = read_csv(r_emu.files[0])
    _, c_t08 = read_csv(r_t08.files[0])
    # emulator vs Tinker08 at matched 200c definition: ~10% at group/cluster
    # masses (the reviewer's linear-theory-vs-emulator overlay, done right)
    sel = c_emu["M200c_Msun_per_h"] < 3e14
    ratio = c_t08["dn_dlnM_h3_Mpc3"][sel] / c_emu["dn_dlnM_h3_Mpc3"][sel]
    assert np.all(np.abs(ratio - 1) < 0.25), "tinker08 vs miratitan deviates >25%"
    # FoF fits run with mass_def='fof' and refuse SO masses
    _ok(halos.compute_hmf(output_dir=OUT, backend="sheth_tormen", mass_def="fof"))
    with pytest.raises(ValueError, match="friends-of-friends"):
        halos.compute_hmf(output_dir=OUT, backend="press_schechter",
                          mass_def="200c")
    with pytest.raises(ValueError, match="M200c only"):
        halos.compute_hmf(output_dir=OUT, backend="miratitan", mass_def="500c")
    # the overlay the reviewer wanted: emulator + theory in one figure
    _ok(analysis.plot_emulator_curves(
        curve_files=[r_emu.files[0], r_t08.files[0]], output_dir=OUT,
        title="HMF: emulator vs Tinker08"))


def test_cluster_gas():
    r = _ok(halos.predict_cluster_gas_params(output_dir=OUT))
    assert len(r.metadata["gas_model_params"]) >= 4


def test_lya_p1d():
    r = _ok(igm.emulate_lya_p1d(output_dir=OUT))
    _, cols = read_csv(r.files[0])
    assert np.all(cols["P1D_Mpc"] > 0)
    assert cols["P1D_Mpc"][0] > cols["P1D_Mpc"][-1]  # decreasing with k


# ---------------- feedback-driven behaviors (Aug 17 review) ----------------

def test_return_data_inline():
    r = _ok(pk.compute_nonlinear_pk(output_dir=OUT, backend="syren_halofit",
                                    n_points=500, return_data=True))
    data = r.metadata["data"]
    assert len(data["k_h_per_Mpc"]) <= 80
    assert len(data["k_h_per_Mpc"]) == len(data["Pk_Mpc_over_h_cubed"])
    assert "stats" in r.metadata and "max_Pk" in r.metadata["stats"]


def test_in_training_box_flag():
    r = _ok(pk.compute_nonlinear_pk(output_dir=OUT, backend="gokunemu"))
    assert r.metadata["in_training_box"] is True
    # Ob=0.06 is inside the union schema but OUTSIDE gokunemu's box (<=0.055)
    r = _ok(pk.compute_nonlinear_pk(output_dir=OUT, backend="gokunemu", Ob=0.06))
    assert r.metadata["in_training_box"] is False
    assert any("Ob" in w for w in r.metadata["extrapolation_warnings"])
    assert "WARNING" in r.message


def test_plot_refuses_mixed_quantities():
    pk_file = pk.compute_nonlinear_pk(output_dir=OUT, backend="syren_halofit").files[0]
    sup_file = baryons.compute_baryon_suppression(output_dir=OUT, model="spk").files[0]
    with pytest.raises(ValueError, match="different quantities"):
        analysis.plot_emulator_curves(curve_files=[pk_file, sup_file], output_dir=OUT)
    _ok(analysis.plot_emulator_curves(curve_files=[pk_file, sup_file], output_dir=OUT,
                                      allow_mixed_quantities=True))


def test_compose_spectra():
    pk_file = pk.compute_nonlinear_pk(output_dir=OUT, backend="syren_halofit",
                                      k_min=0.1, k_max=8.0).files[0]
    sup_file = baryons.compute_baryon_suppression(output_dir=OUT, model="spk").files[0]
    r = _ok(pk.compose_spectra(spectrum_files=[pk_file, sup_file],
                               output_dir=OUT, op="multiply", return_data=True))
    _, cols = read_csv(r.files[0])
    _, pk_cols = read_csv(pk_file)
    assert np.all(cols["value"] <= np.interp(cols["k_h_per_Mpc"],
                                             pk_cols["k_h_per_Mpc"],
                                             pk_cols["Pk_Mpc_over_h_cubed"]) + 1e-6)
    # P(k) x S(k) is still a power spectrum (plots on P(k) axes)
    assert r.metadata["output_quantity"] == "power_spectrum"


def test_baryonify_pk():
    pk_file = pk.compute_nonlinear_pk(output_dir=OUT, backend="baccoemu",
                                      k_min=0.1, k_max=4.5).files[0]
    r = _ok(baryons.baryonify_pk(pk_file=pk_file, output_dir=OUT, model="spk"))
    _, cols = read_csv(r.files[0])
    assert np.all(cols["suppression"] <= 1.001)
    assert "suppression_stats" in r.metadata


def test_labels_and_filenames_vary():
    r1 = _ok(halos.compute_hmf(output_dir=OUT, sigma_8=0.8))
    r2 = _ok(halos.compute_hmf(output_dir=OUT, sigma_8=0.75))
    assert r1.files[0] != r2.files[0], "parameter scans must not overwrite"
    h2, _ = read_csv(r2.files[0])
    assert "sigma_8=0.75" in h2["label"]


def test_hmf_reproducible():
    r1 = halos.compute_hmf(output_dir=OUT, random_seed=7)
    r2 = halos.compute_hmf(output_dir=OUT, random_seed=7)
    _, c1 = read_csv(r1.files[0])
    _, c2 = read_csv(r2.files[0])
    assert np.array_equal(c1["emulator_rel_std"], c2["emulator_rel_std"])


def test_convert_cosmology():
    r = _ok(meta.convert_cosmology(sigma8=0.81))
    conv = r.metadata["conversions"]
    assert abs(conv["hmf_miratitan"]["Ommh2"] - 0.31 * 0.67**2) < 1e-4
    assert 2.7 < conv["cmb"]["ln10As"] < 3.3
    # round-trip: feed the derived As back, sigma8 must come back
    r2 = _ok(meta.convert_cosmology(As=conv["pk_and_lensing"]["As"]))
    assert abs(r2.metadata["conversions"]["pk_and_lensing"]["sigma8"] - 0.81) < 0.01
    r3 = _ok(meta.convert_cosmology(sigma8=0.81, w0=-0.9))
    assert any("CANNOT" in n for n in r3.metadata["notes"])



# ---------------- usage-driven additions (Oct 2026 review) ----------------

def test_boost_times_suppression_crossings():
    b = gravity.compute_mg_boost(output_dir=OUT, model="fofr",
                                 minus_log10_fR0=5.0).files[0]
    sup = baryons.compute_baryon_suppression(output_dir=OUT, model="spk",
                                             k_min=0.03, k_max=5).files[0]
    r = _ok(pk.compose_spectra(spectrum_files=[b, sup], output_dir=OUT))
    assert r.metadata["output_quantity"] == "composed"
    assert "unity_crossings_k" in r.metadata
    # the figure for B, S, and B*S together needs no override flag
    r = _ok(analysis.plot_emulator_curves(curve_files=[b, sup, r.files[0]],
                                          output_dir=OUT))
    assert r.metadata["axes"]["logy"] is False


def test_bacco_k_max_message():
    with pytest.raises(ValueError, match="exceeds the baccoemu emulator limit"):
        pk.compute_nonlinear_pk(output_dir=OUT, backend="baccoemu", k_max=10)


def test_subgrid_real_x_axis():
    r = _ok(baryons.emulate_subgrid_statistic(output_dir=OUT, statistic="Pk"))
    _, cols = read_csv(r.files[0])
    k = cols["k_h_per_Mpc"]
    assert len(k) == 255 and np.all(np.diff(k) > 0)
    assert 0.05 < k[0] < 0.1 and 10 < k[-1] < 15
    r = _ok(baryons.emulate_subgrid_statistic(output_dir=OUT, statistic="fGas"))
    assert r.metadata["x_axis"]["column"] == "Mhalo_Msun"


def test_scan_emulator_parameters():
    r = _ok(analysis.scan_emulator_parameters(
        tool="compute_baryon_suppression", output_dir=OUT,
        scan={"z": [0.0, 0.5, 1.0]}, fixed_args={"model": "syren_IllustrisTNG"}))
    assert len(r.metadata["runs"]) == 3 and not r.metadata["errors"]
    assert r.metadata["figures"] and r.files[0].endswith(".png")
    # two parameters -> one figure per value of the second
    r = _ok(analysis.scan_emulator_parameters(
        tool="compute_mg_boost", output_dir=OUT,
        scan={"minus_log10_fR0": [4.5, 5.5], "z": [0.0, 1.0]}))
    assert len(r.metadata["runs"]) == 4 and len(r.metadata["figures"]) == 2
    # zip mode = paired finite-difference steps
    r = _ok(analysis.scan_emulator_parameters(
        tool="compute_lensing_cls", output_dir=OUT, combine="zip",
        scan={"Om": [0.30, 0.32], "sigma8": [0.80, 0.82]},
        fixed_args={"n_ell": 10}, plot=False))
    assert len(r.metadata["runs"]) == 2
    with pytest.raises(ValueError, match="has no argument"):
        analysis.scan_emulator_parameters(tool="compute_mg_boost",
                                          output_dir=OUT, scan={"nope": [1]})
    # a failing point is reported, not fatal
    r = _ok(analysis.scan_emulator_parameters(
        tool="compute_nonlinear_pk", output_dir=OUT, plot=False,
        scan={"k_max": [4.0, 9.0]}, fixed_args={"backend": "baccoemu"}))
    assert len(r.metadata["runs"]) == 1 and len(r.metadata["errors"]) == 1


# ---------------- skills must match the server they describe ----------------

def test_skills_reference_real_tools():
    import re
    from mcp_server.server import create_server
    import asyncio
    names = {t.name for t in asyncio.run(create_server().list_tools())}
    from tools.meta.skills import skill_files
    for path in skill_files():
        text = path.read_text()
        called = set(re.findall(r"`([a-z_]+)(?:\(|`)", text))
        called |= set(re.findall(r"\b((?:compute|emulate|predict|plot|scan|load|list|describe|convert|compose|baryonify|export|set|get)_[a-z_]+)\(", text))
        tool_like = {c for c in called if re.match(
            r"(compute|emulate|predict|plot|scan|load|list|describe|convert|compose|baryonify|export|set|get|auth)_?", c)
            and "_" in c}
        unknown = tool_like - names - {"run_pack_kernel", "pip_deps_by_backend",
                                       "base_pip_deps", "pack_path"}
        assert not unknown, f"{path.name} names non-tools: {sorted(unknown)}"


def test_compose_refuses_mixed_redshift():
    b = gravity.compute_mg_boost(output_dir=OUT, z=0.0).files[0]
    s = baryons.compute_baryon_suppression(output_dir=OUT, model="syren_SIMBA",
                                           z=0.5, k_min=0.03, k_max=4.9).files[0]
    with pytest.raises(ValueError, match="different redshifts"):
        pk.compose_spectra(spectrum_files=[b, s], output_dir=OUT)


def test_skill_recipes_run():
    """The scan calls written in the skills, executed as written."""
    cosmo = {"Om": 0.31, "Ob": 0.049, "h": 0.67, "ns": 0.965, "sigma8": 0.81}
    # baryon-budget part 1
    r = _ok(analysis.scan_emulator_parameters(
        tool="compute_baryon_suppression", output_dir=OUT,
        scan={"model": ["spk", "bacco", "syren_IllustrisTNG", "syren_Astrid",
                        "syren_SIMBA", "syren_Swift_EAGLE"]},
        fixed_args=dict(cosmo, k_min=0.1, k_max=4.9, n_points=200)))
    assert len(r.metadata["runs"]) == 6 and not r.metadata["errors"]
    # baryon-budget part 2
    r = _ok(analysis.scan_emulator_parameters(
        tool="emulate_subgrid_statistic", output_dir=OUT, combine="zip",
        scan={"v_kin": [0.2, 1.0], "e_kin": [0.1, 1.0]},
        fixed_args={"statistic": "Pk"}))
    assert len(r.metadata["runs"]) == 2
    # mg-baryon-degeneracy: B and S grids, then pairwise composition
    grid = dict(k_min=0.03, k_max=4.9, n_points=200)
    rb = _ok(analysis.scan_emulator_parameters(
        tool="compute_mg_boost", output_dir=OUT,
        scan={"minus_log10_fR0": [4, 5, 6], "z": [0, 0.5]},
        fixed_args=dict(cosmo, **grid)))
    rs = _ok(analysis.scan_emulator_parameters(
        tool="compute_baryon_suppression", output_dir=OUT,
        scan={"model": ["bacco", "syren_IllustrisTNG"], "z": [0, 0.5]},
        fixed_args=dict(cosmo, **grid)))
    b = next(x for x in rb.metadata["runs"] if x["params"] == {"minus_log10_fR0": 5, "z": 0.5})
    s = next(x for x in rs.metadata["runs"] if x["params"] == {"model": "bacco", "z": 0.5})
    c = _ok(pk.compose_spectra(spectrum_files=[b["file"], s["file"]], output_dir=OUT))
    assert "unity_crossings_k" in c.metadata
    # lensing-fisher-inputs: central differences in one zip scan
    r = _ok(analysis.scan_emulator_parameters(
        tool="compute_lensing_cls", output_dir=OUT, combine="zip",
        scan={"Om": [0.31, 0.30, 0.32, 0.31, 0.31],
              "sigma8": [0.81, 0.81, 0.81, 0.80, 0.82]},
        fixed_args={"z_source": 1.0, "ell_min": 100, "ell_max": 5000, "n_ell": 80}))
    ells = [read_csv(x["file"])[1]["ell"] for x in r.metadata["runs"]]
    assert all(np.array_equal(ells[0], e) for e in ells)
    # pk-crosscheck
    r = _ok(analysis.scan_emulator_parameters(
        tool="compute_nonlinear_pk", output_dir=OUT, plot=False,
        scan={"backend": ["baccoemu", "euclidemu2", "csst", "gokunemu",
                          "miratitan", "camb_hmcode", "syren_halofit"]},
        fixed_args={"k_min": 0.01, "k_max": 4.5, "n_points": 200, "sigma8": 0.81}))
    assert not r.metadata["errors"], r.metadata["errors"]

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
