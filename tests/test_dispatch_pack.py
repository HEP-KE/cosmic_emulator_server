"""export_dispatch_pack: the client-side HPC handoff contract.

The pack must be stageable by any hep-genesis client that receives it:
sanitized relative paths, engine size caps, and a manifest that names the
kernel entry point and the node-side pip requirements.
"""

import json

from mcp_server.dispatch import _PACK_MAX_BYTES, _PACK_MAX_FILES, export_dispatch_pack


def test_pack_shape_and_caps():
    pack = export_dispatch_pack()["dispatch_pack"]
    assert pack["name"] == "tools"
    assert 0 < pack["file_count"] <= _PACK_MAX_FILES
    assert 0 < pack["bytes"] <= _PACK_MAX_BYTES
    assert pack["file_count"] == len(pack["files"])
    # Serializable as one JSON tool result.
    json.dumps(pack)


def test_paths_are_safe_and_rooted():
    files = export_dispatch_pack()["dispatch_pack"]["files"]
    for rel in files:
        assert rel.startswith("tools/"), rel
        assert not rel.startswith("/"), rel
        parts = rel.split("/")
        assert all(p and p != ".." and not p.startswith(".") for p in parts), rel
        assert "__pycache__" not in rel, rel
    # The kernels that dispatch actually uses must ship.
    assert "tools/pk/backends.py" in files
    assert "tools/common.py" in files
    assert "tools/pk/__init__.py" in files


def test_manifest_matches_kernels():
    from tools.pk.backends import DISPATCH_PIP_DEPS

    pk = export_dispatch_pack()["dispatch_pack"]["kernels"]["pk"]
    assert pk["function"] == "pk.backends.compute_pk"
    assert pk["pip_deps_by_backend"] == {b: list(d) for b, d in DISPATCH_PIP_DEPS.items()}
    assert "numpy" in pk["base_pip_deps"]
