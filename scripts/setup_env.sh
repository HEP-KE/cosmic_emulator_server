#!/usr/bin/env bash
# Recreate the cosmic-emu Python environment from scratch.
#
# Usage:  bash scripts/setup_env.sh [env-name]
#
# Installs every emulator selected in EMULATOR_CATALOG.md waves 1-5 into one
# conda env, applying the three vendored patches documented in ENVIRONMENT.md
# (SEPIA scipy>=1.11 fix, csstemu numpy-2 fix, LaCE GPy import fix).
set -euo pipefail

ENV_NAME="${1:-cosmic-emu}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXTERNAL="$REPO_DIR/external"

conda create -n "$ENV_NAME" python=3.12 -y
PIP="$(conda info --base)/envs/$ENV_NAME/bin/pip"
PYTHON="$(conda info --base)/envs/$ENV_NAME/bin/python"

$PIP install --upgrade pip
# pkg_resources still needed by jax_cosmo and CubicGalileonEmu
$PIP install "setuptools<81"

# --- base scientific stack + pip-clean emulators -------------------------
$PIP install numpy scipy matplotlib h5py scikit-learn pandas \
    camb pyspk MiraTitanHMFemulator emantis

# --- JAX cluster ---------------------------------------------------------
# NOTE: jaxcapse pins jax==0.4.x; install it first so the resolver settles,
# then the rest against that jax.
$PIP install jaxcapse
$PIP install jax-cosmo cosmopower-jax baccoemu jaxeffort

# --- PyTorch cluster (CPU wheels) ---------------------------------------
$PIP install torch --index-url https://download.pytorch.org/whl/cpu
$PIP install gokunemu

# --- TensorFlow cluster --------------------------------------------------
$PIP install py21cmemu

# --- compiled: EuclidEmulator2 needs GSL; conda-forge GSL provides the
#     libgsl.28.dylib/.so the PyPI wheel links against -------------------
conda install -n "$ENV_NAME" -c conda-forge gsl -y
$PIP install euclidemu2

# --- classy_sz (emulated CLASS + tSZ; data downloads on first use) ------
$PIP install classy_sz

# --- git-installed emulators --------------------------------------------
$PIP install git+https://github.com/DeaglanBartlett/symbolic_pofk
$PIP install git+https://github.com/BartolomeoF/nDGPemu
$PIP install "pybird-lss"
$PIP install git+https://github.com/fkeruzore/picasso
$PIP install GPy

# subgrid_emu: metadata pins python<3.12 and an old numpy — install without
# deps (sklearn/pandas already present). TODO upstream: relax the pins.
$PIP install --no-deps --ignore-requires-python git+https://github.com/nesar/subgrid_emu
$PIP install --no-deps git+https://github.com/nesar/CubicGalileonEmu

# --- vendored patched clones (see ENVIRONMENT.md for the diffs) ---------
mkdir -p "$EXTERNAL"

if [ ! -d "$EXTERNAL/SEPIA" ]; then
    git clone --depth 1 https://github.com/lanl/SEPIA "$EXTERNAL/SEPIA"
    # scipy >= 1.11 removed solve(sym_pos=True); assume_a='pos' is identical
    grep -rl "sym_pos=True" "$EXTERNAL/SEPIA/sepia" | \
        xargs sed -i.bak "s/sym_pos=True/assume_a='pos'/g"
fi
$PIP install -e "$EXTERNAL/SEPIA"

if [ ! -d "$EXTERNAL/csstemu" ]; then
    git clone https://github.com/czymh/csstemu "$EXTERNAL/csstemu"
    # numpy 2 refuses assigning a shape-(1,) array into a scalar slot
    grep -rl "\.predict(Normcosmo)$" "$EXTERNAL/csstemu/CEmulator" | \
        xargs sed -i.bak "s/\.predict(Normcosmo)$/.predict(Normcosmo)[0]/"
fi
$PIP install -e "$EXTERNAL/csstemu"

if [ ! -d "$EXTERNAL/LaCE" ]; then
    git clone --depth 1 https://github.com/igmhub/LaCE "$EXTERNAL/LaCE"
    # gp_emulator.py imports GPy inside __init__ but uses it at module scope
    sed -i.bak "s/^import lace$/import lace\nimport GPy/" \
        "$EXTERNAL/LaCE/lace/emulator/gp_emulator.py"
fi
$PIP install -e "$EXTERNAL/LaCE"
$PIP install git+https://github.com/igmhub/ForestFlow

# --- the server itself ---------------------------------------------------
$PIP install -e "$REPO_DIR"

echo
echo "Environment '$ENV_NAME' ready. Validate with:"
echo "  $PYTHON $REPO_DIR/tests/smoke_env.py"
