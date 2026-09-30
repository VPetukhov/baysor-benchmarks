#!/usr/bin/env bash
# Build and install the cellAdmix Python bindings (pinned commit + patches
# from $CELLADMIX_PATCHES, outside git) into the bench environment.
# Idempotent; safe to re-run.
#
# Exact commands and rationale: INSTALL.md
#
# Environment overrides:
#   BENCH_REPO         benchmarks repo root  (default: this script's parent)
#   BENCH_DEPS         dependency prefix     (default $BENCH_REPO/.deps)
#   BAYSOR_BENCH_DATA  benchmark data root   (default $BENCH_REPO/.bench-data)
#   CELLADMIX_PATCHES  patch directory       (default $BAYSOR_BENCH_DATA/celladmix/patches)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_REPO=${BENCH_REPO:-$(cd "$ROOT/.." && pwd)}
DEPS=${BENCH_DEPS:-$BENCH_REPO/.deps}
DATA=${BAYSOR_BENCH_DATA:-$BENCH_REPO/.bench-data}
PATCHES=${CELLADMIX_PATCHES:-$DATA/celladmix/patches}
PY=$DEPS/bench/bin/python
ENV=$DEPS/env
MICROMAMBA=$DEPS/bin/micromamba
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$DEPS/mamba}

COMMIT=7d3fe7ae70c61d2b9e57469d38d9a88fcf6ac14d
REPO=https://github.com/kharchenkolab/cellAdmix-core
SRC=$DATA/cache/celladmix/src
BUILDPREFIX=$DEPS/celladmix-build   # local prefix for extra C++ deps (openjpeg)

# --- 1. extra C++ dependencies (OpenJPEG; everything else is in .deps/env) --
if [[ ! -e "$BUILDPREFIX/lib/pkgconfig/libopenjp2.pc" ]]; then
  echo "=== creating $BUILDPREFIX (openjpeg)"
  "$MICROMAMBA" create -y -p "$BUILDPREFIX" -c conda-forge openjpeg
fi

# --- 2. source at the pinned commit ----------------------------------------
if [[ ! -d "$SRC/.git" ]]; then
  echo "=== cloning $REPO"
  mkdir -p "$(dirname "$SRC")"
  git clone "$REPO" "$SRC"
fi
git -C "$SRC" fetch --quiet origin || true
head=$(git -C "$SRC" rev-parse HEAD)
if [[ "$head" != "$COMMIT" ]]; then
  if [[ -n "$(git -C "$SRC" status --porcelain)" ]]; then
    echo "source tree at $SRC is dirty and not at $COMMIT; remove it and re-run" >&2
    exit 1
  fi
  git -C "$SRC" checkout --quiet "$COMMIT"
fi
echo "=== cellAdmix-core at $(git -C "$SRC" rev-parse --short HEAD)"

# --- 3. patches from outside git (see INSTALL.md) --------------------------
# The four patches are not committed; they live under $CELLADMIX_PATCHES
# (default <data-dir>/celladmix/patches). Patches 0002 + 0003 are also
# proposed upstream in kharchenkolab/cellAdmix-core#2.
if ! compgen -G "$PATCHES/*.patch" > /dev/null; then
  echo "no *.patch files in $PATCHES" >&2
  echo "set CELLADMIX_PATCHES to the directory holding the four patches" >&2
  echo "(see celladmix/INSTALL.md)" >&2
  exit 1
fi
for patch in "$PATCHES"/*.patch; do
  if git -C "$SRC" apply --check "$patch" 2>/dev/null; then
    echo "=== applying $(basename "$patch")"
    git -C "$SRC" apply "$patch"
  elif git -C "$SRC" apply --reverse --check "$patch" 2>/dev/null; then
    echo "=== $(basename "$patch") already applied"
  else
    echo "patch $(basename "$patch") does not apply cleanly" >&2
    exit 1
  fi
done

# --- 4. Python build toolchain in the bench env ----------------------------
"$DEPS/bench/bin/pip" install --quiet scikit-build-core pybind11

# --- 5. build + install the bindings ---------------------------------------
echo "=== building celladmix wheel"
export PATH=$ENV/bin:$PATH
export CC=$ENV/bin/x86_64-conda-linux-gnu-cc
export CXX=$ENV/bin/x86_64-conda-linux-gnu-c++
export PKG_CONFIG_PATH=$BUILDPREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}
PYBIND11_DIR=$("$PY" -m pybind11 --cmakedir)
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$ENV;$BUILDPREFIX -Dpybind11_DIR=$PYBIND11_DIR"
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-6}
"$PY" -m pip install --no-build-isolation "$SRC/python"

# --- 6. verify ---------------------------------------------------------------
"$PY" - <<'EOF'
import celladmix as ca
from celladmix import _core
assert hasattr(_core, "build_tabular_store"), "tabular binding patch missing"
print("celladmix", ca.__version__, "core", _core.core_version(), "OK")
EOF
echo "=== done"
