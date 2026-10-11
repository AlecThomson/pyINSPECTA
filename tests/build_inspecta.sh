#!/usr/bin/env bash
# Build INSPECTA (sdhdf_tools) for interoperability testing on Debian/Ubuntu.
#
# Usage: tests/build_inspecta.sh [PREFIX]
#
# Installs the build dependencies with apt (needs root), builds the CALCEPH C
# library from its PyPI source distribution, then builds INSPECTA into PREFIX
# (default: ./inspecta). Afterwards, run the INSPECTA tests with:
#
#   SDHDF_INSPECTA_BIN=PREFIX/bin SDHDF_RUNTIME=PREFIX/runtime pytest tests/test_inspecta.py
set -euo pipefail

PREFIX=$(realpath -m "${1:-inspecta}")
SDHDF_TOOLS_REPO=${SDHDF_TOOLS_REPO:-https://github.com/too043/sdhdf_tools.git}
SDHDF_TOOLS_REF=${SDHDF_TOOLS_REF:-6d7380c}
CALCEPH_VERSION=${CALCEPH_VERSION:-5.0.1}
BUILD=$(mktemp -d)
trap 'rm -rf "$BUILD"' EXIT

SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
$SUDO apt-get update -qq
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  build-essential gfortran autoconf automake libtool cmake pkg-config curl git \
  libhdf5-dev libcfitsio-dev liberfa-dev pgplot5

# CALCEPH is not packaged for Ubuntu; its PyPI sdist contains the C library.
url=$(curl -fsSL "https://pypi.org/pypi/calcephpy/${CALCEPH_VERSION}/json" |
  python3 -c "import json,sys; print([u['url'] for u in json.load(sys.stdin)['urls'] if u['packagetype'] == 'sdist'][0])")
curl -fsSL "$url" | tar xz -C "$BUILD"
cmake -S "$BUILD/calcephpy-${CALCEPH_VERSION}" -B "$BUILD/calceph-build" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DENABLE_FORTRAN=OFF -DENABLE_PYTHON=OFF \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD/calceph-build" -j "$(nproc)"
cmake --install "$BUILD/calceph-build"

git clone --quiet "$SDHDF_TOOLS_REPO" "$BUILD/sdhdf_tools"
git -C "$BUILD/sdhdf_tools" checkout --quiet "$SDHDF_TOOLS_REF"
cd "$BUILD/sdhdf_tools/inspecta"
autoreconf --install --force
./configure --prefix="$PREFIX" \
  CPPFLAGS="-I/usr/include/hdf5/serial -I$PREFIX/include" \
  LDFLAGS="-L/usr/lib/x86_64-linux-gnu/hdf5/serial -L$PREFIX/lib -Wl,-rpath,$PREFIX/lib"
make -j "$(nproc)"
make install
rm -rf "$PREFIX/runtime"
cp -r runtime "$PREFIX/runtime"

echo
echo "INSPECTA installed in $PREFIX"
echo "export SDHDF_INSPECTA_BIN=$PREFIX/bin SDHDF_RUNTIME=$PREFIX/runtime"
