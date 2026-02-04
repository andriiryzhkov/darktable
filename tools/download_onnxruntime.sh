#!/bin/bash
#
# Download ONNX Runtime pre-built binaries from GitHub releases.
# Places the result in src/external/onnxruntime/ ready for CMake.
#
# The GitHub release includes CoreML support on macOS and
# CUDA/TensorRT support on Linux (via the -gpu variants).
#
# Usage:  ./tools/download_onnxruntime.sh
#

set -euo pipefail

ONNXRUNTIME_VERSION="1.23.2"

# ---------------------------------------------------------------------------
# Detect platform and architecture
# ---------------------------------------------------------------------------
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Darwin)
    case "$ARCH" in
      arm64)  PACKAGE="onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}.tgz" ;;
      x86_64) PACKAGE="onnxruntime-osx-x86_64-${ONNXRUNTIME_VERSION}.tgz" ;;
      *)      echo "Error: unsupported macOS architecture: $ARCH"; exit 1 ;;
    esac
    ;;
  Linux)
    case "$ARCH" in
      x86_64)  PACKAGE="onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" ;;
      aarch64) PACKAGE="onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz" ;;
      *)       echo "Error: unsupported Linux architecture: $ARCH"; exit 1 ;;
    esac
    ;;
  MINGW*|MSYS*|CYGWIN*)
    case "$ARCH" in
      x86_64|AMD64) PACKAGE="onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip" ;;
      aarch64|ARM64) PACKAGE="onnxruntime-win-arm64-${ONNXRUNTIME_VERSION}.zip" ;;
      *)             echo "Error: unsupported Windows architecture: $ARCH"; exit 1 ;;
    esac
    ;;
  *)
    echo "Error: unsupported OS: $OS"
    exit 1
    ;;
esac

URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${PACKAGE}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../src" && pwd)"
DEST_DIR="$SRC_DIR/external/onnxruntime"
TMPDIR="$(mktemp -d)"

cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

echo "ONNX Runtime ${ONNXRUNTIME_VERSION}"
echo "Package:     ${PACKAGE}"
echo "Destination: ${DEST_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
DOWNLOAD_PATH="$TMPDIR/$PACKAGE"

echo "Downloading ${URL} ..."
if command -v curl &>/dev/null; then
  curl -L -o "$DOWNLOAD_PATH" "$URL"
elif command -v wget &>/dev/null; then
  wget -O "$DOWNLOAD_PATH" "$URL"
else
  echo "Error: neither curl nor wget found"
  exit 1
fi

# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------
EXTRACT_DIR="$TMPDIR/extracted"
mkdir -p "$EXTRACT_DIR"

echo "Extracting ..."
case "$PACKAGE" in
  *.tgz|*.tar.gz)
    tar xzf "$DOWNLOAD_PATH" -C "$EXTRACT_DIR"
    ;;
  *.zip)
    unzip -q "$DOWNLOAD_PATH" -d "$EXTRACT_DIR"
    ;;
esac

# The archive contains a single top-level directory (e.g. onnxruntime-osx-arm64-1.23.2/)
INNER_DIR="$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d | head -1)"
if [ -z "$INNER_DIR" ]; then
  echo "Error: unexpected archive structure"
  exit 1
fi

# ---------------------------------------------------------------------------
# Install into src/external/onnxruntime/
# ---------------------------------------------------------------------------
if [ -d "$DEST_DIR" ]; then
  echo "Removing existing ${DEST_DIR} ..."
  rm -rf "$DEST_DIR"
fi

mkdir -p "$DEST_DIR"

# Copy lib/ and include/ (cmake configs live under lib/cmake/)
cp -R "$INNER_DIR/lib" "$DEST_DIR/lib"
cp -R "$INNER_DIR/include" "$DEST_DIR/include"

# The shipped CMake config sets INTERFACE_INCLUDE_DIRECTORIES to
# ${prefix}/include/onnxruntime, but the release puts headers directly in
# include/. Patch the cmake config to match the actual layout rather than
# moving files around.
CMAKE_TARGETS="$DEST_DIR/lib/cmake/onnxruntime/onnxruntimeTargets.cmake"
if [ -f "$CMAKE_TARGETS" ]; then
  if grep -q 'include/onnxruntime' "$CMAKE_TARGETS" && [ ! -d "$DEST_DIR/include/onnxruntime" ]; then
    echo "Patching CMake config: include/onnxruntime -> include ..."
    sed -i.bak 's|/include/onnxruntime|/include|g' "$CMAKE_TARGETS"
    rm -f "$CMAKE_TARGETS.bak"
  fi
fi

echo ""
echo "Done. ONNX Runtime ${ONNXRUNTIME_VERSION} installed to:"
echo "  ${DEST_DIR}"
echo ""
echo "Contents:"
echo "  lib/    -> $(ls "$DEST_DIR/lib/" | head -5 | tr '\n' ' ')..."
echo "  include/ -> $(ls "$DEST_DIR/include/" | wc -l | tr -d ' ') headers"
