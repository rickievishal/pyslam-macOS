#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"

usage() {
  cat <<'EOF'
Usage: ./run_macbook_webcam_slam.sh [--semantics | --no-semantics]

  --semantics      Enable semantic segmentation and semantic map fusion.
  --no-semantics   Disable semantic segmentation (the default).

SEMANTIC_ENABLED=true|false may also be used; an explicit command-line option
overrides the environment variable.
EOF
}

SEMANTIC_ENABLED="${SEMANTIC_ENABLED:-false}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --semantics)
      SEMANTIC_ENABLED=true
      ;;
    --no-semantics)
      SEMANTIC_ENABLED=false
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

case "$SEMANTIC_ENABLED" in
  true|TRUE|1|yes|YES|on|ON)
    SEMANTIC_ENABLED=true
    ;;
  false|FALSE|0|no|NO|off|OFF)
    SEMANTIC_ENABLED=false
    ;;
  *)
    echo "Invalid SEMANTIC_ENABLED value: $SEMANTIC_ENABLED (use true or false)." >&2
    exit 2
    ;;
esac

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing .venv. See MACBOOK_M4_SETUP.md for the environment setup command." >&2
  exit 1
fi

if ! "$PYTHON" -c 'import pyslam, pypangolin, glutils'; then
  echo "Viewer dependencies are unavailable. Build the viewer modules with:" >&2
  echo "  cmake --build thirdparty/pangolin/build --target pypangolin --parallel 8" >&2
  echo "  cmake --build cpp/build --target glutils --parallel 4" >&2
  exit 1
fi

CAMERA_ID="${CAMERA_ID:-0}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-30}"

if [[ "$SEMANTIC_ENABLED" == "true" ]]; then
  MPS_AVAILABLE="$($PYTHON -c 'import torch; print(str(torch.backends.mps.is_available()).lower())')"
  if [[ "$MPS_AVAILABLE" == "true" ]]; then
    echo "Semantic device: MPS"
  else
    echo "Semantic device: CPU (MPS unavailable in this PyTorch runtime)"
  fi
else
  echo "Semantic segmentation: disabled"
fi

if ! "$PYTHON" -c "import cv2; cap=cv2.VideoCapture($CAMERA_ID); ok=cap.isOpened(); cap.release(); raise SystemExit(0 if ok else 1)"; then
  echo "Cannot access camera $CAMERA_ID. Grant Camera permission to this terminal/Codex, then retry." >&2
  exit 1
fi

export PYSLAM_USE_CPP="${PYSLAM_USE_CPP:-false}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.matplotlib}"
export PYTHONPATH="$ROOT_DIR/thirdparty/g2opy/lib${PYTHONPATH:+:$PYTHONPATH}"
# Homebrew g2o and pip OpenCV ship separate OpenMP runtimes on macOS.
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export CAMERA_ID WIDTH HEIGHT FPS
if [[ "$SEMANTIC_ENABLED" == "true" ]]; then
  exec "$PYTHON" main_slam.py -c config_macbook_m4_webcam.yaml
fi
exec "$PYTHON" main_slam.py -c config_macbook_m4_webcam_no_semantics.yaml
