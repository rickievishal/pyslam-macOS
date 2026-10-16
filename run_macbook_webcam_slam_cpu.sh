#!/usr/bin/env bash
set -euo pipefail
export PYSLAM_USE_CPP=false
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_macbook_webcam_slam.sh" "$@"
