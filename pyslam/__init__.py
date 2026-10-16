"""pySLAM package bootstrap.

The project builds a few native extensions inside the repository tree (for example
``thirdparty/g2opy/lib/g2o.cpython-311-darwin.so``). When pySLAM is imported from a
source checkout, those build artifacts are not installed into the active Python site
packages, so we add the repository-local extension directories to ``sys.path``
explicitly before optional modules such as ``g2o`` are imported.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _append_repo_libs() -> None:
    root = Path(__file__).resolve().parent.parent
    candidate_paths = [
        root,
        root / "thirdparty" / "pangolin",
        root / "thirdparty" / "g2opy" / "lib",
        root / "cpp" / "lib",
        root / "thirdparty" / "gtsam_factors" / "lib",
        root / "thirdparty" / "pydbow3" / "lib",
        root / "thirdparty" / "pydbow2" / "lib",
        root / "thirdparty" / "pyibow" / "lib",
        root / "thirdparty" / "ros2_pybindings" / "lib",
    ]
    for path in candidate_paths:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


_append_repo_libs()

__all__ = ["_append_repo_libs"]
