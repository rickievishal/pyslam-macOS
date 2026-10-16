"""Repo-local Python startup bootstrap for optional native dependencies.

This project builds some extension modules in the repository tree (for example
``thirdparty/g2opy/lib/g2o.cpython-311-darwin.so``). Those modules are not
installed into site-packages, so the repository root must be made importable
before user code imports them directly.

The bootstrap is intentionally conservative: it only adds existing project-local
library directories to ``sys.path`` and avoids forcing any optional dependency to
load when the corresponding build artifacts are absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _add_existing_paths(paths):
    seen = set(sys.path)
    for raw in paths:
        if not raw:
            continue
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            continue
        string_path = str(path)
        if string_path not in seen:
            sys.path.insert(0, string_path)
            seen.add(string_path)


def _bootstrap():
    root = Path(__file__).resolve().parent
    config_libs = root / "config_libs.yaml"
    candidates = [str(root)]

    if config_libs.exists():
        try:
            import yaml  # local dependency is already required by the project
        except Exception:
            yaml = None

        if yaml is not None:
            with config_libs.open("r", encoding="utf-8") as handle:
                try:
                    data = yaml.safe_load(handle) or {}
                except Exception:
                    data = {}

            for section in ("CORE_LIB_PATHS", "LIB_PATHS"):
                settings = data.get(section, {})
                for rel_path in settings.values():
                    if isinstance(rel_path, str):
                        for token in rel_path.split(","):
                            candidates.append(str(root / token.strip()))
                    elif isinstance(rel_path, (list, tuple)):
                        for token in rel_path:
                            candidates.append(str(root / str(token).strip()))

    candidates.extend(
        [
            str(root / "thirdparty" / "pangolin"),
            str(root / "thirdparty" / "g2opy" / "lib"),
            str(root / "cpp" / "lib"),
            str(root / "thirdparty" / "gtsam_factors" / "lib"),
            str(root / "thirdparty" / "pydbow3" / "lib"),
            str(root / "thirdparty" / "pydbow2" / "lib"),
            str(root / "thirdparty" / "pyibow" / "lib"),
            str(root / "thirdparty" / "ros2_pybindings" / "lib"),
        ]
    )

    _add_existing_paths(candidates)


_bootstrap()
