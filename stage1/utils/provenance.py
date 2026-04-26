"""Runtime provenance helper for Stage 1 manifests.

Builds the canonical block embedded in every Stage 1 / Phase A / Phase B /
Phase C manifest so a run is traceable and reproducible offline:

- ``git_sha``                 — repo HEAD at run time
- ``python_version``          — interpreter version
- ``torch_version``           — torch.__version__ (or "unavailable")
- ``transformers_version``    — transformers.__version__ (or "unavailable")
- ``numpy_version``           — numpy.__version__ (or "unavailable")
- ``platform``                — platform.platform()
- ``hostname``                — gethostname() — useful for distinguishing runs
                                across machines (e.g. RunPod vs local)
- ``command``                 — sys.argv list (the exact CLI invocation)
- ``cwd``                     — working directory at run time
- ``timestamp``                — ISO-8601 wall-clock at build time
- ``dataset``                 — dataset provenance block (revision + sha256)
                                propagated from ``config.dataset._provenance``
                                if present, else built from configured pins.

The helper deliberately swallows lookup errors and stamps "unavailable" so a
broken environment query never aborts the parent run before it can write its
manifest.
"""

from __future__ import annotations

import logging
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _safe_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def _git_sha(repo_root: Optional[str] = None) -> str:
    """Return ``HEAD`` SHA for the repo containing this file (best-effort)."""
    if repo_root is None:
        # provenance.py is at <repo>/stage1/utils/provenance.py — climb two.
        repo_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def build_runtime_provenance(
    *,
    config: Optional[Any] = None,
    config_path: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the runtime provenance block for embedding in a manifest.

    Args:
        config: Optional Stage1Config instance — used to surface dataset
            provenance (revision + realised SHA-256 stamped by the loader).
        config_path: Optional path to the YAML used for this run — embedded
            verbatim alongside the parsed dataset block.
        extra: Optional caller-supplied fields merged into the result.

    Returns:
        Dict suitable for serialising as ``runtime_provenance`` in a manifest.
    """
    block: Dict[str, Any] = {
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "torch_version": _safe_version("torch"),
        "transformers_version": _safe_version("transformers"),
        "numpy_version": _safe_version("numpy"),
        "scipy_version": _safe_version("scipy"),
        "platform": platform.platform(),
        "hostname": _hostname(),
        "command": list(sys.argv),
        "cwd": os.getcwd(),
        "timestamp": datetime.now().isoformat(),
        "config_path": config_path,
    }
    # Surface dataset provenance from the loader's stamp (preferred) or the
    # configured pin (fallback). Either is enough to reproduce the bytes.
    if config is not None:
        prov = getattr(config.dataset, "_provenance", None)
        if prov is not None:
            block["dataset"] = dict(prov)
        else:
            block["dataset"] = {
                "name": getattr(config.dataset, "name", None),
                "lang": getattr(config.dataset, "lang", None),
                "split": getattr(config.dataset, "split", None),
                "revision": getattr(config.dataset, "revision", None),
                "expected_sha256_pin": getattr(
                    config.dataset, "expected_sha256", None,
                ),
                "_note": "Loader provenance unavailable; pin echoed from config.",
            }
    if extra:
        block.update(extra)
    return block
