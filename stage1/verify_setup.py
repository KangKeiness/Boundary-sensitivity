"""Setup verification script for Stage 1 pipeline.

Run this first on RunPod (or any new environment) to confirm the runtime is
ready. Smoke-checks imports, CUDA, MGSM dataset access (with SHA-256
verification), and reports the full provenance block that will be written
into manifests.

Usage:
    python -m stage1.verify_setup
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Resolve repo root for absolute (`stage1.*`) imports when invoked as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def check_imports() -> None:
    print("Checking imports...")
    import transformers
    import datasets
    import scipy
    import yaml
    import accelerate
    import numpy as np
    print(f"  transformers : {transformers.__version__}")
    print(f"  datasets     : {datasets.__version__}")
    print(f"  scipy        : {scipy.__version__}")
    print(f"  pyyaml       : {yaml.__version__}")
    print(f"  accelerate   : {accelerate.__version__}")
    print(f"  numpy        : {np.__version__}")


def check_python_version() -> None:
    """Fail fast on unsupported Python versions for the Stage 1 runtime."""
    print("Checking Python runtime...")
    v = sys.version_info
    print(f"  python       : {v.major}.{v.minor}.{v.micro}")
    if (v.major, v.minor) < (3, 10) or (v.major, v.minor) >= (3, 13):
        raise RuntimeError(
            "Stage 1 runtime is validated on Python 3.10-3.12. "
            f"Current interpreter is {v.major}.{v.minor}. "
            "Create a 3.12 virtualenv (e.g. `py -3.12 -m venv .venv312`) "
            "and reinstall requirements."
        )


def check_cuda() -> None:
    print("Checking CUDA...")
    import torch
    print(f"  torch        : {torch.__version__}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024 ** 3
            print(f"  GPU {i}: {props.name}  {mem_gb:.1f} GB")
    else:
        print("  CUDA not available - running on CPU")


def check_dataset() -> None:
    """Resolve, download (if needed), SHA-256 verify, and report provenance.

    Stage 1 hardening (2026-04-25): this previously loaded only the first
    sample. We now reuse the production loader so the SHA-256 pin is
    exercised end-to-end exactly the way Phase A does it.
    """
    print("Checking MGSM dataset (1 sample) with SHA-256 verification...")
    from stage1.utils.config import load_config
    from stage1.data.loader import load_mgsm
    cfg_path = _REPO_ROOT / "stage1" / "configs" / "stage1_main.yaml"
    config = load_config(str(cfg_path))
    config.dataset.debug_n = 1
    samples = load_mgsm(config)
    s = samples[0]
    print(f"  sample_id    : {s['sample_id']}")
    print(f"  gold_answer  : {s['gold_answer']}")
    print(f"  prompt[:80]  : {s['prompt'][:80]}...")
    prov = getattr(config.dataset, "_provenance", None)
    if prov is not None:
        print(f"  revision     : {prov['revision']}")
        print(f"  sha256       : {prov['sha256']}")
        print(f"  expected pin : {prov['expected_sha256_pin']}")
        if prov["expected_sha256_pin"] is None:
            print("  WARNING      : no expected_sha256 pin configured")


def check_provenance_block() -> None:
    print("Building runtime provenance block...")
    from stage1.utils.provenance import build_runtime_provenance
    block = build_runtime_provenance(
        config=None, config_path=None,
    )
    print(f"  git_sha             : {block['git_sha']}")
    print(f"  python_version      : {block['python_version']}")
    print(f"  torch_version       : {block['torch_version']}")
    print(f"  transformers_version: {block['transformers_version']}")
    print(f"  hostname            : {block['hostname']}")


def main() -> None:
    steps = [
        check_python_version,
        check_imports,
        check_cuda,
        check_dataset,
        check_provenance_block,
    ]
    for step in steps:
        try:
            step()
        except Exception as e:
            print(f"\nFAILED at {step.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    print("\nSetup OK")


if __name__ == "__main__":
    main()
