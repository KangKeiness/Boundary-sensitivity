"""Setup verification script for Stage 1 pipeline.

Run this first on RunPod to confirm the environment is ready:
    python stage1/verify_setup.py
"""

import sys
import os

# Resolve imports from stage1/
sys.path.insert(0, os.path.dirname(__file__))


def check_imports():
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


def check_cuda():
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
        print("  CUDA not available — running on CPU")


def check_dataset():
    print("Checking MGSM dataset (1 sample)...")
    from utils.config import load_config
    from data.loader import load_mgsm
    cfg_path = os.path.join(os.path.dirname(__file__), "configs", "stage1_main.yaml")
    config = load_config(cfg_path)
    config.dataset.debug_n = 1
    samples = load_mgsm(config)
    s = samples[0]
    print(f"  sample_id    : {s['sample_id']}")
    print(f"  gold_answer  : {s['gold_answer']}")
    print(f"  prompt[:80]  : {s['prompt'][:80]}...")


def main():
    steps = [check_imports, check_cuda, check_dataset]
    for step in steps:
        try:
            step()
        except Exception as e:
            print(f"\nFAILED at {step.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    print("\nSetup OK")


if __name__ == "__main__":
    main()
