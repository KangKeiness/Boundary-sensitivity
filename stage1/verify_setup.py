"""Setup verification script for Stage 1 pipeline.

Run this first on RunPod to confirm the environment is ready:
    python stage1/verify_setup.py
"""

import sys


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
    from datasets import load_dataset
    ds = load_dataset("juletxara/mgsm", "te", split="test")
    sample = ds[0]
    print(f"  question : {sample['question'][:60]}...")
    print(f"  answer   : {sample['answer']}")


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
