"""Pytest configuration — stub out unavailable heavy dependencies.

In the CI/sandbox environment, transformers and torch may not be installed.
We provide lightweight stubs so that unit tests covering only the pure-Python
logic (seed formulas, condition parsers, RNG behaviour) can run without
loading any model weights.

This conftest is ONLY active for stage1/tests/.  It does NOT affect
production imports in stage1/.
"""

import sys
import types
import unittest.mock as mock


def _ensure_stub(module_name: str) -> types.ModuleType:
    """Return existing module or install a MagicMock stub."""
    if module_name not in sys.modules:
        stub = mock.MagicMock(name=module_name)
        stub.__name__ = module_name
        sys.modules[module_name] = stub
    return sys.modules[module_name]


# Stub top-level packages that may be absent in the sandbox
_heavy = [
    "transformers",
    "transformers.AutoModelForCausalLM",
    "transformers.AutoTokenizer",
]

_already_present = "transformers" in sys.modules

if not _already_present:
    # Install stubs before any test module imports stage1.models.composer
    _tf = types.ModuleType("transformers")
    _tf.AutoModelForCausalLM = mock.MagicMock(name="AutoModelForCausalLM")
    _tf.AutoTokenizer = mock.MagicMock(name="AutoTokenizer")
    sys.modules["transformers"] = _tf
