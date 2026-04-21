"""Pytest configuration — deterministic transformers stub for analysis tests.

In the CI/sandbox environment, ``transformers`` is often unavailable. Many
unit tests cover only pure-Python logic (seed formulas, parser predicates,
RNG behaviour, anchor-gate decision) and do not need the real package; for
those we install a lightweight stub so test modules that transitively import
``stage1.models.composer`` (which imports transformers at module load) can
still be collected.

v4 PRIORITY 4 — order-independence:

Earlier versions installed the stub conditionally (``if "transformers" not
in sys.modules``) and did not restore the prior state at session end. That
made test behavior depend on what had been imported before pytest started
and on the order in which test modules were collected. In particular,
``test_phase_b_patcher.py`` deleted the stub from ``sys.modules`` at module
load to expose any real installation, and never restored it on failure — so
subsequent test modules collected after it might see a different
``sys.modules`` than tests collected before it.

The fix:

1. The stub is installed at the start of the session, OVERWRITING any
   existing ``transformers`` module. This makes the starting state of every
   test session identical regardless of host environment.
2. A session-scoped autouse fixture snapshots the original ``transformers``
   state and restores it at session end, so we don't pollute outer pytest
   processes (e.g. when this suite is invoked from a parent test runner).
3. Tests that need real ``transformers`` (currently only
   ``test_phase_b_patcher.py``) are responsible for swapping the stub for
   the real package WITH RESTORATION — see that module's import dance.

The ``_ensure_stub`` helper that previously sat here was dead code and has
been removed.
"""

from __future__ import annotations

import sys
import types
import unittest.mock as mock

import pytest


def _build_transformers_stub() -> types.ModuleType:
    stub = types.ModuleType("transformers")
    stub.AutoModelForCausalLM = mock.MagicMock(name="AutoModelForCausalLM")
    stub.AutoTokenizer = mock.MagicMock(name="AutoTokenizer")
    # Mark as a stub so test_phase_b_patcher's dance can detect it cleanly
    # without relying on the absence of __file__.
    stub.__is_test_stub__ = True  # type: ignore[attr-defined]
    return stub


def _snapshot_transformers_modules() -> dict:
    """Return {module_name: module} for every transformers* entry in sys.modules."""
    return {
        k: v for k, v in list(sys.modules.items())
        if k == "transformers" or k.startswith("transformers.")
    }


def _replace_transformers_modules(snapshot: dict) -> None:
    """Atomically replace transformers* entries in sys.modules with ``snapshot``."""
    for k in list(sys.modules):
        if k == "transformers" or k.startswith("transformers."):
            del sys.modules[k]
    sys.modules.update(snapshot)


# ── Eager install (collection-time) ─────────────────────────────────────────
#
# Pytest collects test modules BEFORE running fixtures. Test modules that
# ``import stage1.models.composer`` need a transformers entry at import time.
# We install the stub eagerly here so collection succeeds deterministically;
# the session fixture below restores the prior state at session end.

_PRE_PYTEST_TRANSFORMERS = _snapshot_transformers_modules()
_replace_transformers_modules({"transformers": _build_transformers_stub()})


@pytest.fixture(scope="session", autouse=True)
def _restore_transformers_after_session():
    """Restore the pre-pytest transformers state at session teardown.

    Prevents this suite from polluting an outer process (e.g., a parent
    pytest invocation that already had a real ``transformers`` installed and
    then ran ``stage1/tests/`` as a sub-suite).
    """
    yield
    _replace_transformers_modules(_PRE_PYTEST_TRANSFORMERS)
