"""Regression tests for test order-independence (v4 PRIORITY 4).

Verifies that the conftest's transformers stub is installed deterministically
(regardless of host environment) and that ``test_phase_b_patcher.py``'s
real-transformers import dance restores the stub on failure so subsequent
test modules see a consistent ``sys.modules`` state.

These tests do not invoke a sub-pytest — they assert on the post-conditions
that real order-sensitivity bugs would violate.
"""

from __future__ import annotations

import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def test_conftest_stub_is_present_at_test_time():
    """If conftest's stub was uninstalled by another test module, this fails.

    Specifically: if test_phase_b_patcher.py's import dance ran first and the
    real-import failed without restoration (the v4 P4 bug), this test would
    see no transformers entry in sys.modules → assertion fails.
    """
    assert "transformers" in sys.modules, (
        "transformers entry missing from sys.modules — conftest stub was "
        "removed without restoration (v4 P4 regression)"
    )


def test_conftest_stub_carries_marker_attribute():
    """The conftest stub sets ``__is_test_stub__`` for clean detection.

    A real transformers package would not set this attribute. If we see
    something other than the marked stub here, either real transformers got
    loaded (acceptable in envs that have it) or some other code installed a
    different stub.
    """
    tf = sys.modules["transformers"]
    has_real_file = hasattr(tf, "__file__") and tf.__file__ is not None
    has_marker = getattr(tf, "__is_test_stub__", False)
    # Acceptable states: the conftest stub (marker), or real transformers
    # (has __file__). The buggy state is: a stub without the marker AND no
    # __file__ — which suggests a different mock got installed mid-session.
    assert has_marker or has_real_file, (
        "transformers in sys.modules is neither the conftest stub (no marker) "
        "nor a real installation (no __file__). Some test silently swapped it."
    )


def test_anchor_gate_module_loads_without_torch():
    """The anchor_gate module is intentionally torch-free; verify it still is.

    A future refactor that imports torch into utils.anchor_gate would couple
    the gate's testability to torch's availability — defeating v3 P1's
    integration-testing design.
    """
    import importlib
    # Force a fresh import to catch lazy-import regressions.
    if "stage1.utils.anchor_gate" in sys.modules:
        del sys.modules["stage1.utils.anchor_gate"]
    mod = importlib.import_module("stage1.utils.anchor_gate")
    # The module should not have pulled in torch as a side-effect.
    # (We don't assert torch is absent — the env may have it for other tests.
    # We assert the module itself doesn't reference torch at module level.)
    src_path = mod.__file__
    assert src_path is not None
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    assert "import torch" not in src, (
        "stage1.utils.anchor_gate must remain torch-free (v3 P1 invariant)."
    )


def test_run_status_module_loads_without_torch():
    """Same invariant for the run_status helper."""
    import importlib
    if "stage1.utils.run_status" in sys.modules:
        del sys.modules["stage1.utils.run_status"]
    mod = importlib.import_module("stage1.utils.run_status")
    src_path = mod.__file__
    assert src_path is not None
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    assert "import torch" not in src


def test_post_analysis_static_grid_does_not_require_composer_import():
    """v4 P3 invariant: fixed_w4_* inference must not transitively import
    composer at module load.

    If post_analysis.py acquires a top-level ``from stage1.models.composer
    import PHASE_A_GRID`` (instead of the current lazy import inside the
    function), the analysis path becomes torch-dependent again.
    """
    src_path = os.path.join(
        _REPO_ROOT, "stage1", "analysis", "post_analysis.py",
    )
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    # Search for the bad pattern: a top-level (non-indented) composer import.
    bad_lines = [
        ln for ln in src.splitlines()
        if ln.startswith("from stage1.models.composer import")
        or ln.startswith("import stage1.models.composer")
    ]
    assert not bad_lines, (
        "post_analysis.py must not import composer at module level — "
        "that would re-introduce the heavy-import dependency v4 P3 fixed."
    )


def test_conftest_no_longer_uses_dead_ensure_stub_helper():
    """The dead ``_ensure_stub`` helper was removed in v4 P4 — check it stays
    removed so future maintainers don't reintroduce the conditional logic
    that caused the original order-sensitivity bug.

    Implementation note: parse the AST and inspect the actual definitions /
    name references rather than substring-matching the source, otherwise the
    docstring that EXPLAINS the removal would itself trip the test.
    """
    import ast
    src_path = os.path.join(_REPO_ROOT, "stage1", "tests", "conftest.py")
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    # Collect every defined function name and every Name reference in the
    # module body (excluding docstrings).
    defined = set()
    referenced = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defined.add(node.name)
        elif isinstance(node, ast.Name):
            referenced.add(node.id)
    bad_names = {"_ensure_stub", "_already_present"}
    reintroduced = (defined | referenced) & bad_names
    assert not reintroduced, (
        f"conftest.py: dead helper(s) re-introduced as actual code: "
        f"{sorted(reintroduced)}. The v4 P4 pattern installs the stub "
        f"deterministically and does not need them."
    )
