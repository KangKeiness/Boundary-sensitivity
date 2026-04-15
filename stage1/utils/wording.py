"""Shared conservative-wording gate for Phase B and Phase C.

Phase A has an inline FORBIDDEN_PHRASES list at ``run_phase_a.py:859-883`` that
is intentionally left untouched (pinned, do-not-modify). This module is ADDITIVE
and is used by Phase B and Phase C artifacts.

Two tuples are exposed:

- ``FORBIDDEN_PHRASES_PHASE_B`` — the pre-existing Phase B list. Phase B reserves
  the Phase-C core vocabulary ("restoration effect", "residual effect",
  "restoration proportion") as forbidden so that Phase B artifacts cannot
  accidentally adopt Phase C's decomposition terms.
- ``FORBIDDEN_PHRASES_PHASE_C`` — the Phase C list. Phase C's core vocabulary is
  allowed (it is the phase's reporting language), but formal causal-mediation
  terminology ("natural direct effect", "natural indirect effect", "NIE", "NDE",
  "causal mediation") is forbidden because the prompt-side restoration
  intervention does not license NIE/NDE claims.

Backward compatibility: ``FORBIDDEN_PHRASES`` remains a module attribute and is
bound to ``FORBIDDEN_PHRASES_PHASE_B`` so that all existing Phase B call sites
and tests that iterate this tuple continue to pass bytewise.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

# Canonical Phase B list. Order matters only for deterministic test iteration.
FORBIDDEN_PHRASES_PHASE_B: Tuple[str, ...] = (
    "proves the mechanism",
    "proves mechanism",
    "causal proof",
    "identifies the true cause",
    "fully explains",
    "demonstrates causation",
    "restoration effect",       # Phase C reserved — forbidden in Phase B
    "residual effect",          # Phase C reserved — forbidden in Phase B
    "restoration proportion",   # Phase C reserved — forbidden in Phase B
)

# Phase C list. Core Phase C vocabulary is intentionally NOT forbidden here.
FORBIDDEN_PHRASES_PHASE_C: Tuple[str, ...] = (
    "proves the mechanism",
    "proves mechanism",
    "causal proof",
    "identifies the true cause",
    "fully explains",
    "demonstrates causation",
    "natural direct effect",
    "natural indirect effect",
    "causal mediation",
    # Note: the "NIE/NDE" / "NDE/NIE" literal tokens in spec §8 are
    # intentionally NOT included — they collide with the verbatim mandated
    # caveat ("...not a formal NIE/NDE decomposition.") that every Phase C
    # summary must contain. The spelled-out "natural direct effect" /
    # "natural indirect effect" phrases above remain forbidden, so formal
    # mediation prose is still blocked.
)

# Backward-compat alias for existing Phase B call sites (including
# ``stage1/run_phase_b.py`` and ``stage1/tests/test_phase_b_patcher.py``).
FORBIDDEN_PHRASES: Tuple[str, ...] = FORBIDDEN_PHRASES_PHASE_B


def check_artifacts_for_forbidden(
    paths: Sequence[str],
    *,
    phrases: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return a list of violation strings for any forbidden phrase occurrences.

    Each violation is formatted as ``"<path>: found forbidden phrase '<p>'"``.
    Files that do not exist are skipped silently (per Phase B spec §8). All
    file reads use ``encoding='utf-8'`` (Windows + Chinese MGSM output
    requirement). The match is case-insensitive to match Phase A's
    ``.lower()`` behaviour.

    Args:
        paths: Iterable of artifact paths to scan.
        phrases: Optional explicit phrase tuple. When ``None`` (default), the
            canonical ``FORBIDDEN_PHRASES`` (== ``FORBIDDEN_PHRASES_PHASE_B``)
            is used, preserving pre-Phase-C behaviour for every existing call
            site. Phase C call sites pass
            ``phrases=FORBIDDEN_PHRASES_PHASE_C`` explicitly.
    """
    active_phrases: Sequence[str] = (
        FORBIDDEN_PHRASES if phrases is None else phrases
    )
    violations: List[str] = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            content = f.read().lower()
        for phrase in active_phrases:
            if phrase.lower() in content:
                violations.append(f"{path}: found forbidden phrase '{phrase}'")
    return violations
