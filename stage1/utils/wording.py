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
  allowed because it is the phase's reporting language, while stronger formal
  causal-decomposition terminology is blocked.

Backward compatibility: ``FORBIDDEN_PHRASES`` remains a module attribute and is
bound to ``FORBIDDEN_PHRASES_PHASE_B`` so that all existing Phase B call sites
and tests that iterate this tuple continue to pass bytewise.
"""

from __future__ import annotations

import os
import re
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
    "mediation-style",
    "mediation analysis",
    "formal mediation",
    "causal mediation",
    "natural direct effect",
    "natural indirect effect",
    "NIE",
    "NDE",
)

# Backward-compat alias for existing Phase B call sites (including
# ``stage1/run_phase_b.py`` and ``stage1/tests/test_phase_b_patcher.py``).
FORBIDDEN_PHRASES: Tuple[str, ...] = FORBIDDEN_PHRASES_PHASE_B


_ACRONYM_PHRASES = {"nie", "nde"}


def _contains_forbidden_phrase(content_lower: str, phrase: str) -> bool:
    phrase_lower = phrase.lower()
    if phrase_lower in _ACRONYM_PHRASES:
        pattern = rf"(?<![a-z0-9_]){re.escape(phrase_lower)}(?![a-z0-9_])"
        return re.search(pattern, content_lower) is not None
    return phrase_lower in content_lower


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
            if _contains_forbidden_phrase(content, phrase):
                violations.append(f"{path}: found forbidden phrase '{phrase}'")
    return violations
