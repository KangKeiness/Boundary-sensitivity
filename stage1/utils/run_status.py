"""Run-status banner + artifact writer for Phase B.

YELLOW-LIGHT v3 PRIORITY 3.

Centralises the run-status convention so a failed Phase B run cannot leave
plausible-looking artifacts on disk. Every persist of phase_b_summary.{json,txt}
goes through ``write_phase_b_status_artifacts`` and stamps the same status into
all three sentinels:

    1. ``phase_b_summary.json`` — top-level ``run_status`` and ``failure_reason`` fields.
    2. ``phase_b_summary.txt`` — leading ``RUN STATUS:`` banner + optional
       ``FAILURE REASON:`` line.
    3. ``RUN_STATUS.txt`` — single-token sentinel for downstream tooling.

Decoupled from torch / numpy so it is straightforwardly unit-testable.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

RUN_STATUS_PENDING: str = "pending"
RUN_STATUS_PASSED: str = "passed"
RUN_STATUS_FAILED: str = "failed"

_VALID_STATUSES = (RUN_STATUS_PENDING, RUN_STATUS_PASSED, RUN_STATUS_FAILED)


def build_status_banner(
    status: str, failure_reason: Optional[str] = None,
) -> List[str]:
    """Return the leading TXT banner lines for a given run status.

    ``failure_reason`` is only embedded when ``status == "failed"``.
    """
    if status not in _VALID_STATUSES:
        raise ValueError(
            f"invalid run_status {status!r}; expected one of {_VALID_STATUSES}"
        )
    marker = {
        RUN_STATUS_PENDING: "RUN STATUS: PENDING — gates not yet evaluated",
        RUN_STATUS_PASSED:  "RUN STATUS: PASSED — all gates satisfied",
        RUN_STATUS_FAILED:  "RUN STATUS: FAILED",
    }[status]
    banner = ["=" * 60, marker, "=" * 60]
    if status == RUN_STATUS_FAILED and failure_reason:
        banner.append(f"FAILURE REASON: {failure_reason}")
        banner.append("=" * 60)
    banner.append("")
    return banner


def write_phase_b_status_artifacts(
    run_dir: str,
    summary: Dict[str, Any],
    body_lines: List[str],
    status: str,
    *,
    failure_reason: Optional[str] = None,
) -> None:
    """Persist phase_b_summary.{json,txt} + RUN_STATUS.txt with ``status``.

    Mutates ``summary`` in place (sets ``run_status`` and ``failure_reason``).
    ``body_lines`` is the TXT body WITHOUT any leading status banner — the
    helper prepends the banner from scratch on every call.
    """
    if status not in _VALID_STATUSES:
        raise ValueError(
            f"invalid run_status {status!r}; expected one of {_VALID_STATUSES}"
        )
    summary["run_status"] = status
    summary["failure_reason"] = failure_reason

    banner = build_status_banner(status, failure_reason)
    text = "\n".join(banner + body_lines)

    with open(os.path.join(run_dir, "phase_b_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "phase_b_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(run_dir, "RUN_STATUS.txt"), "w", encoding="utf-8") as f:
        f.write(f"{status.upper()}\n")
        if failure_reason:
            f.write(f"failure_reason: {failure_reason}\n")
