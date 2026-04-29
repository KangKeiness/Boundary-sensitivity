"""Phase B — Restoration Intervention (prompt-side hidden-state patching).

Tests whether recovery-side disruption is causally relevant to degradation
via prompt-side hidden-state patching. See
``notes/specs/phase_b_rewrite.md`` for the authoritative spec.

IMPORTANT METHODOLOGICAL CONSTRAINT:
    Patching applies only to prompt-side hidden-state processing. Clean hidden
    states are available for prompt tokens only. This is prompt-side
    restoration intervention, NOT full-sequence causal intervention. All
    summaries state this explicitly.

Phase C terminology ("restoration effect", "residual effect", "restoration
proportion") is reserved and MUST NOT appear in Phase B artifacts; this is
enforced by the FORBIDDEN_PHRASES gate in ``utils.wording``.

Usage (CLI):
    python stage1/run_phase_b.py --config stage1/configs/stage2_confound.yaml
    python stage1/run_phase_b.py --config stage1/configs/stage2_confound.yaml --sanity

Usage (Jupyter):
    from stage1.run_phase_b import run_phase_b
"""

from __future__ import annotations

import argparse
import builtins
import csv
import gc
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import transformers

from stage1.utils.config import load_config, setup_logging
from stage1.utils.logger import create_run_dir
from stage1.utils.manifest_parity import extract_parity_block
from stage1.utils.provenance import build_runtime_provenance
from stage1.utils.anchor_gate import (
    ANCHOR_WORKFLOW_DOC,
    PHASE_A_CROSS_CHECK_TOL,
    evaluate_phase_b_anchor_gate,
    render_anchor_gate_diagnostic,
)
from stage1.utils.run_status import (
    RUN_STATUS_FAILED,
    RUN_STATUS_PASSED,
    RUN_STATUS_PENDING,
    write_phase_b_status_artifacts,
)
from stage1.utils.wording import FORBIDDEN_PHRASES, check_artifacts_for_forbidden
from stage1.data.loader import load_mgsm
from stage1.models.composer import load_models, compose_model
from stage1.inference.parser import parse_answer
from stage1.analysis.evaluator import exact_match
from stage1.intervention.patcher import (
    METHODOLOGY_TAG,
    PatchConfig,
    RESTORATION_PATCHES,
    REVERSE_CORRUPTION_PATCHES,
    run_patched_inference,
)

logger = logging.getLogger(__name__)


_CLI_ASCII_REPLACEMENTS = {
    "—": "-",
    "–": "-",
    "−": "-",
    "←": "<-",
    "→": "->",
    "↔": "<->",
    "≤": "<=",
    "≥": ">=",
    "Δ": "delta",
    "×": "x",
}


def _ascii_cli_text(text: str) -> str:
    out = text
    for src, dst in _CLI_ASCII_REPLACEMENTS.items():
        out = out.replace(src, dst)
    return out.encode("ascii", errors="replace").decode("ascii")


def _safe_print(*args, **kwargs):
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    file = kwargs.pop("file", sys.stdout)
    flush = kwargs.pop("flush", False)
    if kwargs:
        return builtins.print(*args, sep=sep, end=end, file=file, flush=flush, **kwargs)
    if file not in (sys.stdout, sys.stderr):
        return builtins.print(*args, sep=sep, end=end, file=file, flush=flush)
    msg = sep.join(str(a) for a in args)
    file.write(_ascii_cli_text(msg) + end)
    if flush:
        file.flush()


# Keep artifacts unchanged; only sanitize stdout/stderr for locale-safe CLI.
print = _safe_print

# Effect-size threshold for the comparative sentence gate (spec §7, §11.10).
EPSILON_DELTA: float = 0.02

# Cross-phase accuracy tolerance (2/250 samples). Authoritative definition lives
# in ``stage1/utils/anchor_gate.py``; re-exported here for backward-compat with
# tests and call sites that imported it from this module.
# (PHASE_A_CROSS_CHECK_TOL is imported above.)

# Canonical methodological-constraint string (contains "prompt-side" per spec §11.2).
METHODOLOGICAL_CONSTRAINT: str = (
    "Patching applies only to prompt-side hidden-state processing. "
    "Clean hidden states are available for prompt tokens only. "
    "This is prompt-side restoration intervention, NOT full-sequence "
    "causal intervention. Claims remain intervention-based and conservative."
)


# ─── Determinism / seeding helpers ───────────────────────────────────────────

def _apply_determinism(seed: int) -> List[str]:
    """Seed all RNGs and enable deterministic algorithms.

    Spec §6. Returns a list of warning strings captured from
    ``torch.use_deterministic_algorithms`` (``warn_only=True``).
    """
    # Must be set BEFORE any CUDA op.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

    warnings: List[str] = []
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:  # pragma: no cover — environment-dependent
        warnings.append(f"use_deterministic_algorithms raised: {exc!r}")
    return warnings


def _git_sha() -> str:
    try:
        _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_repo_root
        ).decode().strip()
    except Exception:
        return "unknown"


def _state_dict_sha256(model: torch.nn.Module) -> str:
    """Deterministic SHA-256 over the model's state_dict tensor bytes.

    Implementation notes:
        * Tensors are moved to CPU and made contiguous before hashing so the
          byte layout is independent of device and stride.
        * The original tensor dtype is preserved (no float32 cast). Stage 1
          loads weights at a single dtype (``torch.float16`` by default in
          ``composer.load_models``); the hash is therefore deterministic
          across re-runs of the same model + revision + dtype combo, which
          is what the spec §10 test #4 ("composed weights unchanged across
          all patched inference passes") actually checks.
        * Keys are sorted so dict iteration order does not affect the digest.
    """
    h = hashlib.sha256()
    sd = model.state_dict()
    for key in sorted(sd.keys()):
        t = sd[key]
        h.update(key.encode("utf-8"))
        h.update(t.detach().to("cpu").contiguous().numpy().tobytes())
    return h.hexdigest()


# ─── Phase A cross-check loader ──────────────────────────────────────────────
# The actual loader / parity-filter / decision logic lives in
# ``stage1/utils/anchor_gate.py`` (torch-free) so that integration-level
# regression tests can exercise it end-to-end. The thin wrappers below preserve
# the original module-level CWD-invariant resolution for callers that imported
# them by name.


def _phase_a_outputs_dir() -> str:
    """Resolve the Phase A outputs directory relative to the repo root.

    CWD-invariant per spec §11.7. Delegates to the canonical resolver in
    ``stage1.utils.anchor_gate``.
    """
    from stage1.utils.anchor_gate import default_phase_a_outputs_dir
    return default_phase_a_outputs_dir()


def _stage1_outputs_dir() -> str:
    """Resolve the Stage 1 sweep outputs dir (``stage1/outputs/run_*``)."""
    from stage1.utils.anchor_gate import default_stage1_outputs_dir
    return default_stage1_outputs_dir()


# ─── Paired bootstrap for comparative-sentence gate ──────────────────────────

def _paired_bootstrap_diff_ci(
    corrects_a: List[int], corrects_b: List[int],
    *, n_resamples: int = 1000, seed: int = 0, ci: float = 0.95,
) -> Tuple[float, float, float]:
    """Return (point_estimate, ci_lo, ci_hi) of mean(a) - mean(b) via paired bootstrap.

    ``corrects_a``/``corrects_b`` are 0/1 integer lists of equal length (per-sample
    correctness). Sampling is paired over indices.
    """
    n = len(corrects_a)
    if n == 0 or n != len(corrects_b):
        raise ValueError("bootstrap requires equal-length non-empty vectors")
    a = np.asarray(corrects_a, dtype=np.float64)
    b = np.asarray(corrects_b, dtype=np.float64)
    point = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(diffs, alpha))
    hi = float(np.quantile(diffs, 1.0 - alpha))
    return point, lo, hi


# ─── IO helpers ──────────────────────────────────────────────────────────────

def _save_condition_results(run_dir: str, name: str, results: List[Dict]) -> None:
    """Save per-condition results as JSONL (UTF-8)."""
    path = os.path.join(run_dir, f"results_{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            row = {k: v for k, v in r.items() if k != "hidden_states"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: str, fieldnames: List[str], data: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(data)


def _compute_acc(results: List[Dict]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("correct", False)) / len(results)


def _annotate_correct(results: List[Dict], samples: List[Dict]) -> None:
    """In-place: parse answers and attach ``correct`` (bool) to each result.

    O(n) gold-answer lookup — Finding #12.
    """
    gold_by_id = {s["sample_id"]: s["gold_answer"] for s in samples}
    for r in results:
        parsed = parse_answer(r["output_text"])
        r.update(parsed)
        r["correct"] = exact_match(
            gold_by_id[r["sample_id"]],
            r.get("normalized_answer"),
        )


# ─── Subset construction + conditional metrics (spec §7.2) ──────────────────

# Canonical Stage 1 / Phase A anchor pins (spec §9). Hard-fail on drift.
CANONICAL_STAGE1_ANCHOR: str = "run_20260427_020843_372153"
CANONICAL_PHASE_A_ANCHOR: str = "run_20260427_104320_510816"

# Subset-size warning thresholds (spec §7.2.3). Conservative defaults from
# brief §E; raising / lowering these requires a spec amendment, not a code
# patch.
SUBSET_SIZE_WARN_BROKEN: int = 20
SUBSET_SIZE_WARN_FLIPPED: int = 20
SUBSET_SIZE_WARN_CORRECT_FLIPPED: int = 10

# Wording-gate artifact filenames (spec §11.11). Explicit, fixed list of
# user-facing report/summary artifacts; NO glob, NO legacy filename heuristic.
_WORDING_GATE_FILENAMES_BASE: Tuple[str, ...] = (
    "phase_b_summary.txt",
    "phase_b_summary.json",
    "restoration_table.csv",
    "corruption_table.csv",
    "phase_b_subsets.json",
    "phase_b_subsets.csv",
    "phase_b_conditional_summary.json",
    "phase_b_conditional_summary.csv",
)
_WORDING_GATE_FILENAMES_DRIFT: Tuple[str, ...] = (
    "phase_b_drift_behavior_summary.json",
    "phase_b_drift_behavior.csv",
)


def _subset_warning_strings(subset_counts: Dict[str, int]) -> List[str]:
    """Return the spec §7.2.3 warning strings for a given subset-count dict.

    Pure-Python helper (testable without torch). ``subset_counts`` must have
    keys ``n_broken``, ``n_flipped``, ``n_correct_flipped``.
    """
    warnings: List[str] = []
    n_broken = int(subset_counts.get("n_broken", 0))
    n_flipped = int(subset_counts.get("n_flipped", 0))
    n_correct_flipped = int(subset_counts.get("n_correct_flipped", 0))
    if n_broken < SUBSET_SIZE_WARN_BROKEN:
        warnings.append(
            f"warning: n_broken={n_broken} < {SUBSET_SIZE_WARN_BROKEN} "
            "— broken-subset recovery metrics are noisy; do not claim "
            "strong restoration"
        )
    if n_flipped < SUBSET_SIZE_WARN_FLIPPED:
        warnings.append(
            f"warning: n_flipped={n_flipped} < {SUBSET_SIZE_WARN_FLIPPED} "
            "— answer-restoration metrics are noisy"
        )
    if 0 < n_correct_flipped < SUBSET_SIZE_WARN_CORRECT_FLIPPED:
        warnings.append(
            f"warning: n_correct_flipped={n_correct_flipped} < "
            f"{SUBSET_SIZE_WARN_CORRECT_FLIPPED} — exploratory only"
        )
    return warnings


def _build_subsets(
    clean_rows: List[Dict],
    corrupt_rows: List[Dict],
) -> Dict[str, List[bool]]:
    """Return dict of subset name -> per-sample boolean membership list.

    Subsets per spec §7.2.1. Length of every list equals
    ``len(clean_rows) == len(corrupt_rows)``. None-handling for ``S_flipped``
    per spec §7.2.1: ``None == None`` is NOT flipped; ``None`` vs ``non-None``
    (in either direction) IS flipped.
    """
    if len(clean_rows) != len(corrupt_rows):
        raise RuntimeError(
            f"_build_subsets: row-count mismatch clean={len(clean_rows)} "
            f"vs corrupt={len(corrupt_rows)}"
        )
    n = len(clean_rows)
    sets: Dict[str, List[bool]] = {
        "S_stable_correct": [False] * n,
        "S_broken": [False] * n,
        "S_repaired": [False] * n,
        "S_stable_wrong": [False] * n,
        "S_flipped": [False] * n,
        "S_correct_flipped": [False] * n,
    }
    for i in range(n):
        cc = bool(clean_rows[i]["correct"])
        cr = bool(corrupt_rows[i]["correct"])
        ans_c = clean_rows[i].get("normalized_answer")
        ans_r = corrupt_rows[i].get("normalized_answer")
        if cc and cr:
            sets["S_stable_correct"][i] = True
        elif cc and not cr:
            sets["S_broken"][i] = True
        elif (not cc) and cr:
            sets["S_repaired"][i] = True
        else:
            sets["S_stable_wrong"][i] = True
        # S_flipped: None==None not flipped; None vs non-None flipped.
        if ans_c is None and ans_r is None:
            flipped = False
        elif ans_c is None or ans_r is None:
            flipped = True
        else:
            flipped = (ans_c != ans_r)
        sets["S_flipped"][i] = flipped
        if cc and cr and flipped:
            sets["S_correct_flipped"][i] = True
    return sets


def _load_anchor_per_sample(
    anchor_jsonl_path: str,
    samples: List[Dict],
) -> Tuple[List[Dict], str]:
    """Load per-sample rows from a Stage 1 / Phase A anchor JSONL, ordered.

    Validates that ``len(rows) == len(samples)`` and that ``sample_id``
    matches pairwise (parity already enforced upstream by
    ``manifest_parity``). Each returned row dict contains at least:
    ``sample_id, output_text, normalized_answer, parse_success, correct,
    recomputed_correct``.

    Correctness-source policy (spec §7.2.1):
        - If the JSONL has a ``correct`` field, recompute via
          ``analysis.evaluator.exact_match(samples[i].gold_answer, normalized_answer)``
          and compare. On disagreement on ANY sample, raise
          ``RuntimeError("evaluator drift: anchor JSONL correct field
          disagrees with recomputed exact_match for sample_id=<id>")``. Use
          the JSONL value as ``correct``; set ``recomputed_correct=False``.
        - If the JSONL lacks ``correct``, recompute via the same call, set
          ``correct = recomputed_value``, ``recomputed_correct=True``.

    Returns:
        ``(rows, label)`` where ``label`` is ``"recomputed"`` or ``"jsonl"``
        indicating which path the MAJORITY of rows took. Used to populate
        ``summary["correct_field_source"]``.
    """
    if not os.path.exists(anchor_jsonl_path):
        raise FileNotFoundError(
            f"_load_anchor_per_sample: anchor JSONL missing at "
            f"{anchor_jsonl_path}"
        )
    rows: List[Dict] = []
    with open(anchor_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if len(rows) != len(samples):
        raise RuntimeError(
            f"_load_anchor_per_sample: row count mismatch in "
            f"{anchor_jsonl_path}: rows={len(rows)} vs samples={len(samples)}"
        )
    gold_by_id = {s["sample_id"]: s["gold_answer"] for s in samples}
    n_recomputed = 0
    n_jsonl = 0
    for i, (row, sample) in enumerate(zip(rows, samples)):
        if row.get("sample_id") != sample["sample_id"]:
            raise RuntimeError(
                f"_load_anchor_per_sample: sample_id parity violation at "
                f"index {i}: anchor row sample_id={row.get('sample_id')!r} "
                f"vs samples[i].sample_id={sample['sample_id']!r}"
            )
        normalized = row.get("normalized_answer")
        recomputed = bool(exact_match(
            gold_by_id[sample["sample_id"]], normalized,
        ))
        if "correct" in row:
            stored = bool(row["correct"])
            if stored != recomputed:
                raise RuntimeError(
                    "evaluator drift: anchor JSONL correct field disagrees "
                    "with recomputed exact_match for sample_id="
                    f"{sample['sample_id']}"
                )
            row["correct"] = stored
            row["recomputed_correct"] = False
            n_jsonl += 1
        else:
            row["correct"] = recomputed
            row["recomputed_correct"] = True
            n_recomputed += 1
        # Defensive: ensure parse_success is a bool (some legacy rows may
        # omit it; fall back to "normalized_answer is not None").
        if "parse_success" not in row:
            row["parse_success"] = (normalized is not None)
    label = "recomputed" if n_recomputed >= n_jsonl else "jsonl"
    return rows, label


def _count_human_continuation(rows: List[Dict]) -> int:
    """Count rows whose ``output_text`` contains the literal substring
    ``"Human:"``. Spec §11.5 / §11.6.
    """
    return sum(
        1 for r in rows
        if "Human:" in (r.get("output_text") or "")
    )


def _mean_output_length_chars(rows: List[Dict]) -> float:
    if not rows:
        return 0.0
    return sum(len(r.get("output_text") or "") for r in rows) / float(len(rows))


# ─── r2: sanity-mode no-patch parity gate ────────────────────────────────────

# Conservative size-adjusted thresholds used in sanity mode (n=5). Stage 1
# anchors per brief §A: ``no_swap`` mean output ≈ 676.7 chars, ``hard_swap_b8``
# mean output ≈ 697.9 chars. The 1000-char cap is ~1.5× of these — generous
# headroom for n=5 noise but well below the failed-run levels (≈ 1456 / ≈ 1688
# in the r1 sanity smoke ``run_20260429_094926_429332``). The Human:
# threshold is 1 in 5 (i.e., effectively 0; tolerates a single fluke).
_SANITY_PARITY_HUMAN_HARD_FAIL = 1   # > this many ⇒ hard fail
_SANITY_PARITY_LENGTH_HARD_FAIL_CHARS = 1000.0  # > this ⇒ hard fail


def _sanity_no_patch_parity_check(
    clean_no_patch_rows: List[Dict],
    restoration_no_patch_rows: List[Dict],
) -> Tuple[str, List[str], Dict[str, Dict[str, float]]]:
    """Sanity-mode no-patch parity gate (operator-mandated, r2 dispatch).

    Operator constraint: in sanity mode, the no-patch parity gates MUST NOT
    be silently skipped. They are size-adjusted but they fire. This is
    SEPARATE from the conditional-metrics-layer skip (spec §7.2.4 — that
    one needs anchors; this gate does not).

    Behavior (per dispatch):
        - For ``clean_no_patch`` and ``restoration_no_patch`` ONLY (not
          patched conditions):
          - sanity Human: threshold ≤ 1 in 5 samples (effectively 0)
          - sanity avg_output_length cap 1000 chars (≈ 1.5× Stage 1 anchors)
        - When EITHER signal triggers a hard-fail, return
          ``("failed", [reason1, ...], metrics)`` — the caller wires
          ``RUN_STATUS_FAILED`` and a ``failure_reason`` starting with
          ``"sanity P0 parity check failed: "``.
        - When both signals pass, return ``("passed", [], metrics)``.

    The ``metrics`` dict mirrors what the conditional-metrics layer would
    have produced for the two no-patch conditions, so it can be embedded
    verbatim into ``phase_b_summary.json`` for downstream review without
    re-iterating the JSONLs.

    Returns:
        Tuple[status: "passed"|"failed", failures: List[str], metrics: Dict].
    """
    metrics: Dict[str, Dict[str, float]] = {}
    failures: List[str] = []

    pairs = [
        ("clean_no_patch", clean_no_patch_rows),
        ("restoration_no_patch", restoration_no_patch_rows),
    ]
    for name, rows in pairs:
        n = len(rows)
        human_count = _count_human_continuation(rows)
        avg_len = _mean_output_length_chars(rows)
        metrics[name] = {
            "n_total": float(n),
            "human_continuation_count": float(human_count),
            "avg_output_length_chars": float(avg_len),
        }
        if human_count > _SANITY_PARITY_HUMAN_HARD_FAIL:
            failures.append(
                f"{name} human_continuation_count={human_count} "
                f"(> {_SANITY_PARITY_HUMAN_HARD_FAIL}/{n} hard-fail threshold)"
            )
        if avg_len > _SANITY_PARITY_LENGTH_HARD_FAIL_CHARS:
            failures.append(
                f"{name} avg_output_length_chars={avg_len:.1f} "
                f"(> {_SANITY_PARITY_LENGTH_HARD_FAIL_CHARS:.0f} hard-fail "
                f"threshold; Stage 1 anchors ≈ 676.7 / 697.9 per brief §A)"
            )

    status = "failed" if failures else "passed"
    return status, failures, metrics


def _compute_conditional_metrics_one_condition(
    pc_rows: List[Dict],
    no_patch_rows: List[Dict],
    ans_clean: List[Optional[str]],
    ans_corrupt: List[Optional[str]],
    correct_clean: List[bool],
    correct_corrupt: List[bool],
    subsets: Dict[str, List[bool]],
    direction: str,
) -> Dict[str, Any]:
    """Return the 8-block dict per spec §7.2.2 / §7.2.5.

    ``no_patch_rows`` is the matching-direction reference: for restoration
    direction it is ``restoration_no_patch`` (alias for the composed
    no_patch); for corruption direction it is ``clean_no_patch``. Per-sample
    membership lists in ``subsets`` align with ``pc_rows`` / ``no_patch_rows``
    ordering.
    """
    if direction not in ("restoration", "corruption"):
        raise RuntimeError(
            f"_compute_conditional_metrics_one_condition: invalid direction "
            f"{direction!r}; expected 'restoration' or 'corruption'"
        )
    n = len(pc_rows)
    if any(len(x) != n for x in (
        no_patch_rows, ans_clean, ans_corrupt, correct_clean, correct_corrupt,
    )):
        raise RuntimeError(
            "_compute_conditional_metrics_one_condition: length mismatch "
            f"n_pc={n} vs no_patch={len(no_patch_rows)} "
            f"ans_clean={len(ans_clean)} ans_corrupt={len(ans_corrupt)} "
            f"correct_clean={len(correct_clean)} "
            f"correct_corrupt={len(correct_corrupt)}"
        )

    correct_pc = [bool(r.get("correct", False)) for r in pc_rows]
    parse_success_pc = [bool(r.get("parse_success", False)) for r in pc_rows]
    ans_pc: List[Optional[str]] = [r.get("normalized_answer") for r in pc_rows]
    correct_npref = [bool(r.get("correct", False)) for r in no_patch_rows]
    parse_success_npref = [
        bool(r.get("parse_success", False)) for r in no_patch_rows
    ]

    def _safe_mean(idx_mask: List[bool], values: List[bool]) -> Optional[float]:
        sel = [v for v, m in zip(values, idx_mask) if m]
        if not sel:
            return None
        return float(sum(1 for v in sel if v) / len(sel))

    def _ans_match_rate(
        idx_mask: List[bool],
        a: List[Optional[str]],
        b: List[Optional[str]],
    ) -> Optional[float]:
        sel_a = [x for x, m in zip(a, idx_mask) if m]
        sel_b = [x for x, m in zip(b, idx_mask) if m]
        if not sel_a:
            return None
        n_match = 0
        for x, y in zip(sel_a, sel_b):
            if x is None and y is None:
                # None == None counted as match only when both parsed to
                # None. Matches the S_flipped semantics described in spec
                # §7.2.1 / §7.2.2 (3).
                n_match += 1
            elif x is None or y is None:
                continue
            elif x == y:
                n_match += 1
        return float(n_match / len(sel_a))

    # Map direction to which subset reference column we treat as "the
    # clean answer" vs "the corrupt answer" — spec §7.2.5 symmetry rule.
    if direction == "restoration":
        ref_clean = ans_clean
        ref_corrupt = ans_corrupt
    else:  # corruption: clean ↔ corrupt swap (per spec §7.2.5)
        ref_clean = ans_clean
        ref_corrupt = ans_corrupt
    # NB: ans_clean / ans_corrupt are the ORIGINAL Stage 1 anchors. Per
    # spec §7.2.5, the formulas read ``ans_pc[i] == ans_clean[i]`` /
    # ``ans_pc[i] == ans_corrupt[i]`` and the symmetry is encoded by the
    # choice of which condition we treat as the no_patch_rows reference;
    # ans_clean / ans_corrupt themselves are direction-invariant (they
    # are the Stage 1 anchor outputs, regardless of which model we're
    # patching now).
    del ref_clean, ref_corrupt  # documented above; named locals only.

    # (1) Global accuracy and (2) global delta vs no_patch reference.
    acc_patched = (
        float(sum(1 for v in correct_pc if v) / n) if n > 0 else None
    )
    acc_no_patch_ref = (
        float(sum(1 for v in correct_npref if v) / n) if n > 0 else None
    )
    delta_acc = (
        None if (acc_patched is None or acc_no_patch_ref is None)
        else float(acc_patched - acc_no_patch_ref)
    )

    # (3) Broken-subset recovery on S_broken
    s_broken = subsets["S_broken"]
    recovery_to_correct = _safe_mean(s_broken, correct_pc)
    recovery_to_clean_answer = _ans_match_rate(s_broken, ans_pc, ans_clean)
    recovery_no_patch_correct = _safe_mean(s_broken, correct_npref)
    recovery_gain = (
        None if (recovery_to_correct is None
                 or recovery_no_patch_correct is None)
        else float(recovery_to_correct - recovery_no_patch_correct)
    )

    # (4) Answer-flipped restoration on S_flipped
    s_flipped = subsets["S_flipped"]
    answer_restoration = _ans_match_rate(s_flipped, ans_pc, ans_clean)
    answer_corrupt_retention = _ans_match_rate(s_flipped, ans_pc, ans_corrupt)
    overlap_rate = _ans_match_rate(s_flipped, ans_clean, ans_corrupt)
    if (answer_restoration is None
            or answer_corrupt_retention is None
            or overlap_rate is None):
        other_answer_rate: Optional[float] = None
    else:
        other_answer_rate = float(
            1.0 - answer_restoration - answer_corrupt_retention + overlap_rate
        )

    # (5) Stable-correct disruption on S_stable_correct
    s_stable_correct = subsets["S_stable_correct"]
    stable_correct_pres = _safe_mean(s_stable_correct, correct_pc)
    clean_answer_pres = _ans_match_rate(s_stable_correct, ans_pc, ans_clean)

    # (6) Repaired-subset reversal on S_repaired (exploratory)
    s_repaired = subsets["S_repaired"]
    patched_correct_on_repaired = _safe_mean(s_repaired, correct_pc)

    # (7) Parse behavior
    parse_success_global = (
        float(sum(1 for v in parse_success_pc if v) / n) if n > 0 else None
    )
    per_subset_parse: Dict[str, Optional[float]] = {}
    for subset_name, mask in subsets.items():
        per_subset_parse[subset_name] = _safe_mean(mask, parse_success_pc)
    parse_failure_pc = (
        None if parse_success_global is None
        else float(1.0 - parse_success_global)
    )
    parse_failure_no_patch = (
        None if n == 0
        else float(1.0 - sum(1 for v in parse_success_npref if v) / n)
    )
    parse_failure_increase = (
        None if (parse_failure_pc is None or parse_failure_no_patch is None)
        else float(parse_failure_pc - parse_failure_no_patch)
    )

    # (8) Output behavior
    output_texts = [r.get("output_text") or "" for r in pc_rows]
    avg_output_length_chars = (
        float(sum(len(t) for t in output_texts) / n) if n > 0 else None
    )
    human_continuation_count = sum(1 for t in output_texts if "Human:" in t)
    human_continuation_rate = (
        None if n == 0 else float(human_continuation_count / n)
    )
    answer_extraction_failure_count = sum(
        1 for v in parse_success_pc if not v
    )

    return {
        "global_accuracy": acc_patched,
        "delta_from_no_patch_reference": delta_acc,
        "n_total": n,
        "broken_subset_recovery": {
            "recovery_to_correct_rate": recovery_to_correct,
            "recovery_to_clean_answer_rate": recovery_to_clean_answer,
            "recovery_gain_vs_no_patch": recovery_gain,
            "n_subset": int(sum(1 for x in s_broken if x)),
        },
        "answer_flipped_restoration": {
            "answer_restoration_rate": answer_restoration,
            "answer_corrupt_retention_rate": answer_corrupt_retention,
            "other_answer_rate": other_answer_rate,
            "overlap_rate": overlap_rate,
            "n_subset": int(sum(1 for x in s_flipped if x)),
        },
        "stable_correct_disruption": {
            "stable_correct_preservation_rate": stable_correct_pres,
            "clean_answer_preservation_rate": clean_answer_pres,
            "n_subset": int(sum(1 for x in s_stable_correct if x)),
        },
        "repaired_subset_reversal": {
            "patched_correct_rate_on_S_repaired": patched_correct_on_repaired,
            "n_subset": int(sum(1 for x in s_repaired if x)),
            "exploratory": True,
            "caveat": (
                "exploratory; subset is small by construction; do not "
                "overinterpret"
            ),
        },
        "parse_behavior": {
            "parse_success_rate_global": parse_success_global,
            "parse_success_rate_per_subset": per_subset_parse,
            "parse_failure_increase_vs_no_patch": parse_failure_increase,
        },
        "output_behavior": {
            "avg_output_length_chars": avg_output_length_chars,
            "human_continuation_count": human_continuation_count,
            "human_continuation_rate": human_continuation_rate,
            "answer_extraction_failure_count": answer_extraction_failure_count,
        },
        "direction": direction,
    }


def _evaluate_anchor_accuracy_parity_precheck(
    *,
    cross_check_failed_anchors: List[str],
    no_patch_acc: float,
    clean_baseline_acc: float,
    anchor_hard_swap_acc: Optional[float],
    anchor_no_swap_acc: Optional[float],
    tolerance: float,
) -> Tuple[str, Dict[str, Any], List[str], Optional[str]]:
    """r3 (Codex BLOCK fix): pre-emit anchor-accuracy parity check.

    Runs upstream of ``_emit_conditional_artifacts`` so that an
    accuracy-parity failure routes through the explicit-skipped artifact
    path BEFORE any computed conditional metrics are written.

    The accuracy comparison itself is delegated to
    ``stage1.utils.anchor_gate.evaluate_phase_b_anchor_gate``: when both
    anchors are present and at least one delta is outside tolerance,
    ``gate.failed_anchors`` is populated with anchor-name-prefixed strings
    (``"hard_swap_b8: ..."`` / ``"no_swap: ..."``). This helper translates
    those into the spec-mandated ``"<phase_b_condition> vs <stage1_anchor>
    delta=... > tol=..."`` fragments and the operator-mandated
    ``"no_patch_anchor_accuracy_parity_failed: ..."`` reason string.

    Args:
        cross_check_failed_anchors: ``gate.failed_anchors`` from the anchor
            gate; non-empty iff at least one accuracy delta exceeds
            ``tolerance`` AND both anchors were present.
        no_patch_acc: Phase B ``restoration_no_patch.accuracy``.
        clean_baseline_acc: Phase B ``clean_no_patch.accuracy``.
        anchor_hard_swap_acc: Stage 1 ``hard_swap_b8.accuracy`` from the
            parity-selected anchor (or None if unavailable — in which case
            ``cross_check_failed_anchors`` will not contain a
            ``hard_swap_b8`` entry).
        anchor_no_swap_acc: Stage 1 ``no_swap.accuracy`` (or None).
        tolerance: ``PHASE_A_CROSS_CHECK_TOL`` (0.008).

    Returns:
        Tuple of:
            * status: ``"failed"`` if the pre-check fired, else ``""``
              (caller layers in ``"n/a (full mode)"`` /
              ``"n/a (sanity mode)"`` defaults).
            * metrics: per-pair dict ready for embedding into
              ``phase_b_summary.json["anchor_accuracy_parity_metrics"]``.
            * fragments: per-anchor reason fragments matching the
              dispatch-mandated example format.
            * reason: the joined parity-failure-reasons string (a single
              entry to append to ``parity_failure_reasons``), or None when
              no failure fired.
    """
    metrics: Dict[str, Any] = {}
    fragments: List[str] = []

    if not cross_check_failed_anchors:
        return "", metrics, fragments, None

    if anchor_hard_swap_acc is not None and any(
        s.startswith("hard_swap_b8:") for s in cross_check_failed_anchors
    ):
        delta_hs = abs(no_patch_acc - anchor_hard_swap_acc)
        fragments.append(
            "restoration_no_patch vs hard_swap_b8 "
            f"delta={delta_hs:.4f} > tol={tolerance:.4f}"
        )
        metrics["restoration_no_patch_vs_hard_swap_b8"] = {
            "no_patch_accuracy": float(no_patch_acc),
            "anchor_accuracy": float(anchor_hard_swap_acc),
            "delta": float(delta_hs),
            "tolerance": float(tolerance),
        }
    if anchor_no_swap_acc is not None and any(
        s.startswith("no_swap:") for s in cross_check_failed_anchors
    ):
        delta_ns = abs(clean_baseline_acc - anchor_no_swap_acc)
        fragments.append(
            "clean_no_patch vs no_swap "
            f"delta={delta_ns:.4f} > tol={tolerance:.4f}"
        )
        metrics["clean_no_patch_vs_no_swap"] = {
            "clean_baseline_accuracy": float(clean_baseline_acc),
            "anchor_accuracy": float(anchor_no_swap_acc),
            "delta": float(delta_ns),
            "tolerance": float(tolerance),
        }

    if not fragments:
        return "", metrics, fragments, None

    reason = "no_patch_anchor_accuracy_parity_failed: " + "; ".join(fragments)
    return "failed", metrics, fragments, reason


def _emit_conditional_artifacts(
    run_dir: str,
    subsets: Dict[str, List[bool]],
    sample_ids: List[str],
    ans_clean: List[Optional[str]],
    ans_corrupt: List[Optional[str]],
    correct_clean: List[bool],
    correct_corrupt: List[bool],
    recomputed_clean: List[bool],
    recomputed_corrupt: List[bool],
    per_condition_metrics: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Write phase_b_subsets.{json,csv} and phase_b_conditional_summary.{json,csv}.

    Returns the subset-summary dict written into ``phase_b_subsets.json`` so
    callers can mirror it into the top-level summary.
    """
    n = len(sample_ids)
    n_stable_correct = sum(1 for x in subsets["S_stable_correct"] if x)
    n_broken = sum(1 for x in subsets["S_broken"] if x)
    n_repaired = sum(1 for x in subsets["S_repaired"] if x)
    n_stable_wrong = sum(1 for x in subsets["S_stable_wrong"] if x)
    n_flipped = sum(1 for x in subsets["S_flipped"] if x)
    n_correct_flipped = sum(1 for x in subsets["S_correct_flipped"] if x)
    subsets_summary = {
        "n_total": n,
        "n_stable_correct": n_stable_correct,
        "n_broken": n_broken,
        "n_repaired": n_repaired,
        "n_stable_wrong": n_stable_wrong,
        "n_flipped": n_flipped,
        "n_correct_flipped": n_correct_flipped,
        "broken_rate": (n_broken / n) if n else 0.0,
        "repaired_rate": (n_repaired / n) if n else 0.0,
        "answer_flip_rate": (n_flipped / n) if n else 0.0,
    }
    with open(
        os.path.join(run_dir, "phase_b_subsets.json"), "w", encoding="utf-8",
    ) as f:
        json.dump(subsets_summary, f, indent=2, ensure_ascii=False)

    # Phase B subsets CSV — one row per sample_id (spec §7.2.1).
    csv_path = os.path.join(run_dir, "phase_b_subsets.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "sample_id", "ans_clean", "ans_corrupt",
            "correct_clean", "correct_corrupt",
            "recomputed_correct_clean", "recomputed_correct_corrupt",
            "in_S_stable_correct", "in_S_broken", "in_S_repaired",
            "in_S_stable_wrong", "in_S_flipped", "in_S_correct_flipped",
        ])
        for i, sid in enumerate(sample_ids):
            w.writerow([
                sid,
                "" if ans_clean[i] is None else ans_clean[i],
                "" if ans_corrupt[i] is None else ans_corrupt[i],
                str(bool(correct_clean[i])).lower(),
                str(bool(correct_corrupt[i])).lower(),
                str(bool(recomputed_clean[i])).lower(),
                str(bool(recomputed_corrupt[i])).lower(),
                str(bool(subsets["S_stable_correct"][i])).lower(),
                str(bool(subsets["S_broken"][i])).lower(),
                str(bool(subsets["S_repaired"][i])).lower(),
                str(bool(subsets["S_stable_wrong"][i])).lower(),
                str(bool(subsets["S_flipped"][i])).lower(),
                str(bool(subsets["S_correct_flipped"][i])).lower(),
            ])

    with open(
        os.path.join(run_dir, "phase_b_conditional_summary.json"), "w",
        encoding="utf-8",
    ) as f:
        json.dump(per_condition_metrics, f, indent=2, ensure_ascii=False)

    # Long-format CSV (spec §11.9): condition, metric_name, subset, value.
    long_csv = os.path.join(run_dir, "phase_b_conditional_summary.csv")
    with open(long_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "metric_name", "subset", "value"])
        for condition, blocks in per_condition_metrics.items():
            # Direction tag and globals.
            for k in ("global_accuracy", "delta_from_no_patch_reference",
                      "n_total"):
                if k in blocks:
                    w.writerow([condition, k, "global", blocks[k]])
            # Block-by-block.
            block_subset_map = {
                "broken_subset_recovery": "S_broken",
                "answer_flipped_restoration": "S_flipped",
                "stable_correct_disruption": "S_stable_correct",
                "repaired_subset_reversal": "S_repaired",
            }
            for block_name, subset_name in block_subset_map.items():
                block = blocks.get(block_name, {}) or {}
                for k, v in block.items():
                    if k in ("n_subset", "exploratory", "caveat"):
                        continue
                    w.writerow([condition, f"{block_name}.{k}", subset_name, v])
                if "n_subset" in block:
                    w.writerow([
                        condition, f"{block_name}.n_subset", subset_name,
                        block["n_subset"],
                    ])
            parse_block = blocks.get("parse_behavior", {}) or {}
            for k, v in parse_block.items():
                if k == "parse_success_rate_per_subset":
                    for sn, sv in (v or {}).items():
                        w.writerow([
                            condition, f"parse_behavior.{k}", sn, sv,
                        ])
                else:
                    w.writerow([
                        condition, f"parse_behavior.{k}", "global", v,
                    ])
            output_block = blocks.get("output_behavior", {}) or {}
            for k, v in output_block.items():
                w.writerow([
                    condition, f"output_behavior.{k}", "global", v,
                ])
    return subsets_summary


def _emit_conditional_artifacts_skipped(
    run_dir: str,
    reason: str,
    n_total: int,
) -> Dict[str, Any]:
    """Write explicit-skipped artifacts per spec §7.2.4.

    Used in two cases:
        (a) sanity mode + anchors unavailable for subset construction
        (b) operator constraint #4: P0 generation parity failed (clean_no_patch
            or restoration_no_patch failed anchor-parity criteria), so the
            conditional metrics layer MUST emit explicit failed/skipped status
            artifacts with reasons rather than silently produce numbers.

    Schema is identical to spec §7.2.4 (sanity-mode anchor-unavailability
    skip): a single explicit-skipped object per JSON, plus a header-only CSV
    and a sidecar ``.SKIPPED.txt`` for each.
    """
    payload = {
        "status": "skipped",
        "reason": reason,
        "n_total": int(n_total),
    }
    with open(
        os.path.join(run_dir, "phase_b_subsets.json"), "w", encoding="utf-8",
    ) as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(
        os.path.join(run_dir, "phase_b_conditional_summary.json"), "w",
        encoding="utf-8",
    ) as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    # Header-only CSVs (zero data rows) per spec §7.2.4.
    with open(
        os.path.join(run_dir, "phase_b_subsets.csv"), "w", newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow([
            "sample_id", "ans_clean", "ans_corrupt",
            "correct_clean", "correct_corrupt",
            "recomputed_correct_clean", "recomputed_correct_corrupt",
            "in_S_stable_correct", "in_S_broken", "in_S_repaired",
            "in_S_stable_wrong", "in_S_flipped", "in_S_correct_flipped",
        ])
    with open(
        os.path.join(run_dir, "phase_b_conditional_summary.csv"), "w",
        newline="", encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow(["condition", "metric_name", "subset", "value"])
    # Sidecar .SKIPPED.txt for each artifact (per spec §7.2.4).
    with open(
        os.path.join(run_dir, "phase_b_subsets.SKIPPED.txt"), "w",
        encoding="utf-8",
    ) as f:
        f.write(reason + "\n")
    with open(
        os.path.join(run_dir, "phase_b_conditional_summary.SKIPPED.txt"), "w",
        encoding="utf-8",
    ) as f:
        f.write(reason + "\n")
    return payload


def _maybe_compute_drift_diagnostic(
    run_dir: str,
    anchor_dir: Optional[str],
    sample_ids: List[str],
    subsets: Dict[str, List[bool]],
) -> str:
    """Optional drift-vs-behavior diagnostic per spec §7.3.

    Returns one of:
        "computed"
        "skipped: anchor_dir unavailable"
        "skipped: hidden_states_no_swap.pt missing"
        "skipped: hidden_states_hard_swap_b8.pt missing"
        "skipped: shape mismatch (expected [N, n_layers, hidden_dim])"

    Primary AUROC score variable is ``boundary_layer_drift = cosine[:, 8]``
    (boundary layer for hard_swap_b8). Secondary aggregates (mean / max /
    downstream-mean over layers + L2 analogues) are persisted alongside but
    NOT used as the default AUROC score; they are exposed in the JSON for
    downstream review only.
    """
    if anchor_dir is None or not os.path.isdir(anchor_dir):
        return "skipped: anchor_dir unavailable"
    no_swap_path = os.path.join(anchor_dir, "hidden_states_no_swap.pt")
    hs_path = os.path.join(anchor_dir, "hidden_states_hard_swap_b8.pt")
    if not os.path.exists(no_swap_path):
        return "skipped: hidden_states_no_swap.pt missing"
    if not os.path.exists(hs_path):
        return "skipped: hidden_states_hard_swap_b8.pt missing"
    try:
        h_clean = torch.load(no_swap_path, map_location="cpu")
        h_corr = torch.load(hs_path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 — best-effort optional path
        logger.warning("drift diagnostic torch.load failed: %r", exc)
        return "skipped: shape mismatch (expected [N, n_layers, hidden_dim])"
    # Stage 1's prompt-pooled tensors are saved as [N, n_layers, hidden_dim]
    # via ``run_inference``'s _extract_prompt_hidden_states stack. Anything
    # else (per-token, missing axis) is unsupported here; we skip silently.
    if (h_clean.dim() != 3 or h_corr.dim() != 3
            or h_clean.shape != h_corr.shape):
        return "skipped: shape mismatch (expected [N, n_layers, hidden_dim])"
    n_samples = h_clean.shape[0]
    n_layers = h_clean.shape[1]
    if n_samples != len(sample_ids):
        return "skipped: shape mismatch (expected [N, n_layers, hidden_dim])"
    if n_layers < 9:
        # Need at least 9 layers to read layer 8 (boundary).
        return "skipped: shape mismatch (expected [N, n_layers, hidden_dim])"

    # Float32 for numerical stability of cosine.
    h_clean_f = h_clean.to(torch.float32)
    h_corr_f = h_corr.to(torch.float32)

    eps = 1e-12
    norm_clean = h_clean_f.norm(dim=-1).clamp_min(eps)
    norm_corr = h_corr_f.norm(dim=-1).clamp_min(eps)
    dot = (h_clean_f * h_corr_f).sum(dim=-1)
    cos_sim = dot / (norm_clean * norm_corr)
    cos_dist = 1.0 - cos_sim  # [N, n_layers]
    l2_dist = (h_clean_f - h_corr_f).norm(dim=-1)  # [N, n_layers]

    boundary_drift = cos_dist[:, 8].tolist()
    mean_drift = cos_dist.mean(dim=1).tolist()
    max_drift = cos_dist.max(dim=1).values.tolist()
    downstream_mean = (
        cos_dist[:, 9:].mean(dim=1).tolist() if n_layers > 9 else [0.0] * n_samples
    )
    boundary_l2 = l2_dist[:, 8].tolist()
    mean_l2 = l2_dist.mean(dim=1).tolist()
    max_l2 = l2_dist.max(dim=1).values.tolist()
    downstream_mean_l2 = (
        l2_dist[:, 9:].mean(dim=1).tolist() if n_layers > 9 else [0.0] * n_samples
    )

    def _safe_auroc(scores: List[float], labels: List[bool]) -> Optional[float]:
        # Mann-Whitney U based AUROC (no scikit dep). None on degenerate input.
        pos = [s for s, y in zip(scores, labels) if y]
        neg = [s for s, y in zip(scores, labels) if not y]
        if not pos or not neg:
            return None
        n_pos = len(pos)
        n_neg = len(neg)
        wins = 0.0
        for p in pos:
            for q in neg:
                if p > q:
                    wins += 1.0
                elif p == q:
                    wins += 0.5
        return float(wins / (n_pos * n_neg))

    def _pearson(x: List[float], y: List[float]) -> Optional[float]:
        if len(x) != len(y) or len(x) < 2:
            return None
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if float(x_arr.std()) == 0.0 or float(y_arr.std()) == 0.0:
            return None
        return float(np.corrcoef(x_arr, y_arr)[0, 1])

    def _spearman(x: List[float], y: List[float]) -> Optional[float]:
        if len(x) != len(y) or len(x) < 2:
            return None

        def _rank_local(v: List[float]) -> List[float]:
            order = sorted(range(len(v)), key=lambda i: v[i])
            ranks = [0.0] * len(v)
            i = 0
            while i < len(order):
                j = i
                while j < len(order) and v[order[j]] == v[order[i]]:
                    j += 1
                avg = (i + j + 1) / 2.0
                for k in range(i, j):
                    ranks[order[k]] = avg
                i = j
            return ranks

        return _pearson(_rank_local(x), _rank_local(y))

    s_flipped = subsets["S_flipped"]
    s_broken = subsets["S_broken"]
    s_repaired = subsets["S_repaired"]
    s_stable_correct = subsets["S_stable_correct"]
    s_stable_wrong = subsets["S_stable_wrong"]
    s_correct_flipped = subsets["S_correct_flipped"]

    auroc_b_to_flip = _safe_auroc(boundary_drift, s_flipped)
    auroc_b_to_broken = _safe_auroc(boundary_drift, s_broken)
    point_biserial_b_flip = _pearson(boundary_drift, [float(x) for x in s_flipped])
    spearman_b_flip = _spearman(boundary_drift, [float(x) for x in s_flipped])

    auroc_m_to_flip = _safe_auroc(mean_drift, s_flipped)
    auroc_m_to_broken = _safe_auroc(mean_drift, s_broken)
    auroc_d_to_flip = _safe_auroc(downstream_mean, s_flipped)
    auroc_d_to_broken = _safe_auroc(downstream_mean, s_broken)

    def _aggregate_per_subset(values: List[float], mask: List[bool]) -> Dict[str, Optional[float]]:
        sel = [v for v, m in zip(values, mask) if m]
        if not sel:
            return {"mean": None, "median": None, "std": None}
        arr = np.asarray(sel, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std(ddof=0)),
        }

    subset_aggregates: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    subset_masks = {
        "S_stable_correct": s_stable_correct,
        "S_broken": s_broken,
        "S_repaired": s_repaired,
        "S_stable_wrong": s_stable_wrong,
        "S_flipped": s_flipped,
        "S_correct_flipped": s_correct_flipped,
    }
    metric_series = {
        "boundary_layer_drift": boundary_drift,
        "mean_layer_drift": mean_drift,
        "max_layer_drift": max_drift,
        "downstream_mean_drift": downstream_mean,
        "boundary_layer_l2": boundary_l2,
        "mean_layer_l2": mean_l2,
        "max_layer_l2": max_l2,
        "downstream_mean_l2": downstream_mean_l2,
    }
    for sname, mask in subset_masks.items():
        subset_aggregates[sname] = {
            mname: _aggregate_per_subset(series, mask)
            for mname, series in metric_series.items()
        }

    drift_summary: Dict[str, Any] = {
        "primary_drift_variable": "boundary_layer_drift",
        "n_samples": n_samples,
        "n_layers": n_layers,
        "boundary_layer_index": 8,
        "subset_aggregates": subset_aggregates,
        "auroc_boundary_drift_to_answer_flip": auroc_b_to_flip,
        "auroc_boundary_drift_to_broken": auroc_b_to_broken,
        "point_biserial_boundary_drift_vs_flip": point_biserial_b_flip,
        "spearman_boundary_drift_vs_flip": spearman_b_flip,
        "auroc_mean_drift_to_answer_flip": auroc_m_to_flip,
        "auroc_mean_drift_to_broken": auroc_m_to_broken,
        "auroc_downstream_drift_to_answer_flip": auroc_d_to_flip,
        "auroc_downstream_drift_to_broken": auroc_d_to_broken,
    }
    with open(
        os.path.join(run_dir, "phase_b_drift_behavior_summary.json"), "w",
        encoding="utf-8",
    ) as f:
        json.dump(drift_summary, f, indent=2, ensure_ascii=False)

    # Per-sample CSV (spec §7.3).
    with open(
        os.path.join(run_dir, "phase_b_drift_behavior.csv"), "w",
        newline="", encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow([
            "sample_id", "boundary_drift", "mean_cos_drift",
            "max_cos_drift", "downstream_mean_cos_drift",
            "boundary_l2", "mean_l2", "max_l2", "downstream_mean_l2",
            "in_S_flipped", "in_S_broken", "in_S_repaired",
        ])
        for i, sid in enumerate(sample_ids):
            w.writerow([
                sid,
                boundary_drift[i],
                mean_drift[i],
                max_drift[i],
                downstream_mean[i],
                boundary_l2[i],
                mean_l2[i],
                max_l2[i],
                downstream_mean_l2[i],
                str(bool(s_flipped[i])).lower(),
                str(bool(s_broken[i])).lower(),
                str(bool(s_repaired[i])).lower(),
            ])
    return "computed"


def _resolve_anchor_jsonl_path(
    source_label: Optional[str],
    *,
    phase_a_summary_path: Optional[str],
    stage1_evaluation_path: Optional[str],
    condition: str,
) -> Optional[str]:
    """Map an anchor source label + condition name to the per-sample JSONL path.

    Stage 1 / Phase A persist results as ``results_<condition>.jsonl``
    alongside their summary / evaluation manifests (see
    ``stage1/inference/runner.py`` callers and the on-disk anchor layout
    ``run_*/results_*.jsonl``).
    """
    if source_label == "phase_a" and phase_a_summary_path:
        anchor_dir = os.path.dirname(phase_a_summary_path)
        return os.path.join(anchor_dir, f"results_{condition}.jsonl")
    if source_label == "stage1" and stage1_evaluation_path:
        anchor_dir = os.path.dirname(stage1_evaluation_path)
        return os.path.join(anchor_dir, f"results_{condition}.jsonl")
    return None


def _verify_canonical_anchor_pin(
    *,
    sanity: bool,
    phase_a_summary_path: Optional[str],
    stage1_evaluation_path: Optional[str],
) -> Optional[str]:
    """Return a hard-fail message if either resolved anchor != canonical pin.

    Per spec §9 / operator constraint #7: the canonical Stage 1 anchor is
    ``run_20260427_020843_372153`` and the canonical Phase A anchor is
    ``run_20260427_104320_510816``. In sanity mode, missing anchors are
    handled by §7.2.4 (explicit-skipped artifacts) and we do NOT enforce
    pinning when an anchor is absent — only when it is present and
    different from canonical.
    """
    msgs: List[str] = []
    if stage1_evaluation_path is not None:
        st_dir = os.path.basename(os.path.dirname(stage1_evaluation_path))
        if st_dir != CANONICAL_STAGE1_ANCHOR:
            msgs.append(
                f"anchor pin drift: expected stage1/outputs/"
                f"{CANONICAL_STAGE1_ANCHOR}, gate selected stage1/outputs/"
                f"{st_dir}"
            )
    if phase_a_summary_path is not None:
        pa_dir = os.path.basename(os.path.dirname(phase_a_summary_path))
        if pa_dir != CANONICAL_PHASE_A_ANCHOR:
            msgs.append(
                f"anchor pin drift: expected stage1/outputs/phase_a/"
                f"{CANONICAL_PHASE_A_ANCHOR}, gate selected stage1/outputs/"
                f"phase_a/{pa_dir}"
            )
    if not msgs:
        return None
    if sanity:
        # In sanity mode we still surface the diagnostic via summary, but
        # do not hard-fail (keeps sanity smokes green when the canonical
        # anchors are absent on the dev box).
        return None
    return " | ".join(msgs)


# ─── Main entrypoint ─────────────────────────────────────────────────────────

def run_phase_b(
    config_path: str,
    sanity: bool = False,
    seed: int = 42,
    run_name: Optional[str] = None,
) -> str:
    """Run Phase B restoration intervention experiment.

    Primary treatment: ``hard_swap_b8``. Primary reference: ``no_swap``.

    Returns:
        Path to the run directory.
    """
    setup_logging()

    # (1) Determinism flags FIRST (before any CUDA op via model load).
    determinism_warnings = _apply_determinism(seed)

    config = load_config(config_path)

    if sanity:
        config.dataset.debug_n = 5

    print("Phase B — Restoration Intervention (prompt-side)")
    print(f"  Config     : {config_path}")
    print(f"  Sanity mode: {sanity}")
    print(f"  Seed       : {seed}")
    print(f"  Run name   : {run_name}")

    run_dir = create_run_dir(base_dir="stage1/outputs/phase_b")
    print(f"  Run dir    : {run_dir}")

    # (2) Load data
    samples = load_mgsm(config)
    print(f"  Samples    : {len(samples)}")

    # (3) Load models
    print("Loading models...")
    recipient, donor, tokenizer = load_models(
        recipient_name=config.models.recipient,
        donor_name=config.models.donor,
        recipient_revision=config.models.recipient_revision,
        donor_revision=config.models.donor_revision,
    )
    n_layers = recipient.config.num_hidden_layers
    hidden_size = recipient.config.hidden_size
    print(f"  Layers     : {n_layers}, hidden_size: {hidden_size}")
    if n_layers != 28 or hidden_size != 1536:
        raise RuntimeError(
            f"Architecture assertion failed: expected num_hidden_layers=28, "
            f"hidden_size=1536; got {n_layers}, {hidden_size}."
        )

    # (4) Compose hard_swap_b8 (b=8, t=20).
    t_fixed = 20
    print(f"Composing hard_swap_b8 (b=8, t={t_fixed})...")
    composed, compose_meta = compose_model(
        recipient, donor, b=8, t=t_fixed, condition="hard_swap",
    )
    compose_meta = dict(compose_meta)  # copy; augment with hashes
    compose_meta["b"] = 8
    compose_meta["t"] = t_fixed
    compose_meta["condition"] = "hard_swap_b8"

    # Record state_dict SHA-256 before any inference (spec §10 test #4).
    sd_sha_before = _state_dict_sha256(composed)
    compose_meta["state_dict_sha256_before"] = sd_sha_before

    # (5) Free donor immediately after compose (spec §12 R6, memory rule).
    del donor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gen_config = {
        "do_sample":      config.generation.do_sample,
        "temperature":    config.generation.temperature,
        "max_new_tokens": config.generation.max_new_tokens,
    }

    # (6) Build patch configs.
    restoration_configs = [
        PatchConfig(name, layers, "restoration")
        for name, layers in RESTORATION_PATCHES.items()
    ]
    corruption_configs = [
        PatchConfig(name, layers, "corruption")
        for name, layers in REVERSE_CORRUPTION_PATCHES.items()
    ]

    if sanity:
        # Sanity-mode conditions per spec §7:
        # {no_patch, patch_recovery_full, clean_no_patch (recipient),
        #  corrupt_recovery_full} × 5 samples.
        keep_r = {"no_patch", "patch_recovery_full"}
        restoration_configs = [pc for pc in restoration_configs if pc.patch_name in keep_r]
        keep_c = {"corrupt_recovery_full"}
        corruption_configs = [pc for pc in corruption_configs if pc.patch_name in keep_c]

    # (7) Restoration interventions on composed.
    print(f"\n{'=' * 60}")
    print("RESTORATION INTERVENTIONS (composed ← clean patches)")
    print(f"{'=' * 60}")

    restoration_results: Dict[str, List[Dict]] = {}
    for pc in restoration_configs:
        print(f"\n--- {pc.patch_name} (layers: {pc.patch_layers}) ---")
        results = run_patched_inference(
            target_model=composed,
            recipient_model=recipient,
            composed_model=None,
            tokenizer=tokenizer,
            samples=samples,
            patch_config=pc,
            generation_config=gen_config,
        )
        _annotate_correct(results, samples)
        restoration_results[pc.patch_name] = results
        _save_condition_results(run_dir, f"restoration_{pc.patch_name}", results)

    # (8) Clean baseline: no_patch on recipient (spec §7).
    print(f"\n{'=' * 60}")
    print("CLEAN BASELINE (recipient, no intervention)")
    print(f"{'=' * 60}")
    clean_baseline_results = run_patched_inference(
        target_model=recipient,
        recipient_model=recipient,
        composed_model=None,
        tokenizer=tokenizer,
        samples=samples,
        patch_config=PatchConfig("clean_no_patch", [], "restoration"),
        generation_config=gen_config,
    )
    _annotate_correct(clean_baseline_results, samples)
    _save_condition_results(run_dir, "clean_no_patch", clean_baseline_results)

    # (9) Reverse corruption interventions on recipient.
    print(f"\n{'=' * 60}")
    print("REVERSE CORRUPTION (recipient ← corrupt patches)")
    print(f"{'=' * 60}")
    corruption_results: Dict[str, List[Dict]] = {}
    for pc in corruption_configs:
        print(f"\n--- {pc.patch_name} (layers: {pc.patch_layers}) ---")
        results = run_patched_inference(
            target_model=recipient,
            recipient_model=recipient,
            composed_model=composed,
            tokenizer=tokenizer,
            samples=samples,
            patch_config=pc,
            generation_config=gen_config,
        )
        _annotate_correct(results, samples)
        corruption_results[pc.patch_name] = results
        _save_condition_results(run_dir, f"corruption_{pc.patch_name}", results)

    # (10) Accuracy tables.
    no_patch_acc = _compute_acc(restoration_results.get("no_patch", []))
    clean_baseline_acc = _compute_acc(clean_baseline_results)

    restoration_table: List[Dict[str, Any]] = []
    for name in RESTORATION_PATCHES:
        if name not in restoration_results:
            continue
        acc = _compute_acc(restoration_results[name])
        restoration_table.append({
            "condition": name,
            "accuracy": round(acc, 4),
            "delta_from_no_patch": round(acc - no_patch_acc, 4),
            "delta_from_clean_baseline": round(acc - clean_baseline_acc, 4),
            "methodology": METHODOLOGY_TAG,
        })

    corruption_table: List[Dict[str, Any]] = []
    for name in REVERSE_CORRUPTION_PATCHES:
        if name not in corruption_results:
            continue
        acc = _compute_acc(corruption_results[name])
        corruption_table.append({
            "condition": name,
            "accuracy": round(acc, 4),
            "delta_from_clean_baseline": round(acc - clean_baseline_acc, 4),
            "methodology": METHODOLOGY_TAG,
        })

    # (11) Write CSV tables.
    _write_csv(
        os.path.join(run_dir, "restoration_table.csv"),
        ["condition", "accuracy", "delta_from_no_patch",
         "delta_from_clean_baseline", "methodology"],
        restoration_table,
    )
    _write_csv(
        os.path.join(run_dir, "corruption_table.csv"),
        ["condition", "accuracy", "delta_from_clean_baseline", "methodology"],
        corruption_table,
    )

    # (12) Cross-phase check: treatment parity + baseline parity.
    # Phase A's confound grid does NOT contain hard_swap_b8 (it uses
    # fixed_w4_posN / fixed_b8_wN — the original b=8,t=20 anchor is absent).
    # We therefore source hard_swap_b8 from the latest Stage 1 sweep run
    # (stage1/outputs/run_*/evaluation.json) and fall back to Phase A only for
    # the no_swap baseline.
    # RED LIGHT P2: anchor selection filtered by manifest parity.
    # YELLOW-LIGHT v3 P1: gate decision delegated to ``utils.anchor_gate`` so
    # full integration-level regression tests can exercise it end-to-end.
    # v4 P2: include sample_regime in the parity block so anchors with a
    # different debug/full mode, sample count, or sample ordering are rejected.
    current_parity = extract_parity_block(
        config, sample_ids=[s["sample_id"] for s in samples],
    )
    gate = evaluate_phase_b_anchor_gate(
        no_patch_acc=no_patch_acc,
        clean_baseline_acc=clean_baseline_acc,
        sanity=sanity,
        current_parity=current_parity,
        tolerance=PHASE_A_CROSS_CHECK_TOL,
        phase_a_dir=_phase_a_outputs_dir(),
        stage1_dir=_stage1_outputs_dir(),
    )
    # Local aliases mirror the legacy variables so the rest of this function
    # (summary block, sanity-check labels) reads identically to before.
    phase_a_path = gate.phase_a_summary_path
    phase_a_hard_swap_acc = gate.phase_a_hard_swap_b8_accuracy
    phase_a_no_swap_acc = gate.phase_a_no_swap_accuracy
    stage1_eval_path = gate.stage1_evaluation_path
    stage1_hs_acc = gate.stage1_hard_swap_b8_accuracy
    stage1_ns_acc = gate.stage1_no_swap_accuracy
    anchor_hard_swap_acc = gate.anchor_hard_swap_b8_accuracy
    anchor_no_swap_acc = gate.anchor_no_swap_accuracy
    anchor_hard_swap_source = gate.anchor_hard_swap_source
    anchor_no_swap_source = gate.anchor_no_swap_source
    cross_check_passed = gate.passed
    cross_check_missing_anchors = list(gate.missing_anchors)
    cross_check_failed_anchors = list(gate.failed_anchors)
    # Re-load the Phase A summary dict for the dataset-mirror block below.
    # (We don't reuse the loader; the path is enough.)
    phase_a: Optional[Dict[str, Any]] = None
    if phase_a_path is not None:
        try:
            with open(phase_a_path, encoding="utf-8") as _f:
                phase_a = json.load(_f)
        except Exception:
            phase_a = None

    # (12.5) Subset construction + conditional metrics (spec §7.2).
    # Pre-condition: gate has been computed (step 12) so we know which
    # canonical Stage 1 / Phase A anchors are in play. The conditional
    # metrics layer DEPENDS on Stage 1 anchor JSONLs being available;
    # behavior on missing anchors is mode-dependent (spec §7.2.4).
    sample_ids: List[str] = [s["sample_id"] for s in samples]
    subsets: Optional[Dict[str, List[bool]]] = None
    subsets_summary: Dict[str, Any] = {}
    subset_warnings: List[str] = []
    per_condition_metrics: Dict[str, Dict[str, Any]] = {}
    correct_field_source: Dict[str, str] = {}
    conditional_status: str = "pending"
    conditional_skip_reason: Optional[str] = None
    ans_clean_global: List[Optional[str]] = []
    ans_corrupt_global: List[Optional[str]] = []
    correct_clean_global: List[bool] = []
    correct_corrupt_global: List[bool] = []
    recomputed_clean_global: List[bool] = []
    recomputed_corrupt_global: List[bool] = []
    stage1_anchor_dir: Optional[str] = None
    drift_diagnostic_status: str = "skipped: anchor_dir unavailable"
    parity_failure_reasons: List[str] = []

    # r3: Anchor-accuracy parity pre-check (Codex BLOCK fix — gate-ordering).
    # The anchor gate's `failed_anchors` list is non-empty when BOTH anchors
    # were located AND at least one accuracy delta is outside
    # ``PHASE_A_CROSS_CHECK_TOL``. Previously this signal was applied to the
    # late sanity-checks block AFTER ``_emit_conditional_artifacts`` had
    # already computed and persisted ``phase_b_subsets.{json,csv}`` /
    # ``phase_b_conditional_summary.{json,csv}``. That violated the spec
    # contract that conditional metrics MUST be skipped/invalid when no-patch
    # accuracy parity fails. We now lift the accuracy-delta check upstream
    # so it routes through the existing ``parity_failure_reasons`` branch
    # (line ~1694 below) and emits the explicit-skipped artifacts before
    # any computed numbers are written.
    #
    # Mode-dependent semantics:
    #   * Full mode: ``gate.failed_anchors`` is populated only when both
    #     anchors are present and at least one is outside tolerance — exactly
    #     the case Codex flagged. Pre-check fires.
    #   * Sanity mode: ``gate.passed`` is ``True`` / ``False`` only when at
    #     least one anchor is available; ``failed_anchors`` is populated only
    #     in the rare sanity-with-real-anchors path. Per spec §7.2.4 the
    #     existing sanity skipped path already handles the ordinary
    #     anchors-absent case; the new pre-check is layered on top so that
    #     IF real anchors are present in sanity mode AND accuracy parity
    #     fails, the skipped path is taken with the new reason rather than
    #     silently producing computed numbers.
    (
        _aa_status,
        anchor_accuracy_parity_metrics,
        anchor_accuracy_parity_failed_msgs,
        _aa_reason,
    ) = _evaluate_anchor_accuracy_parity_precheck(
        cross_check_failed_anchors=cross_check_failed_anchors,
        no_patch_acc=no_patch_acc,
        clean_baseline_acc=clean_baseline_acc,
        anchor_hard_swap_acc=anchor_hard_swap_acc,
        anchor_no_swap_acc=anchor_no_swap_acc,
        tolerance=PHASE_A_CROSS_CHECK_TOL,
    )
    if _aa_status == "failed":
        anchor_accuracy_parity_status = "failed"
        # Append the helper-built reason; the existing branch at line ~1765
        # (`if parity_failure_reasons:`) routes us through the explicit
        # ``_emit_conditional_artifacts_skipped`` path BEFORE any computed
        # numbers are written. This is the Codex BLOCK fix.
        if _aa_reason is not None:
            parity_failure_reasons.append(_aa_reason)
    else:
        anchor_accuracy_parity_status = (
            "n/a (full mode)" if not sanity else "n/a (sanity mode)"
        )

    # Resolve anchor JSONL paths.
    no_swap_jsonl = _resolve_anchor_jsonl_path(
        anchor_no_swap_source,
        phase_a_summary_path=phase_a_path,
        stage1_evaluation_path=stage1_eval_path,
        condition="no_swap",
    )
    hard_swap_jsonl = _resolve_anchor_jsonl_path(
        anchor_hard_swap_source,
        phase_a_summary_path=phase_a_path,
        stage1_evaluation_path=stage1_eval_path,
        condition="hard_swap_b8",
    )
    if stage1_eval_path is not None:
        stage1_anchor_dir = os.path.dirname(stage1_eval_path)

    # Anchor pin verification (spec §9, operator constraint #7). Hard-fail
    # in full mode if the gate selected a non-canonical anchor; in sanity
    # mode this is informational only.
    pin_drift_msg = _verify_canonical_anchor_pin(
        sanity=sanity,
        phase_a_summary_path=phase_a_path,
        stage1_evaluation_path=stage1_eval_path,
    )
    if pin_drift_msg:
        # Defer raising until the sanity-check loop so the failure is
        # surfaced through the normal _persist_summary(RUN_STATUS_FAILED)
        # path rather than a bare RuntimeError. We register it via the
        # parity_failure_reasons list which is also surfaced into
        # subset_warnings (informational).
        parity_failure_reasons.append(pin_drift_msg)

    anchors_loadable = (
        no_swap_jsonl is not None
        and hard_swap_jsonl is not None
        and os.path.exists(no_swap_jsonl)
        and os.path.exists(hard_swap_jsonl)
    )

    if not anchors_loadable:
        if sanity:
            conditional_skip_reason = (
                "sanity mode: stage1 anchors unavailable for subset "
                "construction"
            )
            conditional_status = "skipped"
            subsets_summary = _emit_conditional_artifacts_skipped(
                run_dir, conditional_skip_reason, n_total=len(samples),
            )
            subset_warnings = [
                "sanity mode: subset construction skipped"
            ]
        else:
            # Full mode: gate already hard-fails on missing anchors via the
            # cross-check sanity-checks block below. We still emit nothing
            # here (precondition unsatisfied per spec §7.2.4); the run will
            # FAIL when checks are evaluated.
            conditional_skip_reason = (
                "full mode: stage1 anchors unavailable for subset "
                "construction (cross-check gate will fail)"
            )
            conditional_status = "skipped"
    else:
        try:
            clean_anchor_rows, label_clean = _load_anchor_per_sample(
                no_swap_jsonl, samples,
            )
            corrupt_anchor_rows, label_corrupt = _load_anchor_per_sample(
                hard_swap_jsonl, samples,
            )
        except Exception as exc:
            # Strict-fail per spec §7.2.1 (evaluator drift) propagates.
            raise

        ans_clean_global = [
            r.get("normalized_answer") for r in clean_anchor_rows
        ]
        ans_corrupt_global = [
            r.get("normalized_answer") for r in corrupt_anchor_rows
        ]
        correct_clean_global = [bool(r["correct"]) for r in clean_anchor_rows]
        correct_corrupt_global = [
            bool(r["correct"]) for r in corrupt_anchor_rows
        ]
        recomputed_clean_global = [
            bool(r["recomputed_correct"]) for r in clean_anchor_rows
        ]
        recomputed_corrupt_global = [
            bool(r["recomputed_correct"]) for r in corrupt_anchor_rows
        ]
        correct_field_source = {
            "clean": label_clean,
            "corrupt": label_corrupt,
        }

        subsets = _build_subsets(clean_anchor_rows, corrupt_anchor_rows)

        # P0 generation parity check (operator constraint #4 + spec
        # §11.4/§11.5/§11.6). If clean_no_patch or restoration_no_patch
        # diverges from the matched Stage 1 anchor in (a) accuracy, (b)
        # Human:-continuation count, or (c) mean output length, the
        # conditional metrics layer MUST emit explicit failed/skipped
        # status artifacts with reasons rather than silently produce
        # numbers. This is the explicit-skipped artifact format from
        # spec §7.2.4.
        clean_no_patch_rows = clean_baseline_results
        restoration_no_patch_rows = restoration_results.get("no_patch", [])

        stage1_clean_no_patch_human_count = _count_human_continuation(
            clean_anchor_rows,
        )
        stage1_restoration_no_patch_human_count = _count_human_continuation(
            corrupt_anchor_rows,
        )
        clean_no_patch_human_count = _count_human_continuation(
            clean_no_patch_rows,
        )
        restoration_no_patch_human_count = _count_human_continuation(
            restoration_no_patch_rows,
        )
        stage1_clean_mean_len = _mean_output_length_chars(clean_anchor_rows)
        stage1_corrupt_mean_len = _mean_output_length_chars(
            corrupt_anchor_rows,
        )
        clean_no_patch_mean_len = _mean_output_length_chars(
            clean_no_patch_rows,
        )
        restoration_no_patch_mean_len = _mean_output_length_chars(
            restoration_no_patch_rows,
        )

        # Spec §11.4 accuracy parity (delegate to gate; we just record).
        # Spec §11.5 Human:-continuation parity (≤ stage1_count + 2).
        if (clean_no_patch_human_count
                > stage1_clean_no_patch_human_count + 2):
            parity_failure_reasons.append(
                "P0 generation parity failed: clean_no_patch Human: "
                f"continuation count = {clean_no_patch_human_count} "
                f"exceeds Stage 1 no_swap anchor count "
                f"({stage1_clean_no_patch_human_count}) by more than 2"
            )
        if (restoration_no_patch_human_count
                > stage1_restoration_no_patch_human_count + 2):
            parity_failure_reasons.append(
                "P0 generation parity failed: restoration_no_patch did "
                "not match Stage1 hard_swap_b8 anchor (Human: continuation "
                f"count = {restoration_no_patch_human_count} vs Stage 1 "
                f"{stage1_restoration_no_patch_human_count})"
            )
        # Spec §11.6 output-length parity (≤ 1.25× Stage 1 reference).
        if stage1_clean_mean_len > 0 and clean_no_patch_mean_len > (
            1.25 * stage1_clean_mean_len
        ):
            parity_failure_reasons.append(
                "P0 generation parity failed: clean_no_patch did not "
                "match Stage1 no_swap anchor (mean output length "
                f"{clean_no_patch_mean_len:.1f} > 1.25 x "
                f"{stage1_clean_mean_len:.1f})"
            )
        if stage1_corrupt_mean_len > 0 and restoration_no_patch_mean_len > (
            1.25 * stage1_corrupt_mean_len
        ):
            parity_failure_reasons.append(
                "P0 generation parity failed: restoration_no_patch did "
                "not match Stage1 hard_swap_b8 anchor (mean output length "
                f"{restoration_no_patch_mean_len:.1f} > 1.25 x "
                f"{stage1_corrupt_mean_len:.1f})"
            )

        # Operator constraint #4: when parity fails, the conditional
        # metrics layer MUST emit explicit failed/skipped artifacts.
        if parity_failure_reasons:
            conditional_status = "skipped"
            conditional_skip_reason = parity_failure_reasons[0]
            subsets_summary = _emit_conditional_artifacts_skipped(
                run_dir, conditional_skip_reason, n_total=len(samples),
            )
            subset_warnings = list(parity_failure_reasons)
        else:
            # Full conditional metrics path (spec §7.2.2 + §7.2.5).
            condition_records: List[Tuple[str, str, List[Dict]]] = []
            # Restoration direction (composed): keys are flat patch names.
            for cname, rows in restoration_results.items():
                # Flat restoration_no_patch alias (spec §11.8 / §11
                # naming): the Stage 1-equivalent reference for the
                # restoration direction is the composed-model no_patch.
                emitted_name = (
                    "restoration_no_patch" if cname == "no_patch" else cname
                )
                condition_records.append(
                    (emitted_name, "restoration", rows)
                )
            # clean_no_patch is the composed-vs-recipient baseline
            # (recipient, no patches) and is ALSO emitted under its own
            # key for the §11 artifact schema.
            condition_records.append(
                ("clean_no_patch", "restoration", clean_baseline_results),
            )
            # Corruption direction (recipient): keys prefixed with
            # ``corruption_`` per spec §7.2.5 to keep restoration and
            # corruption flat-namespaced.
            for cname, rows in corruption_results.items():
                condition_records.append(
                    (f"corruption_{cname}", "corruption", rows),
                )

            no_patch_ref_restoration = restoration_results.get(
                "no_patch", []
            )
            no_patch_ref_corruption = clean_baseline_results

            for emitted_name, direction, rows in condition_records:
                no_patch_ref = (
                    no_patch_ref_restoration
                    if direction == "restoration"
                    else no_patch_ref_corruption
                )
                per_condition_metrics[emitted_name] = (
                    _compute_conditional_metrics_one_condition(
                        pc_rows=rows,
                        no_patch_rows=no_patch_ref,
                        ans_clean=ans_clean_global,
                        ans_corrupt=ans_corrupt_global,
                        correct_clean=correct_clean_global,
                        correct_corrupt=correct_corrupt_global,
                        subsets=subsets,
                        direction=direction,
                    )
                )

            subsets_summary = _emit_conditional_artifacts(
                run_dir=run_dir,
                subsets=subsets,
                sample_ids=sample_ids,
                ans_clean=ans_clean_global,
                ans_corrupt=ans_corrupt_global,
                correct_clean=correct_clean_global,
                correct_corrupt=correct_corrupt_global,
                recomputed_clean=recomputed_clean_global,
                recomputed_corrupt=recomputed_corrupt_global,
                per_condition_metrics=per_condition_metrics,
            )
            subset_warnings = _subset_warning_strings(subsets_summary)
            conditional_status = "computed"

            # Optional drift-vs-behavior diagnostic (spec §7.3).
            drift_diagnostic_status = _maybe_compute_drift_diagnostic(
                run_dir=run_dir,
                anchor_dir=stage1_anchor_dir,
                sample_ids=sample_ids,
                subsets=subsets,
            )

    # (13) Environment block.
    # Stage 1 hardening (2026-04-25): merge in the canonical runtime_provenance
    # so Phase B manifests carry the same git_sha / versions / command /
    # dataset-revision/sha256 fields as Phase A, in addition to the Phase B
    # specific determinism + device fields.
    runtime_provenance = build_runtime_provenance(
        config=config, config_path=config_path,
    )
    env_block: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": str(next(composed.parameters()).device),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "git_sha": _git_sha(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "deterministic_algorithms_enabled": True,
        "determinism_warnings": determinism_warnings,
        "seed": seed,
        # RED LIGHT P4: record generation config for decode-budget traceability.
        "generation_config": gen_config,
        "runtime_provenance": runtime_provenance,
    }

    # (14) Build summary JSON.
    dataset_block: Dict[str, Any] = {
        "name": config.dataset.name,
        "lang": config.dataset.lang,
        "split": config.dataset.split,
        "n_samples": len(samples),
    }
    # Mirror the Phase A dataset/manifest entry verbatim when available.
    if phase_a is not None and "dataset" in phase_a:
        dataset_block["phase_a_dataset_manifest"] = phase_a["dataset"]

    summary: Dict[str, Any] = {
        "phase": "B",
        "description": "Restoration intervention — prompt-side hidden-state patching",
        "treatment_condition": "hard_swap_b8",
        "reference_condition": "no_swap",
        "t_fixed": t_fixed,
        "epsilon_delta": EPSILON_DELTA,
        "sanity_mode": sanity,
        "mode": "sanity" if sanity else "full",
        "run_name": run_name,
        "seed": seed,
        "no_patch_accuracy": round(no_patch_acc, 4),
        "clean_baseline_accuracy": round(clean_baseline_acc, 4),
        "restoration_table": restoration_table,
        "corruption_table": corruption_table,
        "methodological_constraint": METHODOLOGICAL_CONSTRAINT,
        "phase_a_cross_check": gate.to_summary_dict(
            phase_a_outputs_dir=_phase_a_outputs_dir(),
        ),
        "compose_meta": compose_meta,
        "dataset": dataset_block,
        "environment": env_block,
        # RED LIGHT P2: embed parity block for downstream manifest checks.
        "parity": current_parity,
        # phase_b_revision §8.2 additive fields.
        "subset_summary": (
            {"status": "skipped", "reason": conditional_skip_reason}
            if conditional_status == "skipped"
            else dict(subsets_summary)
        ),
        "subset_warnings": list(subset_warnings),
        "drift_diagnostic": drift_diagnostic_status,
        "correct_field_source": (
            correct_field_source if correct_field_source
            else {"clean": "unavailable", "corrupt": "unavailable"}
        ),
        "conditional_metrics_status": conditional_status,
        "p0_parity_failure_reasons": list(parity_failure_reasons),
        # r3: Codex BLOCK fix — anchor-accuracy parity gate-ordering result.
        # Set unconditionally so downstream tooling can rely on field
        # presence in every summary regardless of mode / gate outcome.
        "anchor_accuracy_parity_status": anchor_accuracy_parity_status,
        "anchor_accuracy_parity_metrics": dict(anchor_accuracy_parity_metrics),
    }

    # (15) Comparative-sentence gate (spec §7, §11.10).
    non_nopatch = [r for r in restoration_table if r["condition"] != "no_patch"]
    # Claims restricted to the 4 corruption-mirrored conditions (§11).
    claim_eligible = [r for r in non_nopatch if r["condition"] in (
        "patch_boundary_local", "patch_recovery_early",
        "patch_recovery_full", "patch_final_only",
    )]
    comparative_sentence: str
    comparative_block: Dict[str, Any] = {"fired": False}
    if len(claim_eligible) >= 2 and all(
        r["condition"] in restoration_results for r in claim_eligible
    ):
        best = max(claim_eligible, key=lambda r: r["delta_from_no_patch"])
        boundary_local = next(
            (r for r in claim_eligible if r["condition"] == "patch_boundary_local"), None,
        )
        if boundary_local is not None and best["condition"] != "patch_boundary_local":
            best_corr = [int(x.get("correct", False))
                         for x in restoration_results[best["condition"]]]
            bl_corr = [int(x.get("correct", False))
                       for x in restoration_results["patch_boundary_local"]]
            # Pair on sample ordering, which matches across conditions by construction.
            point, ci_lo, ci_hi = _paired_bootstrap_diff_ci(
                best_corr, bl_corr, n_resamples=1000, seed=0, ci=0.95,
            )
            both_positive = (best["delta_from_no_patch"] > 0.0
                             and boundary_local["delta_from_no_patch"] > 0.0)
            # NB: this boolean is the comparative-sentence gate. Do NOT name it
            # ``gate`` — the outer-scope ``gate`` is the AnchorGateResult from
            # ``evaluate_phase_b_anchor_gate`` and must remain accessible to
            # ``render_anchor_gate_diagnostic`` on the failure path below.
            comparative_gate = both_positive and point > EPSILON_DELTA and ci_lo > 0.0
            comparative_block = {
                "fired": comparative_gate,
                "best_condition": best["condition"],
                "best_delta": best["delta_from_no_patch"],
                "boundary_local_delta": boundary_local["delta_from_no_patch"],
                "point_estimate_diff": point,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "epsilon_delta": EPSILON_DELTA,
            }
            if comparative_gate:
                comparative_sentence = (
                    f"Recovery-side intervention at {best['condition']} yields a larger "
                    f"prompt-side accuracy delta than patch_boundary_local "
                    f"(point={point:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], "
                    f"eps={EPSILON_DELTA})."
                )
            else:
                comparative_sentence = (
                    "Restoration deltas do not meet the effect-size threshold for a "
                    "directional claim "
                    f"(point={point:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], "
                    f"eps={EPSILON_DELTA})."
                )
        else:
            comparative_sentence = (
                "Restoration deltas do not meet the effect-size threshold for a "
                "directional claim (insufficient eligible conditions)."
            )
    else:
        comparative_sentence = (
            "Restoration deltas do not meet the effect-size threshold for a "
            "directional claim (insufficient eligible conditions)."
        )
    summary["comparative_claim"] = comparative_block

    # YELLOW-LIGHT v3 P3: explicit run-status field. Initialised as "pending"
    # and updated to "passed" / "failed" by every ``_persist_summary`` call.
    summary["run_status"] = RUN_STATUS_PENDING
    summary["failure_reason"] = None

    # r3: When the upstream anchor-accuracy parity pre-check fired
    # (``anchor_accuracy_parity_failed_msgs`` non-empty), pre-populate
    # ``summary["failure_reason"]`` with the operator-mandated format so the
    # eventual ``_persist_summary(RUN_STATUS_FAILED, ...)`` call in the
    # failed-checks branch (full mode: late cross-check FAIL label triggers it)
    # surfaces the spec-mandated reason. r4 watcher LOW #2 canonicalisation:
    # the reason-string prefix is the underscore form
    # ("no_patch_anchor_accuracy_parity_failed: ...") on BOTH the artifact
    # JSONs (set inside ``_evaluate_anchor_accuracy_parity_precheck``) and
    # the summary["failure_reason"] field below. The precedence detector in
    # the late ``_persist_summary`` site matches against the same prefix so
    # downstream tooling can grep one canonical form.
    if anchor_accuracy_parity_failed_msgs:
        summary["failure_reason"] = (
            "no_patch_anchor_accuracy_parity_failed: "
            + "; ".join(anchor_accuracy_parity_failed_msgs)
        )

    # (16) Human-readable summary TXT body (contains t=20 header and exactly one
    #      comparative sentence). The leading status banner is prepended by
    #      ``write_phase_b_status_artifacts`` on each persist, so a failed run
    #      cannot be mistaken for a passed one by a human glancing at the file.
    # Operator constraint #6: when sanity mode, the first non-blank body line
    # MUST include the substring [SANITY] or [DEBUG] so a downstream reader
    # cannot mistake a sanity smoke for a full Phase B validation.
    title_line = (
        "[SANITY] PHASE B — RESTORATION INTERVENTION RESULTS"
        if sanity else
        "PHASE B — RESTORATION INTERVENTION RESULTS"
    )
    lines: List[str] = [
        "=" * 60,
        title_line,
        "=" * 60,
        "",
        f"recovery-zone layers [20..27] defined at fixed t={t_fixed}",
        "",
        "Treatment condition: hard_swap_b8",
        "Reference condition: no_swap",
        f"Samples: {len(samples)}",
        f"Sanity mode: {sanity}",
        f"Seed: {seed}",
        "",
        "This is prompt-side restoration intervention only.",
        "Clean hidden states are available for prompt tokens only.",
        "",
        "-" * 60,
        "Restoration table (clean states into composed model)",
        "-" * 60,
        f"{'Condition':<25} {'Accuracy':>8} {'dNoPatch':>10} {'dClean':>10}",
    ]
    for r in restoration_table:
        lines.append(
            f"{r['condition']:<25} {r['accuracy']:>8.4f} "
            f"{r['delta_from_no_patch']:>+10.4f} {r['delta_from_clean_baseline']:>+10.4f}"
        )
    lines += [
        "",
        "-" * 60,
        "Reverse corruption table (corrupt states into recipient)",
        "-" * 60,
        f"{'Condition':<25} {'Accuracy':>8} {'dClean':>10}",
    ]
    for r in corruption_table:
        lines.append(
            f"{r['condition']:<25} {r['accuracy']:>8.4f} "
            f"{r['delta_from_clean_baseline']:>+10.4f}"
        )

    # phase_b_revision §7.2 / §8.2 / §11.10: "Conditional metrics
    # (subset-level)" TXT block. Numbers only — no comparative sentence
    # in this block; the single existing comparative sentence remains
    # gated by the EPSILON_DELTA / CI rule below.
    lines += [
        "",
        "-" * 60,
        "Conditional metrics (subset-level)",
        "-" * 60,
    ]
    if conditional_status == "computed":
        # Per spec §8.2: per restoration patch condition list
        # recovery_to_correct_rate, answer_restoration_rate,
        # stable_correct_preservation_rate, parse_success_rate_global,
        # human_continuation_rate.
        lines.append(
            f"  subsets: n_total={subsets_summary.get('n_total', 0)} "
            f"stable_correct={subsets_summary.get('n_stable_correct', 0)} "
            f"broken={subsets_summary.get('n_broken', 0)} "
            f"repaired={subsets_summary.get('n_repaired', 0)} "
            f"flipped={subsets_summary.get('n_flipped', 0)}"
        )
        header = (
            f"  {'Condition':<28} {'recov':>8} {'ans_rest':>8} "
            f"{'st_pres':>8} {'parse':>8} {'human':>8}"
        )
        lines.append(header)
        for cname in (
            "restoration_no_patch", "patch_boundary_local",
            "patch_recovery_early", "patch_recovery_full",
            "patch_final_only", "patch_all_downstream",
        ):
            block = per_condition_metrics.get(cname)
            if not block:
                continue
            recov = block.get("broken_subset_recovery", {}).get(
                "recovery_to_correct_rate"
            )
            ar = block.get("answer_flipped_restoration", {}).get(
                "answer_restoration_rate"
            )
            sp = block.get("stable_correct_disruption", {}).get(
                "stable_correct_preservation_rate"
            )
            ps = block.get("parse_behavior", {}).get(
                "parse_success_rate_global"
            )
            hr = block.get("output_behavior", {}).get(
                "human_continuation_rate"
            )

            def _fmt(v):
                return "  n/a  " if v is None else f"{v:>8.4f}"
            lines.append(
                f"  {cname:<28} {_fmt(recov)} {_fmt(ar)} "
                f"{_fmt(sp)} {_fmt(ps)} {_fmt(hr)}"
            )
    else:
        lines.append(
            f"  status: {conditional_status} "
            f"(reason: {conditional_skip_reason or 'unspecified'})"
        )

    if subset_warnings:
        lines.append("")
        for w in subset_warnings:
            lines.append(f"  [warning] {w}")

    lines += [
        "",
        "-" * 60,
        "Interpretation",
        "-" * 60,
        comparative_sentence,
        "",
        "Note: These results are intervention-based evidence under the "
        "prompt-side constraint described above. They do NOT by themselves "
        "establish a complete mechanism; scientific claims remain conservative.",
        "",
    ]
    body_lines = list(lines)  # banner-free TXT body, used by every persist call
    summary_text = "\n".join(body_lines)
    print(summary_text)

    # (17) Record state_dict hash AFTER all inference passes.
    sd_sha_after = _state_dict_sha256(composed)
    summary["compose_meta"]["state_dict_sha256_after"] = sd_sha_after
    if sd_sha_after != sd_sha_before:
        raise RuntimeError(
            "Composed model state_dict SHA-256 changed across the run "
            f"(before={sd_sha_before[:12]}..., after={sd_sha_after[:12]}...). "
            "In-place weight mutation is forbidden."
        )

    # YELLOW-LIGHT v3 P3: every persist of phase_b_summary.{json,txt} +
    # RUN_STATUS.txt goes through the centralised helper. This is the ONLY
    # path that writes those files — there is no flow where a failed run
    # can leave status="pending"/"passed" artifacts on disk.
    def _persist_summary(status: str, failure_reason: Optional[str] = None) -> None:
        write_phase_b_status_artifacts(
            run_dir, summary, body_lines, status,
            failure_reason=failure_reason,
        )
        # Operator constraint #6: in sanity mode, the FIRST non-blank line of
        # phase_b_summary.txt MUST contain the substring "[SANITY]" or
        # "[DEBUG]" so a downstream reader cannot mistake a sanity smoke for
        # a full Phase B validation. ``write_phase_b_status_artifacts`` always
        # prepends a "RUN STATUS:" banner first; we additively prepend a
        # single sanity/debug marker line ABOVE that banner. Done as an
        # additive post-write rewrite so ``run_status.py`` (do-not-touch)
        # is not modified.
        if sanity:
            txt_path = os.path.join(run_dir, "phase_b_summary.txt")
            try:
                with open(txt_path, "r", encoding="utf-8") as _f:
                    _existing_txt = _f.read()
                _marker = (
                    "[SANITY] [DEBUG] — sanity smoke; not a full Phase B "
                    "validation (mode=sanity, see phase_b_summary.json"
                    "[\"mode\"])"
                )
                if not _existing_txt.lstrip().startswith("[SANITY]"):
                    with open(txt_path, "w", encoding="utf-8") as _f:
                        _f.write(_marker + "\n" + _existing_txt)
            except OSError:
                # If for any reason the file is not yet on disk (it should
                # be — write_phase_b_status_artifacts wrote it), fail soft;
                # the marker is a soft constraint relative to the persist
                # primitive's guarantees.
                pass

    # (18) Initial write with status="pending" so the wording gate can scan files.
    _persist_summary(RUN_STATUS_PENDING)

    # (19) Conservative-wording gate (spec §11.11). Fail hard on any
    # violation. Per phase_b_revision §11.11: explicit, fixed list of
    # user-facing report/summary artifact filenames; no glob, no legacy
    # filename heuristic. The optional drift artifacts are added only
    # when the diagnostic was actually computed (spec §15).
    wording_artifacts = [
        os.path.join(run_dir, name) for name in _WORDING_GATE_FILENAMES_BASE
    ]
    if drift_diagnostic_status == "computed":
        wording_artifacts += [
            os.path.join(run_dir, name)
            for name in _WORDING_GATE_FILENAMES_DRIFT
        ]
    wording_violations = check_artifacts_for_forbidden(wording_artifacts)
    if wording_violations:
        print("\nFATAL: Conservative-wording gate FAILED:")
        for v in wording_violations:
            print(f"  {v}")
        wording_reason = (
            "wording_violations: " + " | ".join(wording_violations[:5])
            + (" (+more)" if len(wording_violations) > 5 else "")
        )
        _persist_summary(RUN_STATUS_FAILED, failure_reason=wording_reason)
        raise RuntimeError(
            "Phase B FAILED: forbidden phrases found in summary artifacts. "
            f"Artifacts updated with run_status='failed'. See {run_dir}/RUN_STATUS.txt."
        )

    # (20) Eval-sanity checks (real, not vacuous).
    print(f"\n{'=' * 60}\nPHASE B SANITY CHECKS\n{'=' * 60}")
    checks: List[Tuple[str, bool]] = []

    # r2: schema consistency — populate sanity_parity_status field on the
    # summary even in full mode so downstream tooling can rely on its
    # presence. Full-mode runs use the existing anchor-comparison gates
    # (spec §11.4 / §11.5 / §11.6); the sanity-parity gate fires only in
    # sanity mode (set later in this block).
    if not sanity:
        summary.setdefault("sanity_parity_status", "n/a (full mode)")

    # no_patch present
    checks.append(("no_patch results exist", "no_patch" in restoration_results))

    # No NaN in any accuracy / delta cell.
    def _no_nan(rows: List[Dict]) -> bool:
        for r in rows:
            for k, v in r.items():
                if isinstance(v, float) and v != v:
                    return False
        return True

    checks.append(("No NaN in restoration table", _no_nan(restoration_table)))
    checks.append(("No NaN in corruption table", _no_nan(corruption_table)))
    checks.append(("clean_baseline_accuracy is finite",
                   clean_baseline_acc == clean_baseline_acc))
    checks.append((f"t_fixed == {t_fixed} in summary", summary["t_fixed"] == t_fixed))
    checks.append(("Conservative-wording gate clean", len(wording_violations) == 0))
    checks.append(("state_dict SHA-256 stable across run",
                   sd_sha_before == sd_sha_after))

    # phase_b_revision §10 ADDED checks:
    # 1. Anchor pin verification (full mode only — sanity is informational).
    if not sanity and pin_drift_msg:
        checks.append((f"Anchor pin verification: {pin_drift_msg}", False))
    elif not sanity:
        checks.append(("Anchor pin verification (canonical paths)", True))
    else:
        checks.append((
            "Anchor pin verification skipped (sanity mode)", True,
        ))

    # 2. Subset construction artifact (full mode requires success or
    #    explicit-skipped sanity-mode artifact per spec §10).
    subsets_path = os.path.join(run_dir, "phase_b_subsets.json")
    if sanity and conditional_status == "skipped":
        # Sanity-mode acceptable: file must exist (with explicit
        # status=skipped) per spec §7.2.4.
        checks.append((
            "Subset construction (sanity-skipped, artifact present)",
            os.path.isfile(subsets_path),
        ))
    elif conditional_status == "computed":
        checks.append((
            "Subset construction succeeded (n_total == len(samples))",
            (
                os.path.isfile(subsets_path)
                and int(subsets_summary.get("n_total", -1))
                == len(samples)
            ),
        ))
    else:
        # Skipped in non-sanity mode (e.g. parity failed): artifact must
        # exist with status=skipped, but this is also a hard fail because
        # P0 generation parity is broken or anchors are missing.
        checks.append((
            "Subset construction failed/skipped: "
            f"{conditional_skip_reason or 'unknown reason'}",
            False,
        ))

    # 3. Human:-continuation parity (P0). Per spec §11.5 / operator
    #    constraint #4, only the no-patch parity conditions are
    #    hard-gated; patched conditions are reported but NOT a pass/fail
    #    criterion. Skipped in sanity mode if anchors are unavailable.
    if conditional_status == "computed":
        cnp_block = per_condition_metrics.get("clean_no_patch", {})
        rnp_block = per_condition_metrics.get("restoration_no_patch", {})
        cnp_human = (
            cnp_block.get("output_behavior", {}).get(
                "human_continuation_count", 0
            )
        )
        rnp_human = (
            rnp_block.get("output_behavior", {}).get(
                "human_continuation_count", 0
            )
        )
        # Stage 1 anchor counts are recorded above; recompute here for
        # the sanity-check label so the message is self-documenting.
        s1_clean_human = _count_human_continuation(
            [
                {"output_text": ans} for ans in []
            ]  # placeholder; recomputed below
        )
        # Reuse the values computed during conditional metrics path.
        # ans_clean_global/correct_clean_global mirror the anchor rows;
        # we recover the original output_text via the per-sample row
        # store. Cheaper: just re-read from the resolved JSONLs (still
        # cheap; <1s for 250 rows).
        if no_swap_jsonl is not None and os.path.exists(no_swap_jsonl):
            with open(no_swap_jsonl, encoding="utf-8") as _f:
                _rows = [json.loads(L) for L in _f if L.strip()]
            s1_clean_human = _count_human_continuation(_rows)
        if hard_swap_jsonl is not None and os.path.exists(hard_swap_jsonl):
            with open(hard_swap_jsonl, encoding="utf-8") as _f:
                _rows = [json.loads(L) for L in _f if L.strip()]
            s1_corrupt_human = _count_human_continuation(_rows)
        else:
            s1_corrupt_human = 0
        clean_pass = cnp_human <= s1_clean_human + 2
        rest_pass = rnp_human <= s1_corrupt_human + 2
        checks.append((
            f"Human:-continuation parity clean_no_patch "
            f"({cnp_human} <= {s1_clean_human}+2)",
            clean_pass,
        ))
        checks.append((
            f"Human:-continuation parity restoration_no_patch "
            f"({rnp_human} <= {s1_corrupt_human}+2)",
            rest_pass,
        ))
    elif sanity:
        checks.append((
            f"Human:-continuation parity skipped (sanity, "
            f"reason: {conditional_skip_reason or 'no anchors'})",
            True,
        ))
    else:
        checks.append((
            "Human:-continuation parity check failed: conditional metrics "
            f"not computed ({conditional_skip_reason or 'unknown'})",
            False,
        ))

    # r2: Sanity-mode P0 parity check (operator-mandated). Even when the
    # conditional-metrics layer is skipped because Stage 1 anchors are
    # unavailable in sanity mode (spec §7.2.4), the no-patch parity gates
    # MUST fire — size-adjusted to n=5. This catches the regression mode
    # that masked itself in the r1 sanity smoke, where 5/5 Human:
    # continuations + ≈ 2.4× output-length ratio passed silently.
    sanity_parity_status: str = "n/a (full mode)"
    if sanity:
        clean_no_patch_rows_sanity = clean_baseline_results
        restoration_no_patch_rows_sanity = restoration_results.get("no_patch", [])
        (
            sanity_parity_status,
            sanity_parity_failures,
            sanity_parity_metrics,
        ) = _sanity_no_patch_parity_check(
            clean_no_patch_rows_sanity,
            restoration_no_patch_rows_sanity,
        )
        # Persist the gate result + metrics for downstream review. Embedded
        # in phase_b_summary.json as a top-level field (additive — does not
        # collide with any existing key).
        summary["sanity_parity_status"] = sanity_parity_status
        summary["sanity_parity_metrics"] = sanity_parity_metrics
        if sanity_parity_status == "failed":
            sanity_parity_reason = (
                "sanity P0 parity check failed: "
                + " | ".join(sanity_parity_failures)
            )
            # Set failure_reason on summary so the eventual _persist_summary
            # call (in the failed-checks branch below) preserves it.
            summary["failure_reason"] = sanity_parity_reason
            checks.append((
                f"Sanity-mode no-patch parity FAILED: "
                + "; ".join(sanity_parity_failures),
                False,
            ))
        else:
            cnp = sanity_parity_metrics.get("clean_no_patch", {})
            rnp = sanity_parity_metrics.get("restoration_no_patch", {})
            checks.append((
                f"Sanity-mode no-patch parity PASSED "
                f"(clean_no_patch human={int(cnp.get('human_continuation_count', 0))}, "
                f"len={cnp.get('avg_output_length_chars', 0.0):.1f}; "
                f"restoration_no_patch human={int(rnp.get('human_continuation_count', 0))}, "
                f"len={rnp.get('avg_output_length_chars', 0.0):.1f})",
                True,
            ))

    # Cross-phase accuracy check — spec §11.7 acceptance criterion.
    # RED LIGHT Fix A: full runs require BOTH anchors present and passing.
    if cross_check_passed is None:
        if sanity:
            checks.append((
                "Phase A cross-check skipped (sanity mode, no prior anchors required)",
                True,
            ))
        else:
            # Full run with missing anchors: hard fail.
            missing_str = ", ".join(cross_check_missing_anchors)
            checks.append((
                f"Cross-check FAILED: missing anchor(s) [{missing_str}]. "
                f"Full runs require BOTH hard_swap_b8 and no_swap anchors from "
                f"a parity-compatible prior run. Searched: "
                f"{_phase_a_outputs_dir()}, {_stage1_outputs_dir()}",
                False,
            ))
    elif cross_check_passed:
        checks.append((
            f"Cross-check PASSED: BOTH anchors within abs_delta <= {PHASE_A_CROSS_CHECK_TOL} "
            f"(hard_swap_b8 from {anchor_hard_swap_source}, "
            f"no_swap from {anchor_no_swap_source})",
            True,
        ))
    else:
        # Both present but at least one failed tolerance.
        failed_str = "; ".join(cross_check_failed_anchors)
        checks.append((
            f"Cross-check FAILED: {failed_str}",
            False,
        ))

    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")

    failed_labels = [label for label, ok in checks if not ok]
    if failed_labels:
        # YELLOW-LIGHT v3 P3: rewrite artifacts with explicit failure status
        # BEFORE raising, so a downstream user inspecting the run dir cannot
        # mistake a failed run for a valid one. Includes the actionable anchor
        # diagnostic when the failure was a cross-check miss.
        diag = render_anchor_gate_diagnostic(
            gate,
            phase_a_dir=_phase_a_outputs_dir(),
            stage1_dir=_stage1_outputs_dir(),
            workflow_doc=ANCHOR_WORKFLOW_DOC,
        )
        # r3: When the upstream anchor-accuracy parity pre-check fired,
        # prefer the operator-mandated reason format
        # "no_patch_anchor_accuracy_parity_failed: ..." so downstream review
        # sees the specific gate-ordering signal Codex flagged (not the
        # generic "sanity_check_failed: ..." prefix). This precedence rule
        # is checked first because the same-run sanity-parity gate is
        # mode-exclusive (sanity-only), while the anchor-accuracy gate is
        # primarily a full-mode signal. r4 watcher LOW #2: the prefix is
        # canonicalised to the underscore form to match the artifact-JSON
        # reason and the summary["failure_reason"] field.
        if (
            anchor_accuracy_parity_status == "failed"
            and isinstance(summary.get("failure_reason"), str)
            and summary["failure_reason"].startswith(
                "no_patch_anchor_accuracy_parity_failed: "
            )
        ):
            sanity_reason = summary["failure_reason"]
            if len(failed_labels) > 1:
                sanity_reason = (
                    f"{sanity_reason} | other failed gates: "
                    + " | ".join(
                        lab for lab in failed_labels
                        if not lab.startswith("Cross-check FAILED")
                    )[:300]
                )
        # r2: When the sanity-mode P0 parity gate fired, prefer the
        # operator-mandated reason format "sanity P0 parity check failed: ..."
        # so downstream review sees the specific signal that fired (not the
        # generic "sanity_check_failed: ..." prefix).
        elif (
            sanity
            and sanity_parity_status == "failed"
            and isinstance(summary.get("failure_reason"), str)
            and summary["failure_reason"].startswith("sanity P0 parity check failed: ")
        ):
            sanity_reason = summary["failure_reason"]
            if len(failed_labels) > 1:
                sanity_reason = (
                    f"{sanity_reason} | other failed gates: "
                    + " | ".join(
                        lab for lab in failed_labels
                        if not lab.startswith("Sanity-mode no-patch parity FAILED")
                    )[:300]
                )
        else:
            sanity_reason = (
                "sanity_check_failed: " + " | ".join(failed_labels[:3])
                + (f" (+{len(failed_labels) - 3} more)" if len(failed_labels) > 3 else "")
            )
        if diag:
            sanity_reason = f"{sanity_reason}. {diag}"
        _persist_summary(RUN_STATUS_FAILED, failure_reason=sanity_reason)
        raise RuntimeError(
            "Phase B FAILED sanity checks (see PASS/FAIL list above). "
            f"Artifacts updated with run_status='failed'. See {run_dir}/RUN_STATUS.txt."
        )

    # All gates passed — finalise artifacts with status="passed".
    _persist_summary(RUN_STATUS_PASSED)
    print(f"\nPhase B PASSED all sanity checks.")
    print(f"Outputs saved to: {run_dir}")

    # Cleanup
    del composed
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return run_dir


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase B: Restoration intervention. "
            "Full runs hard-fail unless BOTH the hard_swap_b8 anchor (from a "
            "parity-compatible Stage 1 run) AND the no_swap anchor (from a "
            "parity-compatible Phase A or Stage 1 run) are available within "
            f"tolerance {PHASE_A_CROSS_CHECK_TOL}. See {ANCHOR_WORKFLOW_DOC} "
            "for the parity contract and the recipe to generate compatible "
            "anchors."
        ),
        epilog=(
            "Anchor workflow: " + ANCHOR_WORKFLOW_DOC + "\n"
            "Common cause of 'missing anchor(s)' failure: Stage 1 was run with "
            "stage1_main.yaml (max_new_tokens=256) instead of stage2_confound.yaml "
            "(max_new_tokens=512), so the parity filter rejects it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to Phase B config YAML (required). Must match the Stage 1 / "
             "Phase A anchors' configs in models, dataset, generation, and "
             "hidden-state pooling - see notes/anchors_workflow.md.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global seed for all RNGs and CLI traceability (default: 42).",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Human-readable label for this run (logged in summary JSON).",
    )
    parser.add_argument(
        "--sanity", action="store_true",
        help="Sanity mode (development only): 5 samples x "
             "{no_patch, patch_recovery_full, clean_no_patch, "
             "corrupt_recovery_full}. Relaxes the anchor gate to best-effort. "
             "Never use for review or final results.",
    )
    args = parser.parse_args()
    run_phase_b(
        config_path=args.config,
        sanity=args.sanity,
        seed=args.seed,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()

# Note: the authoritative FORBIDDEN_PHRASES list lives in
# ``stage1/utils/wording.py`` and is invariant-tested in
# ``stage1/tests/test_phase_b_patcher.py::test_forbidden_phrases_gate``.
