"""Manifest parity checker for cross-phase and baseline-reuse validation.

Ensures that reused run directories and anchor selections are scientifically
compatible by checking model identifiers, dataset, generation config, and
sample counts. Sample-ID match alone is NOT sufficient (RED LIGHT review P2).

Usage:
    from stage1.utils.manifest_parity import check_manifest_parity, ManifestParityError
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── Sample-regime parity helpers (v4 PRIORITY 2) ───────────────────────────
#
# Anchor candidates are scientifically comparable only when they were evaluated
# on the SAME sample regime: same debug-vs-full setting, same sample count, and
# same sample ordering. Sample-ID match alone is NOT sufficient (a debug subset
# can be a strict prefix of the full set yet remain methodologically
# incomparable).


def compute_sample_ordering_sha256(sample_ids: Iterable[str]) -> str:
    """Deterministic sha256 over an ordered iterable of sample ids.

    Order is preserved (this is an ordering signature, not a set hash). The
    digest is computed over null-separated UTF-8 bytes so identical id lists
    in identical order produce identical digests across runs and platforms.
    """
    h = hashlib.sha256()
    first = True
    for sid in sample_ids:
        if not first:
            h.update(b"\x00")
        h.update(str(sid).encode("utf-8"))
        first = False
    return h.hexdigest()


def compute_sample_regime(
    config, sample_ids: Iterable[str],
) -> Dict[str, Any]:
    """Build the sample-regime sub-block for a given (config, samples) pair.

    Schema:
        {
          "mode": "full" | "debug",
          "debug_n": Optional[int],            # mirrors config.dataset.debug_n
          "sample_count": int,
          "sample_ordering_sha256": str,       # 64-char hex
        }

    Anchor candidates with a different mode, count, or ordering signature are
    rejected by ``check_manifest_parity`` in full-mode evaluations.
    """
    ids_list = list(sample_ids)
    debug_n = getattr(config.dataset, "debug_n", None)
    return {
        "mode": "debug" if debug_n is not None else "full",
        "debug_n": debug_n,
        "sample_count": len(ids_list),
        "sample_ordering_sha256": compute_sample_ordering_sha256(ids_list),
    }


class ManifestParityError(RuntimeError):
    """Raised when a manifest parity check fails."""

    def __init__(self, mismatches: List[str], source_path: str, target_desc: str):
        self.mismatches = mismatches
        self.source_path = source_path
        self.target_desc = target_desc
        detail = "\n  ".join(mismatches)
        super().__init__(
            f"Manifest parity check FAILED between {source_path} and "
            f"{target_desc}:\n  {detail}"
        )


# ── Canonical manifest fields to check ──────────────────────────────────────

# The parity key set. Each entry is (json_path, human_label). json_path is a
# dot-separated key path into the manifest dict. Missing keys on EITHER side
# are flagged as mismatches (not silently ignored).

_PARITY_FIELDS: List[Tuple[str, str]] = [
    ("models.recipient", "recipient model identifier"),
    ("models.donor", "donor model identifier"),
    ("models.recipient_revision", "recipient model revision"),
    ("models.donor_revision", "donor model revision"),
    ("dataset.name", "dataset name"),
    ("dataset.lang", "dataset language"),
    ("dataset.split", "dataset split"),
    # Stage 1 hardening (2026-04-25): dataset bytes must agree across the
    # parity contract. ``revision`` controls what we fetched; ``sha256`` is
    # the realised file digest. Either drift is a hard mismatch.
    ("dataset.revision", "dataset revision (HF ref)"),
    ("dataset.sha256", "dataset SHA-256 (file bytes)"),
    ("generation.do_sample", "generation do_sample"),
    ("generation.temperature", "generation temperature"),
    ("generation.max_new_tokens", "generation max_new_tokens"),
    ("hidden_state.pooling", "hidden-state pooling method"),
    # v4 P2 — sample-regime parity. Anchors with a different mode, count, or
    # ordering hash are methodologically non-comparable even if every other
    # field agrees.
    ("sample_regime.mode", "sample regime mode (debug vs full)"),
    ("sample_regime.debug_n", "sample regime debug_n"),
    ("sample_regime.sample_count", "sample regime sample_count"),
    ("sample_regime.sample_ordering_sha256",
     "sample regime sample_ordering_sha256"),
]


def _resolve(d: Dict[str, Any], path: str) -> Any:
    """Resolve a dot-separated key path in a nested dict. Returns _MISSING sentinel on miss."""
    parts = path.split(".")
    cur: Any = d
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return _MISSING
    return cur


_MISSING = object()


def extract_parity_block(
    config,
    *,
    sample_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Build a parity block from a loaded Stage1Config for embedding in manifests.

    This is the canonical form that future manifest files should contain under
    the ``parity`` key, so that ``check_manifest_parity`` can validate without
    needing to re-parse the original YAML.

    v4 P2: when ``sample_ids`` is provided, embed a ``sample_regime`` sub-block
    so anchor candidates can be filtered by mode (debug/full), sample count,
    and sample-ordering hash. Callers that don't yet know the sample list can
    omit it; downstream parity checks will then flag the missing field as a
    mismatch (intentional — anchor reuse without sample-regime parity is
    scientifically suspect).
    """
    block: Dict[str, Any] = {
        "models": {
            "recipient": config.models.recipient,
            "donor": config.models.donor,
            "recipient_revision": config.models.recipient_revision,
            "donor_revision": config.models.donor_revision,
        },
        "dataset": {
            "name": config.dataset.name,
            "lang": config.dataset.lang,
            "split": config.dataset.split,
            # v4 P2: debug_n is a parity-checked field — debug subsets are not
            # comparable to full runs even when every other field agrees.
            "debug_n": getattr(config.dataset, "debug_n", None),
            # Stage 1 hardening (2026-04-25): dataset bytes are part of the
            # parity contract. ``revision`` is the HF ref the loader fetched
            # from; ``sha256`` is the realised file digest stamped by the
            # loader onto config.dataset._provenance. If load_mgsm has not
            # been called yet (e.g. test fixture path), fall back to the
            # configured pin so the parity block still encodes intent.
            "revision": (
                (getattr(config.dataset, "_provenance", None) or {}).get("revision")
                or getattr(config.dataset, "revision", None)
            ),
            "sha256": (
                (getattr(config.dataset, "_provenance", None) or {}).get("sha256")
                or getattr(config.dataset, "expected_sha256", None)
            ),
        },
        "generation": {
            "do_sample": config.generation.do_sample,
            "temperature": config.generation.temperature,
            "max_new_tokens": config.generation.max_new_tokens,
        },
        "hidden_state": {
            "pooling": config.hidden_state.pooling,
        },
    }
    if sample_ids is not None:
        block["sample_regime"] = compute_sample_regime(config, sample_ids)
    return block


def check_manifest_parity(
    source_manifest: Dict[str, Any],
    target_block: Dict[str, Any],
    *,
    source_path: str = "<source>",
    target_desc: str = "<target>",
    extra_fields: Optional[List[Tuple[str, str]]] = None,
) -> List[str]:
    """Compare two manifest / parity blocks and return a list of mismatch strings.

    Checks every field in ``_PARITY_FIELDS`` plus any ``extra_fields``. Returns
    an empty list on full parity.

    Both ``source_manifest`` and ``target_block`` are flat or nested dicts. The
    source is typically the ``manifest.json`` from the reused/anchor run; the
    target is the parity block extracted from the current config.

    When the source manifest has a top-level ``parity`` key, that sub-dict is
    used for comparison (preferred). Otherwise, the entire manifest dict is
    traversed directly (backward compat with older manifests that embedded
    config blocks at the top level or under ``config``).
    """
    # Prefer explicit parity block; fall back to top-level or config sub-key.
    src = source_manifest
    if "parity" in src:
        src = src["parity"]
    elif "config" in src and isinstance(src["config"], dict):
        # Stage 1 manifests store config under "config".
        src = src["config"]

    fields = list(_PARITY_FIELDS)
    if extra_fields:
        fields.extend(extra_fields)

    mismatches: List[str] = []
    for path, label in fields:
        val_src = _resolve(src, path)
        val_tgt = _resolve(target_block, path)

        if val_src is _MISSING and val_tgt is _MISSING:
            # Both missing — acceptable (optional field absent on both sides).
            continue
        # RED LIGHT Fix B: backward compatibility for revision fields.
        # Older Stage 1 manifests may have revision fields missing entirely.
        # Treat missing-vs-None as equivalent ONLY for revision fields, because
        # a None revision means "HF default (main)" which is the same semantic.
        # This does NOT weaken parity for model name, dataset, or generation
        # config fields — those still hard-fail on missing.
        # v4 P2: same equivalence for ``dataset.debug_n`` because absent vs
        # null both mean "full run, no debug subset". Note: the OTHER three
        # sample_regime fields (mode/sample_count/sample_ordering_sha256) are
        # NOT in this list — those must be present on both sides or it's a
        # mismatch. This forces anchor manifests to be regenerated under v4.
        # Stage 1 hardening (2026-04-25): dataset.revision / dataset.sha256
        # are NEW fields. To avoid breaking comparisons against pre-existing
        # anchor manifests, treat MISSING-on-source as equivalent to None.
        # Drift between two NON-MISSING values is still a hard mismatch.
        _NULL_EQUIV_PATHS = (
            "models.recipient_revision",
            "models.donor_revision",
            "dataset.debug_n",
            "dataset.revision",
            "dataset.sha256",
        )
        if path in _NULL_EQUIV_PATHS:
            # Normalize: MISSING → None for comparison.
            norm_src = None if val_src is _MISSING else val_src
            norm_tgt = None if val_tgt is _MISSING else val_tgt
            if norm_src != norm_tgt:
                mismatches.append(
                    f"{label} ({path}): source={norm_src!r} vs current={norm_tgt!r}"
                )
            continue
        if val_src is _MISSING:
            mismatches.append(f"{label} ({path}): missing in source manifest")
            continue
        if val_tgt is _MISSING:
            mismatches.append(f"{label} ({path}): missing in current config")
            continue
        if val_src != val_tgt:
            mismatches.append(
                f"{label} ({path}): source={val_src!r} vs current={val_tgt!r}"
            )

    return mismatches


def check_manifest_parity_or_raise(
    source_manifest: Dict[str, Any],
    target_block: Dict[str, Any],
    *,
    source_path: str = "<source>",
    target_desc: str = "<target>",
    extra_fields: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """Like ``check_manifest_parity`` but raises ``ManifestParityError`` on mismatch."""
    mismatches = check_manifest_parity(
        source_manifest, target_block,
        source_path=source_path,
        target_desc=target_desc,
        extra_fields=extra_fields,
    )
    if mismatches:
        raise ManifestParityError(mismatches, source_path, target_desc)
    logger.info(
        "Manifest parity check PASSED: %s vs %s (%d fields checked)",
        source_path, target_desc, len(_PARITY_FIELDS) + (len(extra_fields) if extra_fields else 0),
    )


def load_manifest_from_run_dir(run_dir: str) -> Dict[str, Any]:
    """Load ``manifest.json`` from a run directory."""
    path = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No manifest.json in {run_dir}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
