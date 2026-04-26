"""MGSM data loader — TSV-based with pinned revision + SHA-256 verification.

Stage 1 hardening (2026-04-25):
    * Stop relying on HuggingFace's mutable ``main`` ref. The TSV is fetched
      from ``resolve/{revision}/...`` where ``revision`` is a config field.
    * Compute SHA-256 of the downloaded bytes and verify against
      ``expected_sha256`` from config. Mismatch = abort the run; do NOT
      silently continue against drifted data.
    * Record the exact revision string and the realised SHA-256 on
      ``config.dataset._provenance`` so callers can embed them in manifests.

Provenance is the trust anchor — even if a non-pinned ``revision`` is supplied
(e.g. for a temporary investigation), the realised SHA-256 still ends up in
the manifest, so any later run can reproduce the byte-identical dataset.
"""

import csv
import hashlib
import json
import logging
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Show your reasoning first, then write your final answer on the last line in this exact format: "
    "'The answer is X.' where X is the numeric answer.\n\n"
    "Problem: {question}\n"
    "Solution:"
)

# Source repo on HuggingFace. Revision is supplied per-call via config.
TSV_URL_TEMPLATE = (
    "https://huggingface.co/datasets/juletxara/mgsm"
    "/resolve/{revision}/mgsm_{lang}.tsv"
)

# Known-good SHA-256 digests for the dataset files we depend on. These are the
# byte-exact pins; the loader hard-fails if config.dataset.expected_sha256 is
# not set AND the lang is not in this map.
KNOWN_DATASET_SHA256: Dict[str, str] = {
    # mgsm Chinese (zh) test split, 250 rows. Recorded in
    # notes/data_changelog.md (mgsm_zh@v1.0-2022-10-03).
    "zh": "b2fa63151022370a0de1f4211c8c284eae74b0f5a3b003b1d5982c0d4a73f661",
}

_DEFAULT_CACHE_ROOT = Path(tempfile.gettempdir()) / "mgsm_cache"
CACHE_DIR = Path(os.getenv("MGSM_CACHE_DIR", str(_DEFAULT_CACHE_ROOT)))


class DatasetProvenanceError(RuntimeError):
    """Raised when dataset bytes do not match the expected SHA-256 pin."""


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_tsv(lang: str, revision: str) -> Path:
    """Download (or reuse cache) the MGSM TSV for ``lang`` at ``revision``.

    The cache key includes the revision so swapping revisions does NOT silently
    reuse the wrong bytes.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Sanitise revision for use as a path segment.
    rev_safe = revision.replace("/", "_")
    cache_path = CACHE_DIR / f"mgsm_{lang}__{rev_safe}.tsv"
    if not cache_path.exists():
        url = TSV_URL_TEMPLATE.format(revision=revision, lang=lang)
        logger.info(f"Downloading MGSM {lang}@{revision} from {url}")
        urllib.request.urlretrieve(url, cache_path)
        logger.info(f"Saved to {cache_path}")
    else:
        logger.info(f"Using cached TSV: {cache_path}")
    return cache_path


def _resolve_expected_sha256(config) -> Optional[str]:
    """Return the SHA-256 the caller has pinned, falling back to KNOWN_DATASET_SHA256."""
    explicit = getattr(config.dataset, "expected_sha256", None)
    if explicit:
        return explicit
    return KNOWN_DATASET_SHA256.get(config.dataset.lang)


def _verify_sha256(path: Path, expected: Optional[str], lang: str) -> str:
    """Compute and (if expected is given) verify SHA-256 of ``path``.

    Returns the realised digest. Raises DatasetProvenanceError on mismatch.
    Logs a loud warning when no expected hash is available.
    """
    realised = _sha256_of_file(path)
    if expected is None:
        logger.warning(
            "No expected_sha256 pin available for lang=%s; skipping integrity check. "
            "Realised SHA-256: %s. Set dataset.expected_sha256 in the YAML to lock this.",
            lang, realised,
        )
        return realised
    if realised != expected:
        raise DatasetProvenanceError(
            f"Dataset SHA-256 mismatch for lang={lang}: "
            f"expected {expected}, got {realised}. "
            f"The upstream dataset has changed (or the cache is corrupt). "
            f"Refusing to run against unverified bytes. Path: {path}"
        )
    logger.info("Dataset SHA-256 OK for lang=%s: %s", lang, realised)
    return realised


def _parse_tsv(path: Path) -> List[Dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if len(row) < 2:
                continue
            question = row[0].strip()
            # TSV format: question \t answer_number
            try:
                gold = int(row[1].strip())
            except ValueError:
                # fallback: try last numeric token
                tokens = row[1].strip().split()
                gold = None
                for t in reversed(tokens):
                    try:
                        gold = int(t.replace(",", "").rstrip("."))
                        break
                    except ValueError:
                        continue
            if question and gold is not None:
                samples.append({
                    "sample_id": f"mgsm_{i:04d}",
                    "question": question,
                    "gold_answer": str(gold),
                })
    if len(samples) == 0:
        raise ValueError(f"No valid samples parsed from {path}. Check TSV column format.")
    logger.info(f"Parsed {len(samples)} samples from {path} (first gold: {samples[0]['gold_answer']})")
    return samples


def build_dataset_provenance(
    config,
    *,
    realised_sha256: str,
    revision: str,
    cache_path: Path,
    raw_row_count: int,
) -> Dict[str, Any]:
    """Build the canonical dataset-provenance block for embedding in manifests.

    Schema is stable — downstream code (manifest_parity, post-analysis tools)
    depends on the exact key set.
    """
    return {
        "name": config.dataset.name,
        "lang": config.dataset.lang,
        "split": config.dataset.split,
        "revision": revision,
        "source": TSV_URL_TEMPLATE.format(revision=revision, lang=config.dataset.lang),
        "sha256": realised_sha256,
        "expected_sha256_pin": getattr(config.dataset, "expected_sha256", None)
            or KNOWN_DATASET_SHA256.get(config.dataset.lang),
        "row_count_raw": raw_row_count,
        "cache_path": str(cache_path),
    }


def load_mgsm(config) -> List[Dict]:
    """Load MGSM samples and apply prompt template.

    Side effect: stamps ``config.dataset._provenance`` with the realised
    revision, SHA-256, source URL, and row count so callers can embed it in
    their manifests without re-doing the work.

    Args:
        config: Stage1Config with dataset.{lang, debug_n, revision, expected_sha256}.

    Returns:
        List of {sample_id, prompt, gold_answer}.

    Raises:
        DatasetProvenanceError: SHA-256 mismatch against the pin.
    """
    lang = config.dataset.lang
    debug_n: Optional[int] = config.dataset.debug_n
    revision: str = getattr(config.dataset, "revision", None) or "main"

    if revision == "main":
        logger.warning(
            "dataset.revision is 'main' (mutable). "
            "Set a pinned commit SHA in the YAML for reproducibility. "
            "SHA-256 verification still applies if expected_sha256 is set."
        )

    tsv_path = _download_tsv(lang, revision)
    expected = _resolve_expected_sha256(config)
    realised_sha = _verify_sha256(tsv_path, expected, lang)
    raw = _parse_tsv(tsv_path)

    # Stamp provenance onto the config so logger.save_manifest / Phase A/B/C
    # manifest builders can pick it up without redoing the download.
    provenance = build_dataset_provenance(
        config,
        realised_sha256=realised_sha,
        revision=revision,
        cache_path=tsv_path,
        raw_row_count=len(raw),
    )
    try:
        # config is a dataclass instance — attach as a normal attribute.
        config.dataset._provenance = provenance  # type: ignore[attr-defined]
    except Exception:
        pass

    # Best-effort sidecar so a bare cache directory is still self-describing.
    try:
        sidecar = tsv_path.with_suffix(tsv_path.suffix + ".sha256.json")
        with open(sidecar, "w", encoding="utf-8") as sf:
            json.dump(provenance, sf, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.debug("Could not write provenance sidecar: %r", exc)

    if debug_n is not None:
        raw = raw[:debug_n]
        logger.info(f"Debug mode: using first {debug_n} samples")

    samples = []
    for item in raw:
        samples.append({
            "sample_id": item["sample_id"],
            "prompt": PROMPT_TEMPLATE.format(question=item["question"]),
            "gold_answer": item["gold_answer"],
        })

    logger.info(
        "Loaded %d MGSM samples (lang=%s, revision=%s, sha256=%s)",
        len(samples), lang, revision, realised_sha,
    )
    if debug_n is None and len(samples) != 250:
        logger.warning(f"Expected 250 MGSM samples but got {len(samples)}")
    return samples
