"""MGSM dataset loader — downloads TSV directly from HuggingFace.

Uses urllib to download mgsm_{lang}.tsv rather than the HuggingFace
datasets library, which no longer supports the juletxara/mgsm loading
script. Downloaded files are cached in ~/.cache/mgsm/.
"""

import csv
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_HF_BASE = "https://huggingface.co/datasets/juletxara/mgsm/resolve/main"
_CACHE_DIR = Path.home() / ".cache" / "mgsm"

PROMPT_TEMPLATE = (
    "Solve the following math problem. "
    "Give the final answer as a single number after ####.\n\n"
    "Question: {question}\n"
    "Answer:"
)


def _download_tsv(lang: str) -> Path:
    """Return path to cached TSV for lang, downloading if necessary."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"mgsm_{lang}.tsv"
    if cache_path.exists():
        logger.info("Using cached TSV: %s", cache_path)
        return cache_path
    url = f"{_HF_BASE}/mgsm_{lang}.tsv"
    logger.info("Downloading %s → %s", url, cache_path)
    urllib.request.urlretrieve(url, cache_path)
    return cache_path


def load_mgsm(
    dataset_name: str = "mgsm",
    lang: str = "te",
    split: str = "test",
    debug_n: Optional[int] = None,
) -> List[Dict]:
    """
    Load MGSM dataset from HuggingFace TSV and apply English zero-shot prompt.

    Downloads mgsm_{lang}.tsv from HuggingFace if not already cached in
    ~/.cache/mgsm/. Uses the ``answer_number`` column as the gold answer
    (integer, already extracted — no regex parsing needed).

    Args:
        dataset_name: Identifier string only; not used for loading.
        lang: Language code (e.g. ``"te"`` for Telugu).
        split: Informational only; the TSV always contains the test set.
        debug_n: If set, return only the first n samples.

    Returns:
        List of dicts with keys: sample_id, prompt, gold_answer.
    """
    tsv_path = _download_tsv(lang)

    samples = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        headers = reader.fieldnames or []
        if "answer_number" not in headers:
            raise ValueError(
                f"Expected 'answer_number' column in {tsv_path}. "
                f"Found columns: {headers}"
            )
        for idx, row in enumerate(reader):
            question = row["question"].strip()
            gold_answer = str(int(row["answer_number"]))
            prompt = PROMPT_TEMPLATE.format(question=question)
            samples.append({
                "sample_id": f"{lang}_{split}_{idx}",
                "prompt": prompt,
                "gold_answer": gold_answer,
            })

    if debug_n is not None:
        samples = samples[:debug_n]

    logger.info("Loaded %d samples from MGSM-%s", len(samples), lang)
    return samples
