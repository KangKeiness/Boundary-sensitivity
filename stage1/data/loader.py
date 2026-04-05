"""MGSM data loader — TSV-based (HF datasets script no longer supported)."""

import csv
import logging
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Show your reasoning first, then write your final answer on the last line in this exact format: "
    "'The answer is X.' where X is the numeric answer.\n\n"
    "Problem: {question}\n"
    "Solution:"
)

TSV_URL = (
    "https://huggingface.co/datasets/juletxara/mgsm"
    "/resolve/main/mgsm_{lang}.tsv"
)

CACHE_DIR = Path("/tmp/mgsm_cache")


def _download_tsv(lang: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"mgsm_{lang}.tsv"
    if not cache_path.exists():
        url = TSV_URL.format(lang=lang)
        logger.info(f"Downloading MGSM {lang} from {url}")
        urllib.request.urlretrieve(url, cache_path)
        logger.info(f"Saved to {cache_path}")
    else:
        logger.info(f"Using cached TSV: {cache_path}")
    return cache_path


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


def load_mgsm(config) -> List[Dict]:
    """
    Load MGSM samples and apply prompt template.

    Args:
        config: Stage1Config with dataset.lang, dataset.debug_n fields.

    Returns:
        List of {sample_id, prompt, gold_answer}
    """
    lang = config.dataset.lang
    debug_n: Optional[int] = config.dataset.debug_n

    tsv_path = _download_tsv(lang)
    raw = _parse_tsv(tsv_path)

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

    logger.info(f"Loaded {len(samples)} MGSM samples (lang={lang})")
    if debug_n is None and len(samples) != 250:
        logger.warning(f"Expected 250 MGSM samples but got {len(samples)}")
    return samples
