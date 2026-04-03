"""Regex-based numeric answer parser."""

import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)


def parse_answer(output_text: str) -> Dict:
    """
    Extract numeric answer from model output.

    Strategy:
    1. PRIMARY: match "The answer is X" (case-insensitive). Prompt instructs
       the model to end with exactly this phrase, so it should fire most often.
    2. SECONDARY fallback: last standalone integer in the text.
       Logged as parse_type="fallback" so we can track frequency.

    Returns:
        Dict with keys: parsed_answer, parse_success, normalized_answer, parse_type
    """
    text = output_text.strip()

    # PRIMARY: "The answer is X" where X starts with digits (may include commas/decimal)
    primary_match = re.search(
        r"(?i)the answer is\s*([\d,]+(?:\.\d+)?)",
        text,
    )
    if primary_match:
        raw = primary_match.group(1)
        normalized = _normalize_number(raw)
        logger.debug("parse_answer primary: raw=%r normalized=%r", raw, normalized)
        return {
            "parsed_answer": raw,
            "parse_success": True,
            "normalized_answer": normalized,
            "parse_type": "primary",
        }

    # SECONDARY fallback: last standalone integer in text
    fallback_matches = re.findall(r"\b(\d[\d,]*(?:\.\d+)?)\b", text)
    if fallback_matches:
        raw = fallback_matches[-1]
        normalized = _normalize_number(raw)
        logger.debug("parse_answer fallback: raw=%r normalized=%r", raw, normalized)
        return {
            "parsed_answer": raw,
            "parse_success": True,
            "normalized_answer": normalized,
            "parse_type": "fallback",
        }

    logger.debug("parse_answer failed: no number found in %r", text[:80])
    return {
        "parsed_answer": None,
        "parse_success": False,
        "normalized_answer": None,
        "parse_type": "failed",
    }


def _normalize_number(raw: str) -> str:
    """Strip whitespace, remove commas, normalize numeric string."""
    s = raw.strip().replace(",", "")
    if s.endswith("."):
        s = s[:-1]
    # Convert to int if possible (drops ".0")
    try:
        s = str(int(float(s))) if "." not in s else s
    except ValueError:
        pass
    return s
