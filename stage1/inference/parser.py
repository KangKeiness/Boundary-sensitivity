"""Regex-based numeric answer parser."""

import logging
import re
import math
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def parse_answer(output_text: str) -> Dict:
    """
    Extract numeric answer from model output.

    Strategy:
    1. PRIMARY: match any of the multilingual answer phrases. Collects ALL
       matches across all patterns, then takes the LAST one by position.
       This handles self-correction (e.g. "answer is 8 ... answer is 18").
    2. SECONDARY fallback: last standalone number in the text.
       Logged as parse_type="fallback" so we can track frequency.

    Returns:
        Dict with keys: parsed_answer, parse_success, normalized_answer, parse_type
    """
    text = output_text.strip()

    # PRIMARY patterns — order doesn't matter; we take last match by position
    primary_patterns = [
        r"(?i)the answer is\s*([\d,]+(?:\.\d+)?)",       # English
        r"(?i)answer\s*[:=]\s*([\d,]+(?:\.\d+)?)",        # English alt
        r"(?:答案是|答案为)\s*([\d,]+(?:\.\d+)?)",          # Chinese
    ]

    all_matches = []  # list of (start_pos, captured_group)
    for pattern in primary_patterns:
        for m in re.finditer(pattern, text):
            all_matches.append((m.start(), m.group(1)))

    if all_matches:
        # Take the match with the largest start position (last in text)
        _, raw = max(all_matches, key=lambda x: x[0])
        normalized = _normalize_number(raw)
        n_total_matches = len(all_matches)
        logger.debug(
            f"parse_answer primary (last of {n_total_matches}): raw={raw!r} normalized={normalized!r}"
        )
        return {
            "parsed_answer": raw,
            "parse_success": True,
            "normalized_answer": normalized,
            "parse_type": "primary",
        }

    # SECONDARY fallback: last standalone number in text
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


def _normalize_number(raw: str) -> Optional[str]:
    """Strip whitespace, remove commas, normalize numeric string."""
    s = raw.strip().replace(",", "")
    if s.endswith("."):
        s = s[:-1]
    # Convert to int if possible (drops ".0")
    try:
        if "." not in s:
            try:
                value = float(s)
                if not math.isfinite(value):
                    return None
                s = str(int(value))
            except (ValueError, OverflowError):
                return None
    except ValueError:
        pass
    return s
