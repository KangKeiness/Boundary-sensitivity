"""Regex-based numeric answer parser."""

import re
from typing import Dict


def parse_answer(output_text: str) -> Dict:
    """
    Extract numeric answer from model output.

    Tries the following strategies in order:
    1. Look for #### followed by a number
    2. Look for the last number in the output

    Returns:
        Dict with keys: parsed_answer, parse_success, normalized_answer
    """
    text = output_text.strip()

    # Strategy 1: look for #### delimiter
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        raw = match.group(1)
        normalized = _normalize_number(raw)
        return {
            "parsed_answer": raw,
            "parse_success": True,
            "normalized_answer": normalized,
        }

    # Strategy 2: find the last number in the text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        raw = numbers[-1]
        normalized = _normalize_number(raw)
        return {
            "parsed_answer": raw,
            "parse_success": True,
            "normalized_answer": normalized,
        }

    # No number found
    return {
        "parsed_answer": None,
        "parse_success": False,
        "normalized_answer": None,
    }


def _normalize_number(raw: str) -> str:
    """Strip whitespace, remove commas, normalize numeric string."""
    s = raw.strip()
    s = s.replace(",", "")
    # Remove trailing dot if no decimal digits follow
    if s.endswith("."):
        s = s[:-1]
    return s
