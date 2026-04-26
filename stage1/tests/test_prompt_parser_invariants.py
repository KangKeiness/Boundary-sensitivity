"""Stage 1 parser/prompt invariants.

These checks lock core evaluation behavior:
  * Prompt suffix remains "Solution:".
  * Prompt explicitly asks for "The answer is X." formatting.
  * parse_answer keeps primary/fallback behavior.
"""

from stage1.data.loader import PROMPT_TEMPLATE
from stage1.inference.parser import parse_answer


def test_prompt_template_has_solution_suffix():
    assert PROMPT_TEMPLATE.endswith("Solution:")


def test_prompt_template_mentions_answer_format():
    assert "'The answer is X.'" in PROMPT_TEMPLATE


def test_parser_uses_last_primary_match_for_self_correction():
    out = "The answer is 8. Wait, correction: The answer is 18."
    parsed = parse_answer(out)
    assert parsed["parse_success"] is True
    assert parsed["parse_type"] == "primary"
    assert parsed["normalized_answer"] == "18"


def test_parser_supports_chinese_primary_pattern():
    out = "计算过程略。答案是 42"
    parsed = parse_answer(out)
    assert parsed["parse_success"] is True
    assert parsed["parse_type"] == "primary"
    assert parsed["normalized_answer"] == "42"


def test_parser_fallback_uses_last_number():
    out = "first guess 3 then final 7"
    parsed = parse_answer(out)
    assert parsed["parse_success"] is True
    assert parsed["parse_type"] == "fallback"
    assert parsed["normalized_answer"] == "7"


def test_parser_failure_when_no_number_present():
    out = "I cannot solve this."
    parsed = parse_answer(out)
    assert parsed["parse_success"] is False
    assert parsed["parse_type"] == "failed"
    assert parsed["normalized_answer"] is None
