"""MGSM dataset loader with English zero-shot prompt template."""

import re
from typing import List, Dict, Optional
from datasets import load_dataset

PROMPT_TEMPLATE = (
    "Solve the following math problem. "
    "Give the final answer as a single number after ####.\n\n"
    "Question: {question}\n"
    "Answer:"
)


def extract_gold_answer(answer_text: str) -> str:
    """Extract the numeric answer after #### from the gold answer string."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        raw = match.group(1).strip()
        return re.sub(r"[,\s]", "", raw)
    # Fallback: try to find the last number in the string
    numbers = re.findall(r"-?[\d,]+\.?\d*", answer_text)
    if numbers:
        return re.sub(r",", "", numbers[-1])
    return answer_text.strip()


def load_mgsm(
    dataset_name: str = "juletxara/mgsm",
    lang: str = "te",
    split: str = "test",
    debug_n: Optional[int] = None,
) -> List[Dict]:
    """
    Load MGSM dataset and apply English zero-shot prompt template.

    Returns:
        List of dicts with keys: sample_id, prompt, gold_answer
    """
    ds = load_dataset(dataset_name, lang, split=split)

    samples = []
    for idx, row in enumerate(ds):
        question = row["question"]
        answer_text = row["answer"]
        gold = extract_gold_answer(answer_text)
        prompt = PROMPT_TEMPLATE.format(question=question)
        samples.append({
            "sample_id": f"{lang}_{split}_{idx}",
            "prompt": prompt,
            "gold_answer": gold,
        })

    if debug_n is not None:
        samples = samples[:debug_n]

    return samples
