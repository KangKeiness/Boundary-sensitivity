---
name: claim-audit
description: Extracts every quantitative and qualitative claim from a paper draft or README and converts them to an evidence-checklist format consumed by claim-skeptic. Only invoked explicitly by claim-skeptic.
allowed-tools: Read, Grep, Glob
disable-model-invocation: true
---

# claim-audit

Deterministic claim extraction. Produces the input list for `result-validation`.

## Procedure

1. Read the target document(s) in full.
2. For each sentence, classify as:
   - `numeric_absolute` — e.g., "our model achieves 84.3 average accuracy"
   - `numeric_relative` — e.g., "improves by 1.8 points over XLM-R"
   - `superlative` — "state-of-the-art", "best", "outperforms"
   - `qualitative_strong` — "robust", "consistent", "generalizes", "matches"
   - `ablation` — "removing X reduces Y by Z"
   - `descriptive` — no evidence requirement
3. For each non-`descriptive` claim, emit a row with the required evidence
   fields.

## Hedging words that force extra evidence

Any claim containing any of these words MUST have a statistical test
artifact cited (`stat_test_ref`):

```
significantly, robustly, consistent(ly), generalizes, matches, outperforms,
state-of-the-art, SOTA, superior, substantially
```

Any claim that averages over languages MUST have a per-language breakdown
cited (`per_language_ref`).

## Output

```json
{
  "document": "paper/sections/4_results.tex",
  "claims": [
    {
      "id": 1,
      "text": "...",
      "location": "paper/...:NN",
      "type": "numeric_relative",
      "required_evidence": {
        "metric_artifact": "runs/*/metrics.json",
        "baseline_artifact": "runs/*/metrics.json",
        "stat_test_ref": "eval/results/sig_tests/*.json",
        "per_language_ref": null
      }
    }
  ]
}
```

## Forced checks

- Superlatives without a comparison table → flag even before validation.
- Numeric claims without a unit or precision → flag as ambiguous.
- "Our method" claims that don't name a config → flag.

## Example invocation

```
Run claim-audit on paper/sections/4_results.tex and emit a checklist for
claim-skeptic.
```
