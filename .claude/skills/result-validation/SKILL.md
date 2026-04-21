---
name: result-validation
description: Validates that a claimed metric value in a notebook, README, or paper draft traces to a concrete runs/<run-name>/metrics.json with matching seed, config hash, git SHA, and metric field. Only invoked explicitly by claim-skeptic and watcher.
allowed-tools: Read, Grep, Glob
disable-model-invocation: true
---

# result-validation

Deterministic procedure for verifying a single numeric claim.

## Inputs

- `claim_text` — verbatim quote from the source document
- `claim_location` — file:line
- `claimed_value` — the number (with units/precision)
- `claim_type` — `absolute | relative | superlative | ablation`

## Procedure

1. Parse `claimed_value` with full precision as written.
2. `Grep` `runs/` for `metrics.json` files.
3. For each candidate, `Read` and check if it contains a field whose value
   equals `claimed_value` at the written precision.
4. For every match, verify the containing run directory has ALL of:
   - `config_hash.txt`
   - `git_sha.txt` (SHA must exist in `git log`)
   - `seed` field in `metrics.json` or `config.yaml`
   - `config.yaml` snapshot
5. If `claim_type == relative`, locate the baseline run by matching
   non-varying config keys and the same seed count; verify the baseline's
   corresponding metric field.
6. If `claim_type == superlative`, require an explicit comparison table file
   under `eval/results/` where every row links to an artifact.
7. If `claim_type == ablation`, verify the ablated `configs/` file exists and
   differs from the base config only in declared ablation keys.

## Output

```json
{
  "claim_text": "...",
  "claim_location": "paper/...:NN",
  "claimed_value": 0.0,
  "matches": [
    {
      "run": "runs/xnli_contrast_s1/",
      "metric_key": "avg_acc",
      "value": 0.0,
      "seed": 1,
      "config_hash": "...",
      "git_sha": "...",
      "baseline": "runs/xnli_base_s1/",
      "baseline_value": 0.0
    }
  ],
  "verdict": "supported|unsupported|overclaim|needs_matched_baseline|rounding_error",
  "note": "..."
}
```

## Forced checks

- Artifact existence. Missing file → `unsupported`.
- Git SHA resolves in `git log`. Unresolved → `unsupported`.
- Precision: rounding that changes a comparison ranking → `rounding_error`.
- Reduction axis: per-language vs aggregate distinction preserved.

## Example invocation

```
Run result-validation on claim "XLM-R+contrastive improves avg XNLI accuracy
by 1.8" at paper/sections/results.tex:42.
```
