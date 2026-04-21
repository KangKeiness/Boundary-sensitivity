---
name: dataset-audit
description: Standard multilingual dataset audit — hashes, per-language counts, script distribution, label balance, train/test leakage, tokenizer OOV, truncation risk, empty rows. Only invoked explicitly by data-auditor.
allowed-tools: Read, Grep, Glob, Bash
disable-model-invocation: true
---

# dataset-audit

Deterministic audit procedure. Produces the JSON schema that the
`training_guard.sh` hook reads.

## Mandatory checks

1. **Hashes** — sha256 of every file referenced in the target dataset
   config. Compare to `notes/data_changelog.md`. Drift without a matching
   changelog entry → `DIRTY` + `block_training: true`.
2. **Row counts** — per split, per language.
3. **Token counts** — using the pinned tokenizer revision from the config.
4. **Script/Unicode block** — for each language field, distribution of
   Unicode scripts. Mismatch (e.g., >1% Cyrillic in an `en` field) → finding.
5. **Label distribution** — per split; compare to last lockfile; drift >5% on
   any class → finding.
6. **Split disjointness** — row hash AND id-column disjointness across
   train/dev/test.
7. **Leakage** — substring overlap AND MinHash near-duplicate (Jaccard >0.8)
   between train and each eval set. Ratio reported; >0.1% → `block_training`.
8. **OOV rate** — per language, using pinned tokenizer revision.
9. **Length distribution** — percentage of examples whose token count
   exceeds the configured `max_len` → truncation risk.
10. **Empty/null/whitespace-only** rows.
11. **Cache keys** — must include `dataset_version`, `tokenizer_name`,
    `tokenizer_revision`, `max_len`, preprocessing script git sha. Missing
    any → finding.
12. **Cache freshness** — cache mtime vs source-of-truth checksum.

## Output schema (strict)

```json
{
  "slug": "...",
  "generated_at": "<UTC ISO8601>",
  "verdict": "CLEAN|WARN|DIRTY",
  "block_training": false,
  "changelog_synced": true,
  "datasets": [
    {
      "name": "...",
      "version": "...",
      "files": [{"path": "...", "sha256": "..."}],
      "row_counts": {"train": 0, "dev": 0, "test": 0},
      "per_language": {"en": 0},
      "token_counts": {"train": 0, "dev": 0, "test": 0},
      "script_distribution": {"en": {"Latin": 1.0}},
      "label_distribution": {"train": {"0": 0.5, "1": 0.5}},
      "split_disjointness": {"train_dev": true, "train_test": true, "dev_test": true},
      "leakage": {"train_test_substring": 0.0, "train_test_minhash": 0.0},
      "oov_rate": {"en": 0.0},
      "truncation_risk": {"pct_over_max_len": 0.0},
      "cache": {"key_valid": true, "stale": false},
      "findings": [{"severity": "block|high|med|low", "issue": "...", "evidence": "..."}]
    }
  ]
}
```

## Forced `block_training: true` conditions

- Any hash drift without changelog entry
- Any leakage ratio > 0.1%
- Any split disjointness failure
- Any cache key missing a required component
- Any dataset with `findings[].severity == "block"`

## Example invocation

```
Run dataset-audit on configs/data/xnli_15way.yaml for the upcoming run
xnli_contrast_seed1.
```
