---
name: papergrade-experiment
description: Enforces paper-grade reproducibility fields in any experiment spec or training script change. Trigger on new experiment, new training entrypoint, or changes to configs/train/*.yaml.
allowed-tools: Read, Grep, Glob
disable-model-invocation: false
---

# papergrade-experiment

Before finalizing any spec or training-related edit, Claude MUST verify each
of the following is present and concrete. Any missing item → halt and request.

## Required fields

1. **Seed policy** — single seed vs multi-seed list. Single seed is allowed
   ONLY for smoke tests; all cited results require ≥3 seeds.
2. **Config hash** — training script must compute and log a hash of the
   resolved config at run start, written to `runs/<name>/config_hash.txt`.
3. **Git SHA capture** — training script must log current `git rev-parse HEAD`
   to `runs/<name>/git_sha.txt`; refuse to run with a dirty tree unless
   `--allow-dirty` is passed.
4. **Determinism flags** — `torch.use_deterministic_algorithms(True)`,
   `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.backends.cudnn.deterministic=True`,
   `torch.backends.cudnn.benchmark=False`. Config must state whether
   determinism is required.
5. **Precision / dtype** — fp32, fp16, or bf16 declared explicitly. No silent
   autocast without a config flag.
6. **Dataset version + hash** — from `notes/data_changelog.md`. Unversioned
   datasets are forbidden.
7. **Tokenizer revision pinning** — every `from_pretrained` call names a
   concrete `revision=` for tokenizer AND model.
8. **Per-language eval breakdown** — required for any multilingual metric.
   Averaging without breakdown is forbidden.
9. **Statistical test plan** — name (paired bootstrap, permutation, etc.),
   alpha, seed list. Required for any comparative claim.
10. **Compute budget** — GPU-hours, wall clock, storage; must be stated.
11. **Acceptance thresholds** — numeric pass/fail criteria tied to artifact
    paths. No vague success statements.

## Output

Return a checklist with `[x]` / `[ ]` per item. If any `[ ]`, Claude must
block the containing workflow and request the missing information.

## Example invocation

```
Run papergrade-experiment against notes/specs/xnli_contrastive.md and report
missing fields.
```
