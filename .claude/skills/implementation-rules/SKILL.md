---
name: implementation-rules
description: Project-specific implementation rules for the writer. Must be loaded before the first Edit in any workflow. Covers seed wiring, config logging, no-silent-fallbacks, tokenizer language-code handling, dtype policy, logging, and path rules.
allowed-tools: Read
disable-model-invocation: false
---

# implementation-rules

These rules are mandatory for all code edits in this repo.

## Seeds and determinism

- Every training/eval entrypoint calls a single `set_all_seeds(seed)` helper
  that seeds `random`, `numpy`, `torch`, `torch.cuda`, and
  `transformers.set_seed`.
- `set_all_seeds` is imported from `src/utils/seed.py`. Do not reimplement.
- When config declares `deterministic: true`, set
  `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`,
  `cudnn.benchmark=False`, and ensure `CUBLAS_WORKSPACE_CONFIG` is set in env.

## Config and run identity

- Every training/eval entrypoint accepts `--config`, `--seed`, `--run-name`
  as required argparse arguments. Missing any → argparse error, non-zero exit.
- On run start, the script writes:
  - `runs/<name>/config.yaml` — resolved config snapshot
  - `runs/<name>/config_hash.txt` — sha256 of the resolved config
  - `runs/<name>/git_sha.txt` — `git rev-parse HEAD`
  - `runs/<name>/env.txt` — `pip freeze` output
- Refuse to run with a dirty git tree unless `--allow-dirty` is passed.

## No silent fallbacks

- Never `try/except` around config key access. Missing key → raise.
- Never default tokenizer/model names in code. Config is the only source.
- Never swallow device or dtype mismatches.

## Tokenizer and model pinning

- Every `AutoTokenizer.from_pretrained` / `AutoModel.from_pretrained` call
  passes `revision=` pulled from the config.
- Language codes: use ISO 639-3 throughout `src/`. Dataset loaders map from
  whatever the dataset uses to ISO 639-3 at load time.

## Dtype policy

- `fp32 | fp16 | bf16` declared in config. No implicit `torch.cuda.amp`
  without a config flag.

## Logging

- Library code under `src/` uses `logging.getLogger(__name__)`. No `print`.
- Scripts may use `print` for user-facing status only.

## Paths

- No absolute paths in configs. All paths are relative to repo root.
- `runs/`, `outputs/`, `data/raw/`, `paper/` are never written from `src/`
  library code — only from `scripts/` entrypoints.

## Tests

- Every new file under `src/` has at least one corresponding test under
  `tests/` mirroring the path.
- Tests are pure (no network, no GPU) unless marked `@pytest.mark.gpu` or
  `@pytest.mark.slow`.

## Output

Before the first Edit in any workflow, Claude lists which of the above rules
apply to the planned change. Rules not applied require an explicit N/A
justification.
