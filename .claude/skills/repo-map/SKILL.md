---
name: repo-map
description: Produces a current-state structural map of the ML/NLP repo (configs, scripts, src, data, runs, outputs, eval, notes) and its conventional entrypoints. Used to ground planners and reviewers in real repo state before making architecture claims.
allowed-tools: Read, Grep, Glob, Bash
disable-model-invocation: false
---

# repo-map

Produce a current-state summary of the repository. Do NOT cache; re-read on
every invocation.

## Procedure

1. `ls` the top level; confirm presence of expected dirs: `configs/`,
   `scripts/`, `src/`, `data/`, `runs/`, `outputs/`, `eval/`, `notes/`,
   `tests/`, `paper/`.
2. `Glob` each dir for primary file types:
   - `configs/**/*.yaml`
   - `scripts/train_*.py`, `scripts/eval_*.py`, `scripts/data/*.py`
   - `src/**/*.py` — list modules, not every file
   - `eval/results/**`, `eval/reports/**`
   - `notes/specs/*.md`, `notes/handoffs/*`
3. For each training/eval entrypoint in `scripts/`, read the first 60 lines to
   extract the argparse surface and the main function signature.
4. For each `src/` top-level package, read `__init__.py` to list the public
   API.

## Output (≤30 lines)

```
# Repo map (generated <UTC>)

## entrypoints
- scripts/train_xling.py       args: --config --seed --run-name [...]
- scripts/eval_xnli.py         args: --run-name --split

## src packages
- src/xling/                   trainers, collators, losses
- src/data/                    loaders, registries
- src/eval/                    metrics, significance tests

## configs
- configs/train/*.yaml  (N files)
- configs/data/*.yaml   (N files; lock files: M)

## state
- runs/       K runs
- eval/reports dataset audits: L
- notes/specs: P specs
- data_changelog.md last entry: <date/hash>
```

## Enforcement

- MUST distinguish `src/` modules from `scripts/` entrypoints.
- MUST list any config schema found (not file names alone).
- MUST NOT make architectural claims not grounded in files read.

## Example invocation

```
Run repo-map and return a 20-line summary focused on cross-lingual transfer
modules under src/xling/.
```
