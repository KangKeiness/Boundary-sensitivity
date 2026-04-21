#!/usr/bin/env bash
# PreToolUse hook for Bash. Blocks training launches unless
# --seed, --config, --run-name are all present AND a fresh clean dataset audit
# exists under eval/reports/.
set -euo pipefail

payload=$(cat)
cmd=$(printf '%s' "$payload" | python -c "import sys, json; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null || echo "")

if [ -z "$cmd" ]; then exit 0; fi

is_training=0
case "$cmd" in
  *"scripts/train_"*|*"torchrun "*|*"accelerate launch "*|*"python -m src.train"*)
    is_training=1 ;;
esac

if [ "$is_training" -eq 0 ]; then exit 0; fi

block() {
  cat <<JSON
{"decision":"block","reason":"training_guard: $1"}
JSON
  exit 2
}

# Required flags
case "$cmd" in *"--seed"*) ;; *) block "training command missing --seed";; esac
case "$cmd" in *"--config"*) ;; *) block "training command missing --config";; esac
case "$cmd" in *"--run-name"*) ;; *) block "training command missing --run-name";; esac

# Require at least one dataset audit with block_training: false, fresh (<24h)
audit=$(ls -t eval/reports/dataset_audit_*.json 2>/dev/null | head -n1 || true)
if [ -z "$audit" ]; then
  block "no dataset audit under eval/reports/. Run data-auditor first."
fi

if ! python -c "import json,sys; d=json.load(open('$audit')); sys.exit(0 if d.get('block_training') is False else 1)" 2>/dev/null; then
  block "latest dataset audit '$audit' has block_training != false. Resolve dataset issues first."
fi

now=$(date +%s)
mtime=$(python -c "import os; print(int(os.path.getmtime('$audit')))" 2>/dev/null || echo 0)
age=$(( now - mtime ))
if [ "$age" -gt 86400 ]; then
  block "dataset audit '$audit' is stale (>24h). Re-run data-auditor."
fi

exit 0
