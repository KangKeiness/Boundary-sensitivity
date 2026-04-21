#!/usr/bin/env bash
# PostToolUse hook for Write|Edit. Runs ruff/black-check on touched Python files.
# Non-blocking: reports issues but does not fail the tool call, because
# formatting issues are caught by CI and writer has explicit budget to fix.
set -euo pipefail

payload=$(cat)
file_path=$(printf '%s' "$payload" | python -c "import sys, json; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null || echo "")

if [ -z "$file_path" ]; then exit 0; fi
case "$file_path" in *.py) ;; *) exit 0 ;; esac

if command -v ruff >/dev/null 2>&1; then
  ruff check "$file_path" >&2 || true
  ruff format --check "$file_path" >&2 || true
fi

exit 0
