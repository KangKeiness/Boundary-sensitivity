#!/usr/bin/env bash
# PreToolUse hook for Write|Edit. Blocks writes to sensitive paths regardless
# of symlinks, path aliasing, or glob edge cases that permissions.deny might miss.
#
# Reads tool invocation JSON from stdin. Exits non-zero with a JSON decision
# object to block.
set -euo pipefail

payload=$(cat)

# Extract file_path from tool_input (Write/Edit use file_path)
file_path=$(printf '%s' "$payload" | python -c "import sys, json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path',''))" 2>/dev/null || echo "")

if [ -z "$file_path" ]; then
  exit 0
fi

# Resolve realpath (follow symlinks). Fall back to as-is if realpath fails.
resolved=$(python - <<PY 2>/dev/null || echo "$file_path"
import os, sys
p = "$file_path".replace("\\", "/")
try:
    print(os.path.realpath(p))
except Exception:
    print(p)
PY
)

# Normalize: lowercase on Windows-ish, strip trailing slash, collapse //.
norm=$(printf '%s' "$resolved" | tr '\\' '/' | sed 's://*:/:g; s:/$::')

# Deny patterns. These are structural, not glob — anchored on dir boundaries.
deny_patterns=(
  "/paper/"
  "/data/raw/"
  "/runs/"
  "/outputs/"
  "/configs/secrets/"
)

# Exact basename denies
deny_basenames=(
  ".env"
)

for pat in "${deny_patterns[@]}"; do
  case "/$norm/" in
    *"$pat"*)
      cat <<JSON
{"decision":"block","reason":"block_sensitive_paths: write denied to protected path '$file_path' (resolved: '$resolved'). This rule is enforced by hook, not by permissions.deny, because deny patterns are bypassable."}
JSON
      exit 2
      ;;
  esac
done

base=$(basename "$norm")
for b in "${deny_basenames[@]}"; do
  if [ "$base" = "$b" ] || [[ "$base" == "$b".* ]]; then
    cat <<JSON
{"decision":"block","reason":"block_sensitive_paths: write to env-like file '$base' denied."}
JSON
    exit 2
  fi
done

exit 0
