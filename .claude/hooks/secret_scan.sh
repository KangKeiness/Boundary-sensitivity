#!/usr/bin/env bash
# PreToolUse hook for Write|Edit. Scans the proposed content for secrets.
set -euo pipefail

payload=$(cat)

# Extract Write.content or Edit.new_string via jq. Fail open (exit 0) on extraction error.
content=$(printf '%s' "$payload" | jq -r '(.tool_input.content // .tool_input.new_string // "")' 2>/dev/null) || content=""

if [ -z "$content" ]; then
  exit 0
fi

patterns=(
  'AKIA[0-9A-Z]{16}'
  'aws_secret_access_key[[:space:]]*=[[:space:]]*[A-Za-z0-9/+=]{40}'
  'sk-[A-Za-z0-9]{32,}'
  'sk-ant-[A-Za-z0-9_-]{20,}'
  'hf_[A-Za-z0-9]{30,}'
  '-----BEGIN (RSA|OPENSSH|EC|DSA|PRIVATE) KEY-----'
  '(api[_-]?key|secret|token|password)[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"']{16,}["'"'"']'
)

for pat in "${patterns[@]}"; do
  if printf '%s' "$content" | grep -qE -e "$pat"; then
    cat <<JSON
{"decision":"block","reason":"secret_scan: proposed content matches secret pattern /$pat/. Refusing to write. If this is a false positive, redact or store in an env var."}
JSON
    exit 2
  fi
done

exit 0
