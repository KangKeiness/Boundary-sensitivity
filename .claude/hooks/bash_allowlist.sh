#!/usr/bin/env bash
# PreToolUse hook for Bash. Blocks destructive commands and networked fetches.
# Subagent-specific narrowing is enforced by the agent's `tools` frontmatter
# and prompt; this hook enforces a global floor.
set -euo pipefail

payload=$(cat)
cmd=$(printf '%s' "$payload" | python -c "import sys, json; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null || echo "")

if [ -z "$cmd" ]; then
  exit 0
fi

block() {
  cat <<JSON
{"decision":"block","reason":"bash_allowlist: $1"}
JSON
  exit 2
}

# Destructive and hard-to-reverse
case "$cmd" in
  *"rm -rf"*|*"rm -fr"*)            block "rm -rf denied";;
  *"git push --force"*|*"git push -f "*) block "force push denied";;
  *"git reset --hard"*)             block "git reset --hard denied";;
  *"git clean -f"*)                 block "git clean -f denied";;
  *"chmod 777"*)                    block "chmod 777 denied";;
  *"dd if="*)                       block "dd denied";;
  *":(){ :|:& };:"*)                block "fork bomb denied";;
esac

# Network
case "$cmd" in
  curl*|*" curl "*)                 block "curl denied — no network";;
  wget*|*" wget "*)                 block "wget denied — no network";;
  *"pip install"*|*"pip3 install"*) block "pip install denied — declare deps in spec";;
  *"pip uninstall"*)                block "pip uninstall denied";;
  *"huggingface-cli upload"*)       block "hf upload denied";;
  *"wandb login"*)                  block "wandb login denied";;
  *"aws s3 cp"*|*"aws s3 sync"*)    block "aws s3 write denied";;
esac

exit 0
