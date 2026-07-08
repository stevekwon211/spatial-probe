#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"
repo_name="$(basename "$repo_root")"

if ! command -v claude >/dev/null 2>&1; then
  echo "error: claude CLI is not installed or not on PATH" >&2
  exit 127
fi

if [ -n "${ANTHROPIC_BASE_URL:-}" ] && [ "${ANTHROPIC_BASE_URL}" != "https://api.anthropic.com" ] && [ "${ANTHROPIC_BASE_URL}" != "api.anthropic.com" ]; then
  echo "warning: unsetting ANTHROPIC_BASE_URL for Remote Control; it requires api.anthropic.com" >&2
  unset ANTHROPIC_BASE_URL
fi

unset ANTHROPIC_API_KEY
unset ANTHROPIC_AUTH_TOKEN
unset CLAUDE_CODE_OAUTH_TOKEN

export CLAUDE_REMOTE_CONTROL_SESSION_NAME_PREFIX="${CLAUDE_REMOTE_CONTROL_SESSION_NAME_PREFIX:-$repo_name}"

if [ "$#" -gt 0 ]; then
  exec claude remote-control "$@"
fi

name="${CLAUDE_RC_NAME:-$repo_name}"
spawn="${CLAUDE_RC_SPAWN:-worktree}"
capacity="${CLAUDE_RC_CAPACITY:-4}"

cmd=(claude remote-control --name "$name" --spawn "$spawn")
if [ "$spawn" != "session" ]; then
  cmd+=(--capacity "$capacity")
fi

exec "${cmd[@]}"
