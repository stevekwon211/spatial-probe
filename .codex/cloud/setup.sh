#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
cd "$repo_root"

export CI="${CI:-1}"
export HUSKY="${HUSKY:-0}"
export NEXT_TELEMETRY_DISABLED="${NEXT_TELEMETRY_DISABLED:-1}"
export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD="${PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD:-1}"
export PUPPETEER_SKIP_DOWNLOAD="${PUPPETEER_SKIP_DOWNLOAD:-1}"
export npm_config_audit="${npm_config_audit:-false}"
export npm_config_fund="${npm_config_fund:-false}"

mkdir -p "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"
export NPM_CONFIG_PREFIX="$HOME/.local"

if ! grep -q "Codex Cloud bootstrap" "$HOME/.bashrc" 2>/dev/null; then
  cat >>"$HOME/.bashrc" <<'BASHRC'

# Codex Cloud bootstrap
export PATH="$HOME/.local/bin:$PATH"
export NPM_CONFIG_PREFIX="$HOME/.local"
export CI=1
export HUSKY=0
export NEXT_TELEMETRY_DISABLED=1
export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
export PUPPETEER_SKIP_DOWNLOAD=1
export npm_config_audit=false
export npm_config_fund=false
BASHRC
fi

activate_pnpm() {
  local version="${1:-10.12.1}"
  if command -v corepack >/dev/null 2>&1; then
    corepack enable || true
    corepack prepare "pnpm@$version" --activate || npm install -g "pnpm@$version"
    pnpm_cmd=(corepack pnpm)
  elif ! command -v pnpm >/dev/null 2>&1; then
    npm install -g "pnpm@$version"
    pnpm_cmd=(pnpm)
  else
    pnpm_cmd=(pnpm)
  fi
}

pnpm_version() {
  node -e 'try { const pm = require("./package.json").packageManager || ""; const m = pm.match(/^pnpm@(.+)$/); process.stdout.write(m ? m[1] : "10.12.1"); } catch { process.stdout.write("10.12.1"); }'
}

if [ -f pnpm-lock.yaml ]; then
  pnpm_cmd=(pnpm)
  activate_pnpm "$(pnpm_version)"
  "${pnpm_cmd[@]}" install --frozen-lockfile
elif [ -f package-lock.json ]; then
  npm ci
elif [ -f package.json ]; then
  npm install --no-package-lock
fi

if [ ! -f pnpm-lock.yaml ] && [ ! -f package-lock.json ]; then
  while IFS= read -r lockfile; do
    dir="$(dirname "$lockfile")"
    (cd "$dir" && npm ci)
  done < <(find . -path ./node_modules -prune -o -path '*/node_modules' -prune -o -path ./.next -prune -o -path '*/.next' -prune -o -name package-lock.json -print)
fi

if [ "${CODEX_SKIP_PYTHON:-0}" != "1" ] && command -v python3 >/dev/null 2>&1; then
  if [ -f pyproject.toml ]; then
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -e .
  elif [ -f requirements.txt ]; then
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
  fi
fi

if [ "${CODEX_SKIP_RUST:-0}" != "1" ] && command -v cargo >/dev/null 2>&1; then
  while IFS= read -r manifest; do
    cargo fetch --manifest-path "$manifest"
  done < <(find . -path ./target -prune -o -path '*/target' -prune -o -name Cargo.toml -print)
fi

echo "$(basename "$repo_root") Codex Cloud setup complete."
