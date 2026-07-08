#!/usr/bin/env bash
set -euo pipefail

is_repo_root() {
  [ -d "$1" ] || return 1
  [ -f "$1/.git" ] || [ -d "$1/.git" ] || [ -f "$1/package.json" ] || [ -f "$1/pubspec.yaml" ] || [ -f "$1/settings.gradle" ] || [ -f "$1/settings.gradle.kts" ] || [ -f "$1/Package.swift" ] || return 1
}

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if ! is_repo_root "$repo_root"; then
  repo_root=""
  for candidate in \
    "${CLAUDE_WORKSPACE_DIR:-}" \
    "${CLAUDE_PROJECT_DIR:-}" \
    "${CLAUDE_REPO_DIR:-}" \
    "${GITHUB_WORKSPACE:-}" \
    "${WORKSPACE_DIR:-}" \
    "${REPO_DIR:-}" \
    "$PWD" \
    "$HOME" \
    "$HOME/project" \
    "$HOME/workspace" \
    "/workspace" \
    "/workspaces" \
    "/mnt/data" \
    "/mnt" \
    "/repo" \
    "/home"; do
    [ -n "$candidate" ] || continue
    if is_repo_root "$candidate"; then
      repo_root="$candidate"
      break
    fi
  done
fi

if [ -z "$repo_root" ]; then
  for search_root in "$HOME" "/workspace" "/workspaces" "/mnt/data" "/mnt" "/repo" "/home" "/tmp"; do
    [ -d "$search_root" ] || continue
    repo_marker="$(find "$search_root" -maxdepth 6 -name .git -print -quit 2>/dev/null || true)"
    if [ -z "$repo_marker" ]; then
      repo_marker="$(find "$search_root" -maxdepth 6 \
        \( -path '*/.*' -o -path '*/node_modules/*' \) -prune -o \
        \( -name package.json -o -name pubspec.yaml -o -name settings.gradle -o -name settings.gradle.kts -o -name Package.swift \) -print -quit 2>/dev/null || true)"
    fi
    if [ -n "$repo_marker" ]; then
      repo_root="$(dirname "$repo_marker")"
      break
    fi
  done
fi

if ! is_repo_root "$repo_root"; then
  echo "error: could not locate repository root from PWD=$PWD HOME=$HOME" >&2
  exit 1
fi
cd "$repo_root"
repo_name="$(basename "$repo_root")"

if [ -f tools/claude/materialize-env-files.sh ]; then
  bash tools/claude/materialize-env-files.sh
fi

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

if ! grep -q "Claude Cloud generic repo setup" "$HOME/.bashrc" 2>/dev/null; then
  cat >>"$HOME/.bashrc" <<'BASHRC'

# Claude Cloud generic repo setup
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

node_version=""
if [ -f .nvmrc ]; then
  node_version="$(tr -d '[:space:]' < .nvmrc)"
elif [ -f package.json ] && command -v node >/dev/null 2>&1; then
  node_version="$(node -e 'try { const v = require("./package.json").engines?.node || ""; console.log(String(v).replace(/^[^0-9]*/, "").split(/[ <>=|]/)[0]); } catch { console.log(""); }' 2>/dev/null || true)"
fi

if [ -n "$node_version" ] && command -v nvm >/dev/null 2>&1; then
  nvm use "$node_version" || nvm install "$node_version"
fi

if command -v corepack >/dev/null 2>&1; then
  corepack enable || true
fi

prepare_package_manager() {
  [ -f package.json ] || return 0
  command -v node >/dev/null 2>&1 || return 0
  local package_manager
  package_manager="$(node -e 'try { console.log(require("./package.json").packageManager || ""); } catch { console.log(""); }' 2>/dev/null || true)"
  case "$package_manager" in
    pnpm@*|yarn@*)
      if command -v corepack >/dev/null 2>&1; then
        corepack prepare "$package_manager" --activate || true
      fi
      ;;
  esac
}

ensure_pnpm() {
  if ! command -v pnpm >/dev/null 2>&1; then
    npm install -g pnpm
  fi
}

install_node_dir() {
  local dir="$1"
  [ -f "$dir/package.json" ] || return 0
  (
    cd "$dir"
    prepare_package_manager
    if [ -f pnpm-lock.yaml ]; then
      ensure_pnpm
      pnpm install --frozen-lockfile
    elif [ -f package-lock.json ]; then
      npm ci
    elif [ -f yarn.lock ]; then
      if command -v yarn >/dev/null 2>&1; then
        yarn install --immutable || yarn install --frozen-lockfile
      else
        npm install
      fi
    elif [ -f bun.lockb ] || [ -f bun.lock ]; then
      if command -v bun >/dev/null 2>&1; then
        bun install --frozen-lockfile
      else
        npm install
      fi
    else
      npm install
    fi
  )
}

if [ -f package.json ]; then
  install_node_dir "."
fi

for dir in functions services/functions services/realtime firebase/test scripts web apps/web; do
  [ -d "$dir" ] || continue
  if [ -f "$dir/package-lock.json" ] || [ -f "$dir/pnpm-lock.yaml" ] || [ -f "$dir/yarn.lock" ] || [ -f "$dir/bun.lockb" ] || [ -f "$dir/bun.lock" ]; then
    install_node_dir "$dir"
  fi
done

if [ -f firebase.json ] || [ -f .firebaserc ]; then
  if ! command -v firebase >/dev/null 2>&1; then
    npm install -g firebase-tools
  fi
fi

if [ -f requirements.txt ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
elif [ -f pyproject.toml ]; then
  if command -v uv >/dev/null 2>&1; then
    uv sync --frozen || uv sync
  else
    echo "warning: pyproject.toml found in $repo_name but uv is unavailable; skipping Python dependency sync" >&2
  fi
fi

if ! command -v gitleaks >/dev/null 2>&1 && command -v go >/dev/null 2>&1; then
  GOBIN="$HOME/.local/bin" go install github.com/zricethezav/gitleaks/v8@v8.28.0 \
    || echo "warning: gitleaks install failed; continuing without it" >&2
fi

echo "Claude Cloud setup complete for $repo_name."
