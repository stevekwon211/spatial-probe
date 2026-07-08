#!/usr/bin/env bash
set -euo pipefail

# Materialize ignored env files from cloud environment variables.
# This script never prints secret values. It only reports destination paths and
# key counts so cloud-session setup can be verified without leaking contents.

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

decode_base64_value() {
  local encoded="$1"
  if printf '%s' "$encoded" | base64 --decode >/dev/null 2>&1; then
    printf '%s' "$encoded" | base64 --decode
  else
    printf '%s' "$encoded" | base64 -D
  fi
}

is_safe_relative_path() {
  local path="$1"
  [ -n "$path" ] || return 1
  case "$path" in
    /*|../*|*/../*|*/..|..)
      return 1
      ;;
  esac
  return 0
}

write_content_b64() {
  local encoded="$1"
  local dest="$2"
  local label="$3"

  if ! is_safe_relative_path "$dest"; then
    echo "warning: refusing unsafe env file path for $label: $dest" >&2
    return 1
  fi

  mkdir -p "$(dirname "$dest")"
  local tmp
  tmp="$(mktemp "${dest}.tmp.XXXXXX")"
  if ! decode_base64_value "$encoded" >"$tmp"; then
    rm -f "$tmp"
    echo "warning: failed to decode $label into $dest" >&2
    return 1
  fi

  chmod 600 "$tmp"
  mv "$tmp" "$dest"

  local key_count
  key_count="$(awk -F= '/^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=/ {c++} END {print c+0}' "$dest")"
  echo "materialized $label -> $dest ($key_count keys)"
}

write_env_file_from_var() {
  local var_name="$1"
  local dest="$2"
  local label="$3"
  local encoded="${!var_name:-}"

  [ -n "$encoded" ] || return 0
  write_content_b64 "$encoded" "$dest" "$label"
}

write_manifest_from_var() {
  local manifest_var="$1"
  local encoded_manifest="${!manifest_var:-}"
  [ -n "$encoded_manifest" ] || return 0

  local manifest
  manifest="$(mktemp)"
  if ! decode_base64_value "$encoded_manifest" >"$manifest"; then
    rm -f "$manifest"
    echo "warning: failed to decode $manifest_var" >&2
    return 1
  fi

  while IFS=$'\t' read -r dest encoded_content || [ -n "$dest" ]; do
    case "$dest" in
      ""|\#*)
        continue
        ;;
    esac
    if [ -z "${encoded_content:-}" ]; then
      echo "warning: skipping malformed env manifest row for path: $dest" >&2
      continue
    fi
    write_content_b64 "$encoded_content" "$dest" "manifest:$dest"
  done <"$manifest"

  rm -f "$manifest"
}

sanitize_var_component() {
  printf '%s' "$1" \
    | tr '[:lower:]' '[:upper:]' \
    | sed -E 's/[^A-Z0-9]+/_/g; s/^_+//; s/_+$//' \
    | sed -E 's/^([0-9])/_\1/'
}

package_name=""
if [ -f package.json ] && command -v node >/dev/null 2>&1; then
  package_name="$(node -e 'try { process.stdout.write(require("./package.json").name || ""); } catch {}' 2>/dev/null || true)"
fi

repo_slug="$(sanitize_var_component "$(basename "$repo_root")")"
package_slug="$(sanitize_var_component "$package_name")"

write_prefixed_defaults() {
  local prefix="$1"
  [ -n "$prefix" ] || return 0
  write_env_file_from_var "${prefix}_ROOT_ENV_B64" ".env" "${prefix}_ROOT_ENV_B64"
  write_env_file_from_var "${prefix}_ROOT_ENV_LOCAL_B64" ".env.local" "${prefix}_ROOT_ENV_LOCAL_B64"
  write_env_file_from_var "${prefix}_WEB_ENV_LOCAL_B64" "apps/web/.env.local" "${prefix}_WEB_ENV_LOCAL_B64"
  write_env_file_from_var "${prefix}_FUNCTIONS_ENV_LOCAL_B64" "functions/.env.local" "${prefix}_FUNCTIONS_ENV_LOCAL_B64"
  write_env_file_from_var "${prefix}_SERVICES_FUNCTIONS_ENV_LOCAL_B64" "services/functions/.env.local" "${prefix}_SERVICES_FUNCTIONS_ENV_LOCAL_B64"
  write_manifest_from_var "${prefix}_ENV_FILE_MANIFEST_B64"
}

# Generic defaults for common repo layouts.
write_env_file_from_var CLAUDE_ROOT_ENV_B64 ".env" "CLAUDE_ROOT_ENV_B64"
write_env_file_from_var CLAUDE_ROOT_ENV_LOCAL_B64 ".env.local" "CLAUDE_ROOT_ENV_LOCAL_B64"
write_env_file_from_var CLAUDE_WEB_ENV_LOCAL_B64 "apps/web/.env.local" "CLAUDE_WEB_ENV_LOCAL_B64"
write_env_file_from_var CLAUDE_FUNCTIONS_ENV_LOCAL_B64 "functions/.env.local" "CLAUDE_FUNCTIONS_ENV_LOCAL_B64"
write_env_file_from_var CLAUDE_SERVICES_FUNCTIONS_ENV_LOCAL_B64 "services/functions/.env.local" "CLAUDE_SERVICES_FUNCTIONS_ENV_LOCAL_B64"

# FODI monorepo defaults.
if [ -f AGENTS.md ] && [ -f apps/web/package.json ] && [ -f services/functions/package.json ]; then
  write_env_file_from_var FODI_ROOT_ENV_LOCAL_B64 ".env.local" "FODI_ROOT_ENV_LOCAL_B64"
  write_env_file_from_var FODI_WEB_ENV_LOCAL_B64 "apps/web/.env.local" "FODI_WEB_ENV_LOCAL_B64"
  write_env_file_from_var FODI_IOS_ENV_LOCAL_B64 "apps/ios/.env.local" "FODI_IOS_ENV_LOCAL_B64"
  write_env_file_from_var FODI_ANDROID_ENV_B64 "apps/android/.env" "FODI_ANDROID_ENV_B64"
  write_env_file_from_var FODI_FUNCTIONS_ENV_MACC_QED_B64 "services/functions/.env.macc-qed" "FODI_FUNCTIONS_ENV_MACC_QED_B64"
fi

# Generic escape hatch for any repo layout. The decoded manifest is TSV:
# relative/path<TAB>base64_file_content
write_manifest_from_var CLAUDE_ENV_FILE_MANIFEST_B64

# Repo-specific variables let one shared "All Repos" cloud environment hold
# env files for multiple repos without writing every env file into every clone.
# Examples:
#   CLAUDE_ROVEN_ENV_FILE_MANIFEST_B64
#   CLAUDE_FODI_WEB_NEXT_JS_ROOT_ENV_LOCAL_B64
write_prefixed_defaults "CLAUDE_${repo_slug}"
if [ -n "$package_slug" ] && [ "$package_slug" != "$repo_slug" ]; then
  write_prefixed_defaults "CLAUDE_${package_slug}"
fi
