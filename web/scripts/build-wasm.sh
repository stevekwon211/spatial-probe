#!/usr/bin/env bash
# Build the Rust/WASM MDC mesher into the web. Homebrew rust lacks the wasm target, so build via
# the rustup toolchain. Output lands next to the glue + public/mdc so webpack emits the .wasm asset.
set -euo pipefail
cd "$(dirname "$0")/../engines/mdc-wasm"
PATH="$HOME/.cargo/bin:$(dirname "$(rustup which cargo)"):$PATH" \
  wasm-pack build --release --target web --out-dir /tmp/mdc-pkg
cp /tmp/mdc-pkg/mdc_wasm.js /tmp/mdc-pkg/mdc_wasm.d.ts /tmp/mdc-pkg/mdc_wasm_bg.wasm /tmp/mdc-pkg/mdc_wasm_bg.wasm.d.ts \
  ../../components/freespace/mdc/
cp /tmp/mdc-pkg/mdc_wasm_bg.wasm ../../public/mdc/mdc_wasm_bg.wasm
echo "wasm built + copied into components/freespace/mdc + public/mdc"
