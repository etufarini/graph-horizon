#!/usr/bin/env bash
# measure.sh — capture comparable before/after metrics for a Rust ablation.
#
# Usage:
#   bash scripts/measure.sh                 # measure current dir
#   bash scripts/measure.sh /path/to/repo   # measure a specific repo root
#
# Output: a single JSON object on stdout. Run it once for the baseline and once
# after simplification, then diff the two.
#
# Metrics:
#   build_ok            : does `cargo build --all-targets` succeed (true/false)
#   warnings            : compiler warning count from that build
#   rust_loc            : lines of Rust source (uses `tokei` if present, else find+wc)
#   direct_deps         : number of direct dependencies (cargo tree --depth 1)
#   total_deps          : number of unique dependencies incl. transitive
#   release_binary_bytes: size of the release binary if exactly one is found, else null
#
# Optional tools that improve results if installed: tokei. Everything else uses
# only the stock cargo toolchain. Requires: bash, cargo, awk/sort/wc (coreutils).

set -uo pipefail

REPO="${1:-.}"
cd "$REPO" 2>/dev/null || { echo "{\"error\":\"cannot cd into $REPO\"}"; exit 1; }

if ! command -v cargo >/dev/null 2>&1; then
  echo "{\"error\":\"cargo not found on PATH\"}"
  exit 1
fi

# --- build status + warning count ---
build_log="$(mktemp)"
if cargo build --all-targets >"$build_log" 2>&1; then
  build_ok=true
else
  build_ok=false
fi
warnings="$(grep -c -E '^warning' "$build_log" 2>/dev/null || true)"
rm -f "$build_log"

# --- lines of Rust ---
if command -v tokei >/dev/null 2>&1; then
  rust_loc="$(tokei --output json 2>/dev/null \
    | grep -o '"Rust"[^}]*"code":[0-9]*' \
    | grep -o '"code":[0-9]*' | grep -o '[0-9]*' | head -n1)"
fi
if [ -z "${rust_loc:-}" ]; then
  rust_loc="$(find . -path ./target -prune -o -name '*.rs' -print 2>/dev/null \
    | xargs wc -l 2>/dev/null | tail -n1 | awk '{print $1}')"
fi
rust_loc="${rust_loc:-0}"

# --- dependency counts (cargo-native, no jq needed) ---
# Direct deps: immediate children of the root in the tree.
direct_deps="$(cargo tree --depth 1 --prefix none 2>/dev/null \
  | tail -n +2 | grep -v '^[[:space:]]*$' | wc -l | tr -d ' ')"
direct_deps="${direct_deps:-0}"

# Total unique deps across the whole tree (name-deduplicated).
total_deps="$(cargo tree --prefix none 2>/dev/null \
  | grep -v '^[[:space:]]*$' | awk '{print $1}' | sort -u | wc -l | tr -d ' ')"
total_deps="${total_deps:-0}"

# --- release binary size (best effort) ---
release_binary_bytes=null
if cargo build --release >/dev/null 2>&1; then
  mapfile -t bins < <(find target/release -maxdepth 1 -type f -perm -u+x \
    ! -name '*.d' ! -name '*.rlib' ! -name '*.so' 2>/dev/null)
  if [ "${#bins[@]}" -eq 1 ]; then
    release_binary_bytes="$(wc -c < "${bins[0]}" | tr -d ' ')"
  fi
fi

printf '{"build_ok":%s,"warnings":%s,"rust_loc":%s,"direct_deps":%s,"total_deps":%s,"release_binary_bytes":%s}\n' \
  "$build_ok" "$warnings" "$rust_loc" "$direct_deps" "$total_deps" "$release_binary_bytes"
