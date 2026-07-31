#!/usr/bin/env bash
#
# Runs the retained family-neutral profile example for one explicit immutable
# model/backend/context/KV tuple. No backend or context retry is performed.

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model=""
backend=""
context=""
kv=""

fail() {
    printf 'profile: %s\n' "$*" >&2
    exit 2
}

while (($#)); do
    case "$1" in
        --model) (($# >= 2)) || fail "missing --model value"; model="$2"; shift 2 ;;
        --backend) (($# >= 2)) || fail "missing --backend value"; backend="$2"; shift 2 ;;
        --context) (($# >= 2)) || fail "missing --context value"; context="$2"; shift 2 ;;
        --kv) (($# >= 2)) || fail "missing --kv value"; kv="$2"; shift 2 ;;
        --help|-h)
            echo "usage: profile.sh --model PATH --backend cpu|vulkan|hybrid --context N --kv f16|int8"
            exit 0
            ;;
        *) fail "unknown argument: $1" ;;
    esac
done

[[ -r "$model" ]] || fail "model is missing or unreadable"
case "$backend" in cpu|vulkan|hybrid) ;; *) fail "invalid backend" ;; esac
case "$kv" in f16|int8) ;; *) fail "invalid KV scheme" ;; esac
[[ "$context" =~ ^[1-9][0-9]*$ ]] || fail "context must be >= 1"

cd "$project_dir"
exec cargo run --locked --release --no-default-features --features "$backend" \
    --example profile -- "$model" "$context" "$kv"
