#!/usr/bin/env bash
#
# Starts local chat for one explicit profile/model/context/KV tuple. It owns only
# the final process, accepts no implicit profile, and never retries another
# artifact, profile, context, or KV scheme.

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model=""
backend=""
context=""
kv=""
max_tokens="128"

fail() {
    printf 'run-ghzero-engine: %s\n' "$*" >&2
    exit 2
}

while (($#)); do
    case "$1" in
        --model) (($# >= 2)) || fail "missing --model value"; model="$2"; shift 2 ;;
        --backend) (($# >= 2)) || fail "missing --backend value"; backend="$2"; shift 2 ;;
        --context) (($# >= 2)) || fail "missing --context value"; context="$2"; shift 2 ;;
        --kv) (($# >= 2)) || fail "missing --kv value"; kv="$2"; shift 2 ;;
        --max-tokens) (($# >= 2)) || fail "missing --max-tokens value"; max_tokens="$2"; shift 2 ;;
        --help|-h)
            echo "usage: run-ghzero-engine.sh --model PATH --backend cpu|vulcan|vulcan-hybrid|metal|metal-hybrid --context N --kv f16|int8 [--max-tokens N]"
            exit 0
            ;;
        *) fail "unknown argument: $1" ;;
    esac
done

[[ -r "$model" ]] || fail "model is missing or unreadable"
case "$backend" in cpu|vulcan|vulcan-hybrid|metal|metal-hybrid) ;; *) fail "invalid backend" ;; esac
case "$kv" in f16|int8) ;; *) fail "invalid KV scheme" ;; esac
[[ "$context" =~ ^[1-9][0-9]*$ ]] || fail "context must be >= 1"
[[ "$max_tokens" =~ ^[0-9]+$ ]] || fail "max tokens must be >= 0"

cd "$project_dir"
exec cargo run --locked --no-default-features --features "$backend" \
    --bin gh_zero_cli -- --provider local --model "$model" \
    --context-tokens "$context" --kv-quant "$kv" --max-tokens "$max_tokens"
