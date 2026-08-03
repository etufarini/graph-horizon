#!/usr/bin/env bash
#
# Runs both KV schemes for one explicit Q4_K_M model/profile/context tuple. It
# owns each synchronous profiler process and never retries another artifact,
# profile, context, KV scheme, or placement.

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model=""
backend=""
context=""

fail() {
    printf 'validate-kv: %s\n' "$*" >&2
    exit 2
}

while (($#)); do
    case "$1" in
        --model) (($# >= 2)) || fail "missing --model value"; model="$2"; shift 2 ;;
        --backend) (($# >= 2)) || fail "missing --backend value"; backend="$2"; shift 2 ;;
        --context) (($# >= 2)) || fail "missing --context value"; context="$2"; shift 2 ;;
        --help|-h)
            echo "usage: validate-kv.sh --model PATH --backend cpu|vulcan|vulcan-hybrid|metal|metal-hybrid --context N"
            exit 0
            ;;
        *) fail "unknown argument: $1" ;;
    esac
done

[[ -n "$model" ]] || fail "--model is required"
case "$backend" in cpu|vulcan|vulcan-hybrid|metal|metal-hybrid) ;; *) fail "invalid backend" ;; esac
[[ "$context" =~ ^[1-9][0-9]*$ ]] || fail "context must be >= 1"

if [[ ! -r "$model" ]]; then
    printf 'Q4_K_M %s: external verification: artifact is missing or unreadable\n' "$backend"
    exit 0
fi

for kv in f16 int8; do
    printf 'Q4_K_M %s %s: running\n' "$backend" "$kv"
    (
        cd "$project_dir"
        cargo run --locked --release --no-default-features --features "$backend" \
            --example profile -- "$model" "$context" "$kv"
    )
done
