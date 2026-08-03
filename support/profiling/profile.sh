#!/usr/bin/env bash
#
# Runs the family-neutral profiler for one explicit model/profile/context/KV
# tuple and optional hybrid placement. It owns the profiler process and never
# retries another artifact, profile, context, scheme, or percentage.

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model=""
backend=""
context=""
kv=""
weights_percent=""

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
        --weights-percent) (($# >= 2)) || fail "missing --weights-percent value"; weights_percent="$2"; shift 2 ;;
        --help|-h)
            echo "usage: profile.sh --model PATH --backend cpu|vulcan|vulcan-hybrid|metal|metal-hybrid --context N --kv f16|int8 [--weights-percent 0..100]"
            exit 0
            ;;
        *) fail "unknown argument: $1" ;;
    esac
done

[[ -r "$model" ]] || fail "model is missing or unreadable"
case "$backend" in cpu|vulcan|vulcan-hybrid|metal|metal-hybrid) ;; *) fail "invalid backend" ;; esac
case "$kv" in f16|int8) ;; *) fail "invalid KV scheme" ;; esac
[[ "$context" =~ ^[1-9][0-9]*$ ]] || fail "context must be >= 1"
if [[ -n "$weights_percent" ]]; then
    [[ "$backend" == vulcan-hybrid || "$backend" == metal-hybrid ]] \
        || fail "--weights-percent requires a hybrid backend"
    [[ "$weights_percent" =~ ^[0-9]+$ ]] && ((weights_percent <= 100)) \
        || fail "weights percent must be in 0..100"
fi

cd "$project_dir"
arguments=("$model" "$context" "$kv")
[[ -z "$weights_percent" ]] || arguments+=(--weights-percent "$weights_percent")
exec cargo run --locked --release --no-default-features --features "$backend" \
    --example profile -- "${arguments[@]}"
