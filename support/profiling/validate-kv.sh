#!/usr/bin/env bash
#
# Runs f16 and int8 chat checks for both public profiles on one explicit backend.
# Missing artifacts are external not-verified rows; an attempted failing run is
# a real failure and is never retried with another feature or context.

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_q8=""
model_q4=""
backend=""
context=""

fail() {
    printf 'validate-kv: %s\n' "$*" >&2
    exit 2
}

while (($#)); do
    case "$1" in
        --model-q8) (($# >= 2)) || fail "missing --model-q8 value"; model_q8="$2"; shift 2 ;;
        --model-q4) (($# >= 2)) || fail "missing --model-q4 value"; model_q4="$2"; shift 2 ;;
        --backend) (($# >= 2)) || fail "missing --backend value"; backend="$2"; shift 2 ;;
        --context) (($# >= 2)) || fail "missing --context value"; context="$2"; shift 2 ;;
        --help|-h)
            echo "usage: validate-kv.sh --model-q8 PATH --model-q4 PATH --backend cpu|vulkan|hybrid --context N"
            exit 0
            ;;
        *) fail "unknown argument: $1" ;;
    esac
done

case "$backend" in cpu|vulkan|hybrid) ;; *) fail "invalid backend" ;; esac
[[ "$context" =~ ^[1-9][0-9]*$ ]] || fail "context must be >= 1"

ran=0
for entry in "Q8_0:$model_q8" "Q4_K_M:$model_q4"; do
    profile="${entry%%:*}"
    model="${entry#*:}"
    if [[ ! -r "$model" ]]; then
        printf '%s %s: not verified: artifact is missing or unreadable\n' "$profile" "$backend"
        continue
    fi
    ran=1
    for kv in f16 int8; do
        printf '%s %s %s: running\n' "$profile" "$backend" "$kv"
        (
            cd "$project_dir"
            cargo run --locked --release --no-default-features --features "$backend" \
                --example profile -- "$model" "$context" "$kv"
        )
    done
done

if ((ran == 0)); then
    printf 'validate-kv: not verified: no pinned artifact is available\n'
fi
