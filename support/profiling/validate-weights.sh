#!/usr/bin/env bash
#
# Verifies pinned size/SHA and artifact histogram read-only, then runs internal
# format tests separately. Artifact counts are facts, never loader whitelists.

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_q8=""
model_q4=""

fail() {
    printf 'validate-weights: %s\n' "$*" >&2
    exit 2
}

while (($#)); do
    case "$1" in
        --model-q8) (($# >= 2)) || fail "missing --model-q8 value"; model_q8="$2"; shift 2 ;;
        --model-q4) (($# >= 2)) || fail "missing --model-q4 value"; model_q4="$2"; shift 2 ;;
        --help|-h)
            echo "usage: validate-weights.sh --model-q8 PATH --model-q4 PATH"
            exit 0
            ;;
        *) fail "unknown argument: $1" ;;
    esac
done

command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is required"
external_failed=0

verify() {
    local profile="$1" model="$2" expected_size="$3" expected_sha="$4"
    shift 4
    if [[ ! -r "$model" ]]; then
        printf '%s: not verified: artifact is missing or unreadable\n' "$profile"
        return
    fi
    local size digest output
    size="$(wc -c < "$model")"
    digest="$(sha256sum "$model")"
    digest="${digest%% *}"
    if [[ "$size" != "$expected_size" || "$digest" != "$expected_sha" ]]; then
        printf '%s: not verified: artifact mismatch (size=%s sha256=%s)\n' \
            "$profile" "$size" "$digest"
        external_failed=1
        return
    fi
    output="$(
        cd "$project_dir"
        cargo run --quiet --locked -p gh_zero_engine --no-default-features \
            --features cpu --example inspect -- "$model"
    )"
    grep -F "weight_profile: $profile" <<<"$output" >/dev/null
    for fact in "$@"; do
        grep -F "  $fact" <<<"$output" >/dev/null \
            || fail "$profile histogram does not contain $fact"
    done
    printf '%s: verified size, SHA-256 and histogram\n' "$profile"
}

verify Q8_0 "$model_q8" 3652204704 \
    8c2b72eb5861304fcfd5e82f1eddd6efa4115737f4239fd216a028a8852413ef \
    "F32: 53" "Q8_0: 183"
verify Q4_K_M "$model_q4" 2147023008 \
    9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8 \
    "F32: 53" "Q4_K: 156" "Q6_K: 27"

printf 'internal formats: running CPU synthetic coverage\n'
(
    cd "$project_dir"
    cargo test --locked -p gh_zero_engine --no-default-features --features cpu \
        load_tensor_preserves_every_internal_weight_format
)
printf 'internal formats: running Vulkan registration coverage\n'
(
    cd "$project_dir"
    cargo test --locked -p gh_zero_engine --no-default-features --features vulkan \
        retained_weight_formats_have_reachable_dense_pipelines
)

exit "$external_failed"
