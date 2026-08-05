#!/usr/bin/env bash
#
# Authenticates and inspects exactly the six catalogued Q4 artifacts read-only.
# It owns synchronous CPU inspector processes, creates no temp state, and never
# inspects an unauthenticated artifact or retries another artifact/profile.

set -euo pipefail
export LC_ALL=C

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
catalog="$project_dir/support/models.tsv"
source "$project_dir/support/artifact.sh"
models_dir=""

catalog_error() { printf 'validate-weights: catalog error: %s\n' "$*" >&2; exit 2; }
usage_error() { printf 'validate-weights: %s\n' "$*" >&2; exit 2; }
real_failure() { printf 'validate-weights: %s\n' "$*" >&2; exit 1; }

while (($#)); do
    case "$1" in
        --models-dir) (($# >= 2)) || usage_error "missing --models-dir value"; models_dir="$2"; shift 2 ;;
        --help|-h) echo "usage: validate-weights.sh --models-dir DIR"; exit 0 ;;
        *) usage_error "unknown argument: $1" ;;
    esac
done
[[ -n "$models_dir" ]] || usage_error "--models-dir is required"
[[ -r "$catalog" ]] || catalog_error "file is missing or unreadable"

set +e
catalog_rows="$(awk -F '\t' '
BEGIN {
    valid["3b-instruct"]="instruct"; valid["3b-reasoning"]="reasoning"
    valid["8b-instruct"]="instruct"; valid["8b-reasoning"]="reasoning"
    valid["14b-instruct"]="instruct"; valid["14b-reasoning"]="reasoning"
}
/^#/ { next }
{
    if (index($0, "\r") || NF != 6 || $0 ~ /\t\t/ || $0 ~ /^[[:space:]]|[[:space:]]$/) { bad=1; exit 2 }
    if (!($1 in valid) || valid[$1] != $2 || $3 !~ /^[A-Za-z0-9_.][A-Za-z0-9_.-]*$/ ||
        $6 !~ /^[A-Za-z0-9_.][A-Za-z0-9_.-]*$/ || $3 == "." || $3 == ".." || $6 == "." || $6 == ".." ||
        $4 !~ /^[1-9][0-9]*$/ || $5 !~ /^[0-9a-f]{64}$/) { bad=1; exit 2 }
    if (ids[$1]++ || q4[$3]++ || hashes[$5]++ || q8[$6]++) { bad=1; exit 2 }
    rows++; print
}
END { if (!bad && rows != 6) exit 2 }
' "$catalog")"
catalog_status=$?
set -e
((catalog_status == 0)) || catalog_error "invalid row, value, or duplicate"

missing_tool=""
command -v cargo >/dev/null 2>&1 || missing_tool="cargo unavailable"
[[ -n "$missing_tool" ]] || artifact_size_tool_available || missing_tool="size tool unavailable"
[[ -n "$missing_tool" ]] || artifact_hash_tool_available || missing_tool="SHA-256 tool unavailable"
verified=0; external=0
while IFS=$'\t' read -r id _ q4_file byte_count hash _; do
    if [[ -n "$missing_tool" ]]; then
        printf '%s: external verification: %s\n' "$id" "$missing_tool"; ((external += 1)); continue
    fi
    model="$models_dir/$q4_file"
    if [[ ! -r "$model" || ! -f "$model" ]]; then
        printf '%s: external verification: artifact is missing or unreadable\n' "$id"; ((external += 1)); continue
    fi
    size="$(artifact_size "$model")" || { printf '%s: external verification: byte count unavailable\n' "$id"; ((external += 1)); continue; }
    [[ "$size" == "$byte_count" ]] || real_failure "$id byte count mismatch"
    digest="$(artifact_sha256 "$model")" || { printf '%s: external verification: SHA-256 unavailable\n' "$id"; ((external += 1)); continue; }
    [[ "$digest" == "$hash" ]] || real_failure "$id SHA-256 mismatch"
    output="$(cd "$project_dir" && cargo run --quiet --locked -p graph_orizon_engine \
        --no-default-features --features cpu --example inspect -- "$model")" || real_failure "$id inspector failed"
    case "$id" in
        3b-*) dimensions="dimensions: blocks=26 hidden=3072 q=4096 k=1024 v=1024 ffn=9216 context=262144"; ownership="tied-to-embedding"; histogram=$'F32: 53\n  Q4_K: 156\n  Q6_K: 27' ;;
        8b-*) dimensions="dimensions: blocks=34 hidden=4096 q=4096 k=1024 v=1024 ffn=14336 context=262144"; ownership="dedicated"; histogram=$'F32: 69\n  Q4_K: 205\n  Q6_K: 35' ;;
        14b-*) dimensions="dimensions: blocks=40 hidden=5120 q=4096 k=1024 v=1024 ffn=16384 context=262144"; ownership="dedicated"; histogram=$'F32: 81\n  Q4_K: 241\n  Q6_K: 41' ;;
    esac
    grep -Fqx 'weight_profile: Q4_K_M' <<<"$output" || real_failure "$id profile mismatch"
    grep -Fqx "$dimensions" <<<"$output" || real_failure "$id dimensions mismatch"
    grep -Fqx "output: $ownership" <<<"$output" || real_failure "$id output ownership mismatch"
    [[ "$output" == *$'\ntensor_histogram:\n'* ]] || real_failure "$id histogram is missing"
    actual_histogram="${output##*$'\ntensor_histogram:\n'}"
    [[ "$actual_histogram" == "  $histogram" ]] || real_failure "$id histogram mismatch"
    printf '%s: pass\n' "$id"; ((verified += 1))
done <<<"$catalog_rows"
printf 'summary: pass=%d external_verification=%d total=6\n' "$verified" "$external"
