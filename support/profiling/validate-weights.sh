#!/usr/bin/env bash
#
# Authenticates the six catalogued Q4 validation artifacts, then checks their
# runtime metadata and exact tensor histograms read-only. Identity facts do not
# expand runtime capability, and an identity mismatch is never inspected.

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
catalog="$project_dir/support/models.tsv"
models_dir=""

catalog_error() {
    printf 'validate-weights: catalog error: %s\n' "$*" >&2
    exit 2
}

usage_error() {
    printf 'validate-weights: %s\n' "$*" >&2
    exit 2
}

real_failure() {
    printf 'validate-weights: %s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --models-dir) (($# >= 2)) || usage_error "missing --models-dir value"; models_dir="$2"; shift 2 ;;
        --help|-h) echo "usage: validate-weights.sh --models-dir DIR"; exit 0 ;;
        *) usage_error "unknown argument: $1" ;;
    esac
done

[[ -n "$models_dir" ]] || usage_error "--models-dir is required"
[[ -r "$catalog" ]] || catalog_error "file is missing or unreadable"

declare -a ids chats q4_files byte_counts hashes q8_files
declare -A seen_ids seen_q4 seen_hashes seen_q8
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == \#* ]] && continue
    [[ -n "$line" && "$line" != *$'\r'* && "$line" != *$'\t\t'* \
        && "$line" != $'\t'* && "$line" != *$'\t' ]] || catalog_error "invalid row"
    IFS=$'\t' read -r -a fields <<<"$line"
    ((${#fields[@]} == 6)) || catalog_error "expected six columns"
    id="${fields[0]}"; chat="${fields[1]}"; q4="${fields[2]}"
    bytes="${fields[3]}"; sha="${fields[4]}"; q8="${fields[5]}"
    [[ "$id" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]] || catalog_error "invalid model ID"
    [[ "$q4" =~ ^[[:alnum:]_.][[:alnum:]_.-]*$ && "$q8" =~ ^[[:alnum:]_.][[:alnum:]_.-]*$ \
        && "$q4" != "." && "$q4" != ".." && "$q8" != "." && "$q8" != ".." ]] \
        || catalog_error "invalid filename"
    [[ "$bytes" =~ ^[1-9][0-9]*$ ]] || catalog_error "invalid byte count"
    [[ "$sha" =~ ^[0-9a-f]{64}$ ]] || catalog_error "invalid SHA-256"
    case "$id:$chat" in
        3b-instruct:instruct|3b-reasoning:reasoning|8b-instruct:instruct|8b-reasoning:reasoning|14b-instruct:instruct|14b-reasoning:reasoning) ;;
        *) catalog_error "unknown model ID or chat value" ;;
    esac
    [[ ! -v 'seen_ids[$id]' && ! -v 'seen_q4[$q4]' && ! -v 'seen_hashes[$sha]' \
        && ! -v 'seen_q8[$q8]' ]] || catalog_error "duplicate ID, file, or hash"
    seen_ids[$id]=1; seen_q4[$q4]=1; seen_hashes[$sha]=1; seen_q8[$q8]=1
    ids+=("$id"); chats+=("$chat"); q4_files+=("$q4")
    byte_counts+=("$bytes"); hashes+=("$sha"); q8_files+=("$q8")
done <"$catalog"
(( ${#ids[@]} == 6 )) || catalog_error "expected exactly six rows"
command -v sha256sum >/dev/null 2>&1 || usage_error "sha256sum is required"
command -v stat >/dev/null 2>&1 || usage_error "stat is required"

verified=0
external=0
for index in "${!ids[@]}"; do
    id="${ids[$index]}"; model="$models_dir/${q4_files[$index]}"
    if [[ ! -r "$model" ]]; then
        printf '%s: external verification: artifact is missing or unreadable\n' "$id"
        ((external += 1))
        continue
    fi
    size="$(stat -c %s -- "$model")" || real_failure "$id byte count failed"
    if [[ "$size" != "${byte_counts[$index]}" ]]; then
        printf '%s: external verification: byte count mismatch (expected=%s actual=%s)\n' \
            "$id" "${byte_counts[$index]}" "$size"
        ((external += 1))
        continue
    fi
    digest="$(sha256sum -- "$model")" || real_failure "$id SHA-256 failed"
    digest="${digest%% *}"
    if [[ "$digest" != "${hashes[$index]}" ]]; then
        printf '%s: external verification: SHA-256 mismatch\n' "$id"
        ((external += 1))
        continue
    fi
    output="$(cd "$project_dir" && cargo run --quiet --locked -p gh_zero_engine \
        --no-default-features --features cpu --example inspect -- "$model")" \
        || real_failure "$id inspector failed"
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
    printf '%s: pass\n' "$id"
    ((verified += 1))
done

printf 'summary: pass=%d external_verification=%d total=6\n' "$verified" "$external"
