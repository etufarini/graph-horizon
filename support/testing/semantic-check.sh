#!/usr/bin/env bash
#
# Authenticates and runs exactly six CPU/f16 semantic suites without retry or
# network access. External artifact conditions continue; attempted failures stop.

set -euo pipefail
export LC_ALL=C

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
catalog="$project_dir/support/models.tsv"
models_dir=""; pass=0; external=0; failure=0

usage_error() { printf 'semantic-check: %s\n' "$*" >&2; exit 2; }
catalog_error() { printf 'semantic-check: catalog error: %s\n' "$*" >&2; exit 2; }
summary() {
    printf 'summary: pass=%d external_verification=%d failure=%d total=%d\n' \
        "$pass" "$external" "$failure" "$((pass + external + failure))"
}
external_row() {
    printf 'semantic model_id=%s: external verification: %s\n' "$1" "$2"
    ((external += 1))
}
usage() { echo "usage: semantic-check.sh --models-dir DIR"; }

while (($#)); do
    case "$1" in
        --models-dir) (($# >= 2)) || usage_error "missing --models-dir value"; models_dir="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
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

declare -a ids files bytes hashes
while IFS=$'\t' read -r id _ file byte_count hash _; do
    ids+=("$id"); files+=("$file"); bytes+=("$byte_count"); hashes+=("$hash")
done <<<"$catalog_rows"

missing_tool=""
for tool in cargo sha256sum stat; do
    command -v "$tool" >/dev/null 2>&1 || { missing_tool="$tool"; break; }
done
if [[ -n "$missing_tool" ]]; then
    for id in "${ids[@]}"; do external_row "$id" "$missing_tool unavailable"; done
    summary; exit 0
fi

for index in "${!ids[@]}"; do
    id="${ids[$index]}"; model="$models_dir/${files[$index]}"
    if [[ ! -r "$model" || ! -f "$model" ]]; then
        external_row "$id" "artifact is missing or unreadable"; continue
    fi
    size="$(stat -c %s -- "$model" 2>/dev/null)" || { external_row "$id" "byte count unavailable"; continue; }
    [[ "$size" == "${bytes[$index]}" ]] || { external_row "$id" "byte count mismatch"; continue; }
    digest="$(sha256sum -- "$model" 2>/dev/null)" || { external_row "$id" "SHA-256 unavailable"; continue; }
    [[ "${digest%% *}" == "${hashes[$index]}" ]] || { external_row "$id" "SHA-256 mismatch"; continue; }
    set +e
    output="$(
        cd "$project_dir"
        GH_ZERO_MODEL="$model" GH_ZERO_MODEL_ID="$id" \
            cargo test --locked -p gh_zero_engine --no-default-features --features cpu \
                --test semantic real_semantic_acceptance -- --ignored --nocapture --exact 2>&1
    )"
    status=$?
    set -e
    grep -E '^(semantic-case|semantic-summary):' <<<"$output" || true
    if ((status != 0)); then
        printf 'semantic model_id=%s: failure\n' "$id"
        ((failure += 1)); summary; exit 1
    fi
    printf 'semantic model_id=%s: pass\n' "$id"
    ((pass += 1))
done
summary
