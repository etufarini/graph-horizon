#!/usr/bin/env bash
#
# Authenticates exactly six read-only models and runs one hybrid placement probe
# per row. Semantic generation is all-GPU or CPU-only, never mixed; external
# conditions and attempted failures continue to the aggregate exit status.

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
protocol_valid() {
    local id="$1" records="$2" selection report timing backend number='[0-9]+'
    [[ "$(grep -c '^semantic-selection:' <<<"$records")" == 1 ]] || return 1
    [[ "$(grep -c '^semantic-summary:' <<<"$records")" == 1 ]] || return 1
    [[ "$(grep -c '^semantic-timing:' <<<"$records")" == 1 ]] || return 1
    [[ "$(grep -c '^semantic-case:' <<<"$records")" == 12 ]] || return 1
    ! grep -q '^semantic-external:' <<<"$records" || return 1
    selection="$(grep '^semantic-selection:' <<<"$records")"
    report="$(grep '^semantic-summary:' <<<"$records")"
    timing="$(grep '^semantic-timing:' <<<"$records")"
    [[ $selection =~ ^semantic-selection:\ model_id=$id\ backend=(vulkan|cpu)\ reason=(full-vram-fit|no-full-vram-fit)\ probe_mode=(all-gpu|mixed|cpu-only)\ run_mode=(all-gpu|cpu-only)\ cpu_layers=$number\ gpu_layers=$number\ cpu_weights=$number\ cpu_kv=$number\ cpu_scratch=$number\ cpu_fixed=$number\ cpu_staging=$number\ cpu_crossing=$number\ cpu_reserve=$number\ cpu_total=$number\ gpu_weights=$number\ gpu_kv=$number\ gpu_scratch=$number\ gpu_fixed=$number\ gpu_staging=$number\ gpu_crossing=$number\ gpu_reserve=$number\ gpu_total=$number$ ]] || return 1
    backend="${BASH_REMATCH[1]}"
    if [[ $backend == vulkan ]]; then
        [[ $selection =~ backend=vulkan\ reason=full-vram-fit\ probe_mode=all-gpu\ run_mode=all-gpu\ cpu_layers=0\ gpu_layers=$number ]] || return 1
    else
        [[ $selection =~ backend=cpu\ reason=no-full-vram-fit\ probe_mode=(mixed|cpu-only)\ run_mode=cpu-only\ cpu_layers=$number\ gpu_layers=0 ]] || return 1
    fi
    [[ $report =~ ^semantic-summary:\ model_id=$id\ backend=$backend\ critical=4/4\ semantic=[89]/9\ semantic_status=pass\ conformance=[0-3]/3\ conformance_status=diagnostic$ ]] || return 1
    if [[ $id == 3b-instruct && $backend == vulkan ]]; then
        [[ $timing =~ ^semantic-timing:\ model_id=$id\ backend=vulkan\ completed_cases=12\ total_ms=$number\ prefill_ms=$number\ decode_ms=$number\ baseline_cpu_ms=1506690\ performance_status=pass$ ]] || return 1
    else
        [[ $timing =~ ^semantic-timing:\ model_id=$id\ backend=$backend\ completed_cases=12\ total_ms=$number\ prefill_ms=$number\ decode_ms=$number\ baseline_cpu_ms=not-applicable\ performance_status=not-applicable$ ]] || return 1
    fi
    [[ $records != *"$models_dir/"* ]] || return 1
    for case_id in S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 S11 S12; do
        [[ "$(grep -c "^semantic-case: model_id=$id case_id=$case_id " <<<"$records")" == 1 ]] || return 1
    done
}

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
            cargo test --locked -p gh_zero_engine --no-default-features --features hybrid \
                --test semantic real_semantic_acceptance -- --ignored --nocapture --exact 2>&1
    )"
    status=$?
    set -e
    records="$(grep -E '^(semantic-selection|semantic-case|semantic-summary|semantic-timing|semantic-external):' <<<"$output" || true)"
    [[ -z "$records" ]] || printf '%s\n' "$records"
    if ((status != 0)); then
        printf 'semantic model_id=%s: failure\n' "$id"
        ((failure += 1)); continue
    fi
    if [[ "$records" == "semantic-external: model_id=$id reason=insufficient RAM" ]]; then
        external_row "$id" "insufficient RAM"; continue
    fi
    if ! protocol_valid "$id" "$records"; then
        printf 'semantic model_id=%s: failure\n' "$id"
        ((failure += 1)); continue
    fi
    printf 'semantic model_id=%s: pass\n' "$id"
    ((pass += 1))
done
summary
if ((failure > 0)); then
    exit 1
fi
