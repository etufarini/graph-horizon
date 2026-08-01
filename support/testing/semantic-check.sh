#!/usr/bin/env bash
#
# Authenticates exactly six read-only models, invokes each semantic test once,
# and strictly validates its all-GPU or CPU-only protocol. External conditions
# and attempted failures continue to the aggregate exit status.

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
    local id="$1" chat="$2" records="$3" selection timing backend number='[0-9]+'
    [[ "$(grep -c '^semantic-selection:' <<<"$records")" == 1 ]] || return 1
    [[ "$(grep -c '^semantic-summary:' <<<"$records")" == 1 ]] || return 1
    [[ "$(grep -c '^semantic-timing:' <<<"$records")" == 1 ]] || return 1
    [[ "$(grep -c '^semantic-case:' <<<"$records")" == 12 ]] || return 1
    ! grep -q '^semantic-external:' <<<"$records" || return 1
    selection="$(grep '^semantic-selection:' <<<"$records")"
    timing="$(grep '^semantic-timing:' <<<"$records")"
    [[ $selection =~ ^semantic-selection:\ model_id=$id\ backend=(vulkan|cpu)\ reason=(full-vram-fit|no-full-vram-fit)\ probe_mode=(all-gpu|mixed|cpu-only)\ run_mode=(all-gpu|cpu-only)\ cpu_layers=$number\ gpu_layers=$number\ cpu_weights=$number\ cpu_kv=$number\ cpu_scratch=$number\ cpu_fixed=$number\ cpu_staging=$number\ cpu_crossing=$number\ cpu_reserve=$number\ cpu_total=$number\ gpu_weights=$number\ gpu_kv=$number\ gpu_scratch=$number\ gpu_fixed=$number\ gpu_staging=$number\ gpu_crossing=$number\ gpu_reserve=$number\ gpu_total=$number$ ]] || return 1
    backend="${BASH_REMATCH[1]}"
    if [[ $backend == vulkan ]]; then
        [[ $selection =~ backend=vulkan\ reason=full-vram-fit\ probe_mode=all-gpu\ run_mode=all-gpu\ cpu_layers=0\ gpu_layers=$number ]] || return 1
    else
        [[ $selection =~ backend=cpu\ reason=no-full-vram-fit\ probe_mode=(mixed|cpu-only)\ run_mode=cpu-only\ cpu_layers=$number\ gpu_layers=0 ]] || return 1
    fi
    if [[ $id == 3b-instruct && $backend == vulkan ]]; then
        [[ $timing =~ ^semantic-timing:\ model_id=$id\ backend=vulkan\ completed_cases=12\ total_ms=$number\ prefill_ms=$number\ decode_ms=$number\ baseline_cpu_ms=1506690\ performance_status=pass$ ]] || return 1
    else
        [[ $timing =~ ^semantic-timing:\ model_id=$id\ backend=$backend\ completed_cases=12\ total_ms=$number\ prefill_ms=$number\ decode_ms=$number\ baseline_cpu_ms=not-applicable\ performance_status=not-applicable$ ]] || return 1
    fi
    [[ $records != *"$models_dir/"* ]] || return 1
    awk -v id="$id" -v chat="$chat" -v backend="$backend" '
    BEGIN {
        for (i=1; i<=12; i++) class[sprintf("S%02d", i)]="semantic"
        class["S05"]="conformance"; class["S11"]="conformance"; class["S12"]="conformance"
        critical["S01"]=1; critical["S02"]=1; critical["S03"]=1; critical["S06"]=1
    }
    $1 == "semantic-case:" {
        case_id=substr($3, 9); status=substr($4, 8); actual_class=substr($6, 7)
        stop=substr($7, 6); prompt=substr($8, 15); completion=substr($9, 19)
        marker=substr($10, 15); cases++
        if ($2 != "model_id=" id || !(case_id in class) || seen[case_id]++ ||
            $5 !~ /^predicate=[A-Za-z0-9-]+$/ || actual_class != class[case_id] ||
            stop !~ /^(eos|max-tokens|context|error)$/ || prompt !~ /^[0-9]+$/ ||
            completion !~ /^[0-9]+$/ || stop != "eos" || length($0) > 1024) bad=1
        if (status == "pass") {
            if (NF != 10 || marker == "invalid") bad=1
            if (actual_class == "semantic") semantic++
            else conformance++
            if (case_id in critical) critical_pass++
        } else if (status != "fail" || NF < 12 || $11 !~ /^reason=[^[:space:]]+$/ || $12 !~ /^excerpt=/) bad=1
        if (chat == "instruct") {
            if (marker != "not-applicable") bad=1
        } else {
            if (marker !~ /^(complete|absent|invalid)$/) bad=1
            if (marker == "complete") complete++
        }
    }
    $1 == "semantic-summary:" { report=$0; summaries++ }
    END {
        format=(chat == "instruct" ? "not-applicable reasoning_format_status=not-applicable" : complete "/12 reasoning_format_status=diagnostic")
        expected="semantic-summary: model_id=" id " backend=" backend " critical=" critical_pass "/4 semantic=" semantic "/9 semantic_status=pass conformance=" conformance "/3 conformance_status=diagnostic reasoning_format=" format
        if (cases != 12 || summaries != 1 || critical_pass != 4 || semantic < 8 || report != expected) bad=1
        exit bad
    }' <<<"$records"
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

declare -a ids chats files bytes hashes
while IFS=$'\t' read -r id chat file byte_count hash _; do
    ids+=("$id"); chats+=("$chat"); files+=("$file"); bytes+=("$byte_count"); hashes+=("$hash")
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
    if ! protocol_valid "$id" "${chats[$index]}" "$records"; then
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
