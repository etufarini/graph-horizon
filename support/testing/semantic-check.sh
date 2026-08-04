#!/usr/bin/env bash
#
# Preserves three Instruct results and classifies three authenticated Reasoning
# runs. It owns synchronous test/temp output only and never retries, falls back,
# changes thresholds, starts an oracle, or owns a production runtime.

set -euo pipefail
export LC_ALL=C

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
catalog="$project_dir/support/models.tsv"
source "$project_dir/support/artifact.sh"
models_dir=""; qualified=0; not_qualified=0; external_verification=0
case_ids="S01 S02 S03 S04 S06 S07 S08 S09 S10"

usage() { echo "usage: semantic-check.sh --models-dir DIR"; }
die() { printf 'semantic-check: %s\n' "$*" >&2; exit 2; }
catalog_die() { printf 'semantic-check: catalog error: %s\n' "$*" >&2; exit 2; }
summary() {
    printf 'summary: qualified=%d not_qualified=%d external_verification=%d total=6\n' \
        "$qualified" "$not_qualified" "$external_verification"
}
row() {
    printf 'qualification: model_id=%s profile=%s evidence=%s status=%s reason=%s critical=%s semantic=%s\n' \
        "$1" "$2" "$3" "$4" "$5" "$6" "$7"
    case "$4" in
        qualified) ((qualified += 1)) ;;
        not-qualified) ((not_qualified += 1)) ;;
        external-verification) ((external_verification += 1)) ;;
    esac
}
reasoning_external() { row "$1" reasoning current external-verification "$2" not-applicable not-applicable; }
reasoning_result() { row "$1" reasoning current not-qualified "$2" "$3" "$4"; }

while (($#)); do
    case "$1" in
        --models-dir) (($# >= 2)) || die "missing --models-dir value"; models_dir="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done
[[ -n "$models_dir" ]] || die "--models-dir is required"
[[ -r "$catalog" ]] || catalog_die "file is missing or unreadable"

set +e
catalog_rows="$(awk -F '\t' '
BEGIN {
    valid["3b-instruct"]="instruct"; valid["3b-reasoning"]="reasoning"
    valid["8b-instruct"]="instruct"; valid["8b-reasoning"]="reasoning"
    valid["14b-instruct"]="instruct"; valid["14b-reasoning"]="reasoning"
}
/^#/ { next }
{
    if (index($0, "\r") || NF != 6 || $0 ~ /\t\t/ || $0 ~ /^[[:space:]]|[[:space:]]$/) exit 2
    if (!($1 in valid) || valid[$1] != $2 || $3 !~ /^[A-Za-z0-9_.][A-Za-z0-9_.-]*$/ ||
        $6 !~ /^[A-Za-z0-9_.][A-Za-z0-9_.-]*$/ || $3 == "." || $3 == ".." || $6 == "." || $6 == ".." ||
        $4 !~ /^[1-9][0-9]*$/ || $5 !~ /^[0-9a-f]{64}$/) exit 2
    if (ids[$1]++ || q4[$3]++ || hashes[$5]++ || q8[$6]++) exit 2
    rows++; print
}
END { if (rows != 6) exit 2 }
' "$catalog")"
catalog_status=$?
set -e
((catalog_status == 0)) || catalog_die "invalid row, value, or duplicate"

missing_tool=""
command -v cargo >/dev/null 2>&1 || missing_tool="cargo-unavailable"
[[ -n "$missing_tool" ]] || artifact_size_tool_available || missing_tool="size-tool-unavailable"
[[ -n "$missing_tool" ]] || artifact_hash_tool_available || missing_tool="sha256-tool-unavailable"

attempt_reason() {
    local id="$1" records="$2" status="$3" cfg sel summary_line timing_line metrics
    cfg="semantic-config: model_id=$id context=4096 max_tokens=4096 temperature=0.7 top_p=1 top_k=0 min_p=0 repeat_penalty=1 seed=0 kv=f16"
    [[ "$(grep -c '^semantic-config:' <<<"$records")" == 1 ]] || { reasoning_result "$id" invalid-validation-protocol not-applicable not-applicable; return; }
    [[ "$(grep '^semantic-config:' <<<"$records")" == "$cfg" ]] || { reasoning_result "$id" configuration-mismatch not-applicable not-applicable; return; }
    if grep -qx "semantic-external: model_id=$id reason=no-full-vram-fit" <<<"$records"; then
        reasoning_external "$id" no-full-vram-fit; return
    fi
    [[ "$(grep -c '^semantic-selection:' <<<"$records")" == 1 &&
       "$(grep -c '^semantic-summary:' <<<"$records")" == 1 &&
       "$(grep -c '^semantic-timing:' <<<"$records")" == 1 &&
       "$(grep -c '^semantic-case:' <<<"$records")" == 9 ]] ||
        { reasoning_result "$id" invalid-validation-protocol not-applicable not-applicable; return; }
    sel="$(grep '^semantic-selection:' <<<"$records")"
    if [[ $sel != *"backend=vulkan reason=full-vram-fit probe_mode=all-gpu run_mode=all-gpu cpu_layers=0 "* ]]; then
        reasoning_external "$id" no-full-vram-fit; return
    fi
    summary_line="$(grep '^semantic-summary:' <<<"$records")"
    timing_line="$(grep '^semantic-timing:' <<<"$records")"
    [[ $summary_line =~ ^semantic-summary:\ model_id=$id\ critical=([0-4])/4\ semantic=([0-9])/9\ semantic_status=(pass|fail)\ reasoning_format=([0-9])/9\ reasoning_format_status=(pass|fail)\ execution_status=(pass|fail)$ ]] ||
        { reasoning_result "$id" invalid-validation-protocol not-applicable not-applicable; return; }
    local critical="${BASH_REMATCH[1]}/4" semantic="${BASH_REMATCH[2]}/9" sem_status="${BASH_REMATCH[3]}" marker_total="${BASH_REMATCH[4]}" marker_status="${BASH_REMATCH[5]}" exec_status="${BASH_REMATCH[6]}"
    [[ $timing_line =~ ^semantic-timing:\ model_id=$id\ completed_cases=9\ total_ms=[0-9]+\ prefill_ms=[0-9]+\ decode_ms=[0-9]+$ ]] ||
        { reasoning_result "$id" invalid-validation-protocol not-applicable not-applicable; return; }
    metrics="$(awk -v id="$id" -v ids="$case_ids" '
    BEGIN { split(ids, want, " "); n=split(ids, order, " ") }
    /^semantic-case:/ {
        c++; case_id=substr($3,9); status=substr($4,8); class=substr($6,7)
        stop=substr($7,6); prompt=substr($8,15); completion=substr($9,19); marker=substr($10,15)
        if ($2 != "model_id=" id || case_id != order[c] || class != "semantic" ||
            status !~ /^(pass|fail)$/ || stop !~ /^(eos|max-tokens|context|error)$/ ||
            prompt !~ /^[0-9]+$/ || completion !~ /^[0-9]+$/ ||
            marker !~ /^(complete|absent|invalid)$/ || length($0) > 1024) bad=1
        if (status == "pass" && (NF != 10 || stop != "eos" || marker != "complete")) bad=1
        if (status == "fail" && (NF < 12 || $11 !~ /^reason=[^[:space:]]+$/ || $12 !~ /^excerpt=/)) bad=1
        if (stop == "error" || $11 == "reason=engine-error") engine=1
        if (stop == "context" || stop == "max-tokens" || $11 == "reason=incomplete-generation") incomplete=1
        if (marker != "complete" || $11 == "reason=invalid-reasoning-markers") markers=1
    }
    END { if (bad || c != n) exit 1; printf "%d %d %d", engine, incomplete, markers }
    ' <<<"$records")" || { reasoning_result "$id" invalid-validation-protocol not-applicable not-applicable; return; }
    read -r has_engine has_incomplete has_markers <<<"$metrics"
    if ((status != 0)) || [[ "$exec_status" == fail || "$has_engine" == 1 ]]; then reasoning_result "$id" engine-error "$critical" "$semantic"
    elif [[ "$has_incomplete" == 1 ]]; then reasoning_result "$id" incomplete-generation "$critical" "$semantic"
    elif [[ "$marker_total" != 9 || "$marker_status" == fail || "$has_markers" == 1 ]]; then reasoning_result "$id" invalid-reasoning-markers "$critical" "$semantic"
    elif [[ "$sem_status" == fail || "$critical" != 4/4 || "${semantic%/9}" -lt 8 ]]; then reasoning_result "$id" semantic-gate-miss "$critical" "$semantic"
    else row "$id" reasoning current qualified semantic-gate-pass "$critical" "$semantic"; fi
}

while IFS=$'\t' read -r id profile file byte_count hash _; do
    if [[ "$profile" == instruct ]]; then
        case "$id" in
            3b-instruct) row "$id" instruct preserved qualified plan-05-pass 4/4 8/9 ;;
            8b-instruct) row "$id" instruct preserved qualified plan-05-pass 4/4 8/9 ;;
            14b-instruct) row "$id" instruct preserved qualified plan-05-pass 4/4 9/9 ;;
            *) catalog_die "unexpected instruct id" ;;
        esac
        continue
    fi
    if [[ -n "$missing_tool" ]]; then reasoning_external "$id" "$missing_tool"; continue; fi
    model="$models_dir/$file"
    [[ -r "$model" && -f "$model" ]] || { reasoning_external "$id" artifact-missing-or-unreadable; continue; }
    size="$(artifact_size "$model")" || { reasoning_external "$id" size-unavailable; continue; }
    [[ "$size" == "$byte_count" ]] || { reasoning_external "$id" byte-count-mismatch; continue; }
    digest="$(artifact_sha256 "$model")" || { reasoning_external "$id" sha256-unavailable; continue; }
    [[ "$digest" == "$hash" ]] || { reasoning_external "$id" sha256-mismatch; continue; }
    set +e
    output="$(
        cd "$project_dir"
        GH_ZERO_MODEL="$model" GH_ZERO_MODEL_ID="$id" cargo test --locked -p gh_zero_engine \
            --no-default-features --features vulkan-hybrid --test semantic real_semantic_acceptance \
            -- --ignored --nocapture --exact 2>&1
    )"
    status=$?
    set -e
    records="$(grep -E '^(semantic-config|semantic-selection|semantic-case|semantic-summary|semantic-timing|semantic-external):' <<<"$output" || true)"
    [[ -z "$records" ]] || printf '%s\n' "$records"
    attempt_reason "$id" "$records" "$status"
done <<<"$catalog_rows"
summary
