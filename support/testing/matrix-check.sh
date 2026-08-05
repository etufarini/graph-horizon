#!/usr/bin/env bash
#
# Runs six Q8 rejections, 60 main parity rows, and eight homogeneous hybrid
# endpoints. It retains successful local ID sequences in memory, compares each
# endpoint with its standalone control, and never substitutes a tuple value.

set -euo pipefail
export LC_ALL=C

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
catalog="$project_dir/support/models.tsv"
parity="$project_dir/support/testing/parity-check.sh"
models_dir=""; reference_server=""; reference_port="18080"
pass=0; external=0; failure=0
declare -A row_status local_ids

usage_error() { printf 'matrix-check: %s\n' "$*" >&2; exit 2; }
catalog_error() { printf 'matrix-check: catalog error: %s\n' "$*" >&2; exit 2; }
summary() { printf 'summary: pass=%d external_verification=%d failure=%d total=%d\n' "$pass" "$external" "$failure" "$((pass + external + failure))"; }
stop_failure() { printf '%s: failure: %s\n' "$1" "$2"; ((failure += 1)); summary; exit 1; }
usage() { echo "usage: matrix-check.sh --models-dir DIR --reference-server PATH [--reference-port PORT]"; }

while (($#)); do
    case "$1" in
        --models-dir) (($# >= 2)) || usage_error "missing --models-dir value"; models_dir="$2"; shift 2 ;;
        --reference-server) (($# >= 2)) || usage_error "missing --reference-server value"; reference_server="$2"; shift 2 ;;
        --reference-port) (($# >= 2)) || usage_error "missing --reference-port value"; reference_port="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) usage_error "unknown argument: $1" ;;
    esac
done
[[ -n "$models_dir" && -n "$reference_server" ]] || usage_error "required argument is missing"
[[ "$reference_port" =~ ^[1-9][0-9]{0,4}$ ]] && ((reference_port <= 65535)) || usage_error "invalid reference port"
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
[[ -x "$parity" ]] || usage_error "parity runner is missing or not executable"
command -v cargo >/dev/null 2>&1 || usage_error "cargo is required"
if (exec 3<>"/dev/tcp/127.0.0.1/$reference_port") 2>/dev/null; then usage_error "reference port is occupied"; fi

declare -a model_ids q8_files
while IFS=$'\t' read -r id _ _ _ _ q8_file; do model_ids+=("$id"); q8_files+=("$q8_file"); done <<<"$catalog_rows"

expected_q8="E04 unsupported GGUF weight profile 'Q8_0'; supported profile: Q4_K_M"
for index in "${!model_ids[@]}"; do
    id="${model_ids[$index]}"; key="q8 model_id=$id"; model="$models_dir/${q8_files[$index]}"
    if [[ ! -r "$model" || ! -f "$model" ]]; then
        printf '%s: external verification: artifact is missing or unreadable\n' "$key"; ((external += 1)); continue
    fi
    set +e
    output="$(cd "$project_dir" && cargo run --quiet --locked -p graph_horizon_engine --no-default-features --features cpu --example inspect -- "$model" 2>&1)"
    status=$?
    set -e
    if ((status == 1)) && [[ "$output" == "$expected_q8" ]]; then printf '%s: pass\n' "$key"; ((pass += 1)); continue; fi
    stop_failure "$key" "Q8 rejection contract mismatch"
done

run_parity() {
    local id="$1" backend="$2" kv="$3" percent="${4:-}" mode="${5:-}" control="${6:-}"
    local key="parity model_id=$id backend=$backend kv=$kv" row_key output status ids control_key
    row_key="$id:$backend:$kv:$percent:$mode"
    local arguments=(--models-dir "$models_dir" --model-id "$id" --backend "$backend" --kv "$kv"
        --reference-server "$reference_server" --reference-port "$reference_port")
    if [[ -n "$percent" ]]; then
        arguments+=(--weights-percent "$percent" --expect-mode "$mode")
        key="$key weights_percent=$percent mode=$mode"
    fi
    set +e
    output="$("$parity" "${arguments[@]}" 2>&1)"; status=$?
    set -e
    if ((status == 0)) && [[ "$output" == pass:* ]]; then
        if [[ "$output" =~ (^|[[:space:]])local_ids=([0-9]+(,[0-9]+){15})($|[[:space:]]) ]]; then
            ids="${BASH_REMATCH[2]}"
        else
            stop_failure "$key" "malformed local ID sequence"
        fi
        if [[ -n "$control" ]]; then
            control_key="$id:$control:$kv::"
            if [[ "${row_status[$control_key]:-}" == pass && "${local_ids[$control_key]}" != "$ids" ]]; then
                stop_failure "$key" "homogeneous endpoint local ID mismatch"
            fi
        fi
        row_status["$row_key"]="pass"; local_ids["$row_key"]="$ids"
        printf '%s: %s\n' "$key" "$output"; ((pass += 1)); return
    fi
    if ((status == 0)) && [[ "$output" == "external verification: "* ]]; then
        row_status["$row_key"]="external"
        printf '%s: %s\n' "$key" "$output"; ((external += 1)); return
    fi
    stop_failure "$key" "parity row failed"
}

for id in "${model_ids[@]}"; do
    for backend in cpu vulkan vulkan-hybrid metal metal-hybrid; do
        for kv in f16 int8; do
            case "$backend" in
                vulkan-hybrid|metal-hybrid) run_parity "$id" "$backend" "$kv" 25 mixed ;;
                *) run_parity "$id" "$backend" "$kv" ;;
            esac
        done
    done
done
for endpoint in vulkan-hybrid:vulkan:all-gpu metal-hybrid:metal:all-metal; do
    IFS=: read -r backend control all_mode <<<"$endpoint"
    for mode_percent in "$all_mode:100" cpu-only:0; do
        mode="${mode_percent%:*}"; percent="${mode_percent#*:}"
        if [[ "$mode" == cpu-only ]]; then control=cpu; fi
        for kv in f16 int8; do
            run_parity 3b-instruct "$backend" "$kv" "$percent" "$mode" "$control"
        done
    done
done
summary
