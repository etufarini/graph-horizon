#!/usr/bin/env bash
#
# Validates one exact artifact/profile/KV/placement row against the pinned CPU
# oracle. It owns one loopback oracle PID and temp directory, accepts only the
# declared tuple, and never retries another artifact, profile, context, or split.

set -euo pipefail
export LC_ALL=C

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
catalog="$project_dir/support/models.tsv"
source "$project_dir/support/artifact.sh"
models_dir=""; model_id=""; backend=""; kv=""; reference_server=""
reference_port="18080"; weights_percent=""; expected_mode=""
server_pid=""; temporary_dir=""
oracle_revision="13f2b28b098623391b1aacfd27995e1c8b7de9a9"

usage_error() { printf 'parity-check: %s\n' "$*" >&2; exit 2; }
catalog_error() { printf 'parity-check: catalog error: %s\n' "$*" >&2; exit 2; }
external() { printf 'external verification: %s\n' "$*"; exit 0; }
real_failure() { printf 'parity-check: %s\n' "$*" >&2; exit 1; }
usage() {
    echo "usage: parity-check.sh --models-dir DIR --model-id ID --backend cpu|vulcan|vulcan-hybrid|metal|metal-hybrid --kv f16|int8 --reference-server PATH [--reference-port PORT] [--weights-percent 0..100 --expect-mode all-gpu|mixed|cpu-only]"
}

while (($#)); do
    case "$1" in
        --models-dir) (($# >= 2)) || usage_error "missing --models-dir value"; models_dir="$2"; shift 2 ;;
        --model-id) (($# >= 2)) || usage_error "missing --model-id value"; model_id="$2"; shift 2 ;;
        --backend) (($# >= 2)) || usage_error "missing --backend value"; backend="$2"; shift 2 ;;
        --kv) (($# >= 2)) || usage_error "missing --kv value"; kv="$2"; shift 2 ;;
        --reference-server) (($# >= 2)) || usage_error "missing --reference-server value"; reference_server="$2"; shift 2 ;;
        --reference-port) (($# >= 2)) || usage_error "missing --reference-port value"; reference_port="$2"; shift 2 ;;
        --weights-percent) (($# >= 2)) || usage_error "missing --weights-percent value"; weights_percent="$2"; shift 2 ;;
        --expect-mode) (($# >= 2)) || usage_error "missing --expect-mode value"; expected_mode="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) usage_error "unknown argument: $1" ;;
    esac
done

[[ -n "$models_dir" && -n "$model_id" && -n "$backend" && -n "$kv" && -n "$reference_server" ]] \
    || usage_error "required argument is missing"
case "$backend" in cpu|vulcan|vulcan-hybrid|metal|metal-hybrid) ;; *) usage_error "invalid backend" ;; esac
case "$kv" in f16|int8) ;; *) usage_error "invalid KV scheme" ;; esac
[[ "$reference_port" =~ ^[1-9][0-9]{0,4}$ ]] && ((reference_port <= 65535)) \
    || usage_error "invalid reference port"
case "$backend" in
    vulcan-hybrid|metal-hybrid)
        [[ "$weights_percent" =~ ^[0-9]+$ ]] && ((weights_percent <= 100)) \
            || usage_error "hybrid backend requires --weights-percent 0..100"
        case "$expected_mode" in all-gpu|mixed|cpu-only) ;; *) usage_error "hybrid backend requires --expect-mode" ;; esac
        ;;
    *) [[ -z "$weights_percent" && -z "$expected_mode" ]] || usage_error "placement arguments require a hybrid backend" ;;
esac
[[ -r "$catalog" ]] || catalog_error "file is missing or unreadable"

set +e
row="$(awk -F '\t' -v wanted="$model_id" '
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
    rows++; if ($1 == wanted) selected=$0
}
END { if (!bad && rows != 6) exit 2; if (!bad && selected == "") exit 3; if (!bad) print selected }
' "$catalog")"
catalog_status=$?
set -e
((catalog_status == 0)) || {
    ((catalog_status == 3)) && usage_error "unknown model ID"
    catalog_error "invalid row, value, or duplicate"
}
IFS=$'\t' read -r _ _ q4_file expected_bytes expected_hash _ <<<"$row"
model="$models_dir/$q4_file"

for tool in curl jq; do command -v "$tool" >/dev/null 2>&1 || external "$tool unavailable"; done
artifact_size_tool_available || external "size tool unavailable"
artifact_hash_tool_available || external "SHA-256 tool unavailable"
[[ -r "$model" && -f "$model" ]] || external "$model_id artifact is missing or unreadable"
size="$(artifact_size "$model")" || external "$model_id byte count unavailable"
[[ "$size" == "$expected_bytes" ]] || real_failure "$model_id byte count mismatch"
digest="$(artifact_sha256 "$model")" || external "$model_id SHA-256 unavailable"
[[ "$digest" == "$expected_hash" ]] || real_failure "$model_id SHA-256 mismatch"
[[ -x "$reference_server" && -f "$reference_server" ]] || external "llama-server is missing or not executable"
server_version="$("$reference_server" --version 2>&1)" || external "llama.cpp revision unavailable"
[[ "$server_version" == *"${oracle_revision:0:9}"* ]] || external "unsupported llama.cpp revision"
if (exec 3<>"/dev/tcp/127.0.0.1/$reference_port") 2>/dev/null; then usage_error "reference port is occupied"; fi

cleanup() {
    if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi
    [[ -z "$temporary_dir" || ! -d "$temporary_dir" ]] || rm -r -- "$temporary_dir"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
temporary_dir="$(mktemp -d)"
"$reference_server" --host 127.0.0.1 --port "$reference_port" --model "$model" \
    --ctx-size 4096 --device none --n-gpu-layers 0 --no-kv-offload --offline \
    --jinja --no-warmup --ignore-eos >"$temporary_dir/server.log" 2>&1 &
server_pid="$!"

healthy=false
for _ in {1..120}; do
    kill -0 "$server_pid" 2>/dev/null || external "llama-server startup failed"
    if curl --fail --silent --show-error --max-time 2 "http://127.0.0.1:$reference_port/health" >/dev/null 2>&1; then
        healthy=true; break
    fi
    sleep 1
done
[[ "$healthy" == true ]] || external "llama-server health timeout"

post_json() {
    local endpoint="$1" body="$2" output="$3"
    curl --fail --silent --show-error -H 'Content-Type: application/json' --data-binary "$body" \
        "http://127.0.0.1:$reference_port/$endpoint" >"$output" || real_failure "oracle HTTP request failed"
}
body="$(jq -cn --arg content 'Quanto fa 17 × 19?' '{messages:[{role:"system",content:""},{role:"user",content:$content}],add_generation_prompt:true}')"
post_json apply-template "$body" "$temporary_dir/template.json"
oracle_prompt="$(jq -er '.prompt | select(type == "string")' "$temporary_dir/template.json")" \
    || real_failure "malformed apply-template response"
body="$(jq -cn --arg content "$oracle_prompt" '{content:$content,add_special:true,parse_special:true}')"
post_json tokenize "$body" "$temporary_dir/tokenize.json"
prompt_ids="$(jq -er '.tokens | select(type == "array" and length > 0) | select(all(.[]; type == "number" and . >= 0 and floor == .)) | map(tostring) | join(",")' "$temporary_dir/tokenize.json")" \
    || real_failure "malformed tokenize response"
body="$(jq -cn --argjson prompt "[$prompt_ids]" '{prompt:$prompt,n_predict:16,temperature:0,top_k:1,top_p:1,min_p:0,repeat_penalty:1,stream:false,return_tokens:true}')"
post_json completion "$body" "$temporary_dir/completion.json"
completion_ids="$(jq -er '.tokens | select(type == "array" and length == 16) | select(all(.[]; type == "number" and . >= 0 and floor == .)) | map(tostring) | join(",")' "$temporary_dir/completion.json")" \
    || real_failure "malformed completion response"

environment=(GH_ZERO_MODEL="$model" GH_ZERO_CONTEXT=4096 GH_ZERO_KV="$kv"
    GH_ZERO_REFERENCE_PROMPT_IDS="$prompt_ids" GH_ZERO_REFERENCE_COMPLETION_IDS="$completion_ids")
if [[ -n "$weights_percent" ]]; then
    environment+=(GH_ZERO_VRAM_WEIGHTS_PERCENT="$weights_percent" GH_ZERO_EXPECTED_MODE="$expected_mode")
fi
set +e
(cd "$project_dir" && env "${environment[@]}" cargo test --locked --release -p gh_zero_engine \
    --no-default-features --features "$backend" --test family_agnostic \
    real_selected_runtime_parity_and_lifecycle -- --ignored --nocapture --exact) \
    >"$temporary_dir/test.log" 2>&1
test_status=$?
set -e
if grep -Eq '(external verification: )?(Vulkan|Metal) backend (is )?unavailable' "$temporary_dir/test.log"; then external "$backend backend unavailable"; fi
if ((test_status != 0)); then
    if grep -Eq '(Vulkan|Metal) memory is insufficient|context 4096 does not fit|model does not fit available RAM and VRAM|mixed placement required' "$temporary_dir/test.log"; then
        external "insufficient memory for $backend row"
    fi
    real_failure "local parity assertion failed"
fi
result="$(grep -F 'ministral-parity:' "$temporary_dir/test.log" | tail -n 1)"
[[ -n "$result" ]] || real_failure "missing local parity result"
local_ids="${result#*local_ids=}"; local_ids="${local_ids%% *}"
printf 'pass: model_id=%s backend=%s kv=%s prompt_ids=%s oracle_ids=%s local_ids=%s\n' \
    "$model_id" "$backend" "$kv" "$prompt_ids" "$completion_ids" "$local_ids"
