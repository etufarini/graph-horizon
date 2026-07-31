#!/usr/bin/env bash
#
# Validates one explicit Reasoning model/backend/context/KV row against a pinned
# CPU-only llama-server oracle: prompt IDs are exact and every oracle completion
# ID must be in the local teacher-forced top two. The local greedy IDs are kept.

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model=""
backend=""
context=""
kv=""
reference_server=""
reference_port="18080"
weights_percent="25"
weights_percent_set="false"
server_pid=""
temporary_dir=""

usage_error() {
    printf 'parity-check: %s\n' "$*" >&2
    exit 2
}

external() {
    printf 'external verification: %s\n' "$*"
    exit 0
}

usage() {
    echo "usage: parity-check.sh --model PATH --backend cpu|vulkan|hybrid --context 4096 --kv f16|int8 --reference-server PATH [--reference-port PORT] [--vram-weights-percent 25]"
}

while (($#)); do
    case "$1" in
        --model) (($# >= 2)) || usage_error "missing --model value"; model="$2"; shift 2 ;;
        --backend) (($# >= 2)) || usage_error "missing --backend value"; backend="$2"; shift 2 ;;
        --context) (($# >= 2)) || usage_error "missing --context value"; context="$2"; shift 2 ;;
        --kv) (($# >= 2)) || usage_error "missing --kv value"; kv="$2"; shift 2 ;;
        --reference-server) (($# >= 2)) || usage_error "missing --reference-server value"; reference_server="$2"; shift 2 ;;
        --reference-port) (($# >= 2)) || usage_error "missing --reference-port value"; reference_port="$2"; shift 2 ;;
        --vram-weights-percent) (($# >= 2)) || usage_error "missing --vram-weights-percent value"; weights_percent="$2"; weights_percent_set="true"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) usage_error "unknown argument" ;;
    esac
done

[[ -n "$model" ]] || usage_error "missing --model"
[[ -n "$reference_server" ]] || usage_error "missing --reference-server"
case "$backend" in cpu|vulkan|hybrid) ;; *) usage_error "invalid backend" ;; esac
case "$kv" in f16|int8) ;; *) usage_error "invalid KV scheme" ;; esac
[[ "$context" =~ ^[1-9][0-9]*$ ]] || usage_error "invalid context"
[[ "$context" == "4096" ]] || usage_error "context must be 4096"
[[ "$reference_port" =~ ^[1-9][0-9]*$ ]] || usage_error "invalid reference port"
((reference_port <= 65535)) || usage_error "invalid reference port"
[[ "$weights_percent" =~ ^[0-9]+$ ]] || usage_error "invalid VRAM weights percentage"
if [[ "$backend" == "hybrid" ]]; then
    [[ "$weights_percent" == "25" ]] || usage_error "hybrid VRAM weights percentage must be 25"
elif [[ "$weights_percent_set" == "true" ]]; then
    usage_error "--vram-weights-percent is valid only for hybrid"
fi

for tool in curl jq sha256sum stat; do
    command -v "$tool" >/dev/null 2>&1 || external "$tool unavailable"
done
[[ -r "$model" && -f "$model" ]] || external "model unavailable"
[[ -x "$reference_server" && -f "$reference_server" ]] || external "llama-server unavailable"

model_size="$(stat -c '%s' -- "$model" 2>/dev/null)" || external "artifact metadata unavailable"
case "$model_size" in
    2147021472)
        profile="Q4_K_M"
        expected_hash="7e9516cc01a039bb3e2d41227cdf388849bc1c942c4624c84567b1684cd9c0fc"
        ;;
    3652203168)
        profile="Q8_0"
        expected_hash="3220ac17e246f741f7371e8b0964c399f363a31f9acd2f9a3aacc3bb19fd6466"
        ;;
    *) external "artifact mismatch" ;;
esac
actual_hash="$(sha256sum -- "$model" 2>/dev/null | cut -d ' ' -f 1)" || external "artifact hash unavailable"
[[ "$actual_hash" == "$expected_hash" ]] || external "artifact mismatch"
server_version="$("$reference_server" --version 2>&1)" || external "unsupported llama.cpp revision"
[[ "$server_version" == *"13f2b28b0"* ]] || external "unsupported llama.cpp revision"

if (exec 3<>"/dev/tcp/127.0.0.1/$reference_port") 2>/dev/null; then
    usage_error "reference port is unavailable"
fi

cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    if [[ -n "$temporary_dir" && -d "$temporary_dir" ]]; then
        rm -r -- "$temporary_dir"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
temporary_dir="$(mktemp -d)"

"$reference_server" \
    --host 127.0.0.1 \
    --port "$reference_port" \
    --model "$model" \
    --ctx-size 4096 \
    --n-gpu-layers 0 \
    --offline \
    --jinja \
    --no-warmup \
    >"$temporary_dir/server.log" 2>&1 &
server_pid="$!"

healthy="false"
for _ in {1..120}; do
    kill -0 "$server_pid" 2>/dev/null || external "llama-server startup failed"
    if curl --fail --silent --show-error --max-time 2 \
        "http://127.0.0.1:$reference_port/health" >/dev/null 2>&1; then
        healthy="true"
        break
    fi
    sleep 1
done
[[ "$healthy" == "true" ]] || external "llama-server startup timed out"

post_json() {
    local endpoint="$1"
    local body="$2"
    local output="$3"
    if ! curl --fail --silent --show-error \
        -H 'Content-Type: application/json' \
        --data-binary "$body" \
        "http://127.0.0.1:$reference_port/$endpoint" >"$output"; then
        printf 'parity-check: oracle HTTP request failed\n' >&2
        exit 1
    fi
}

apply_body="$(jq -cn --arg content 'Quanto fa 17 × 19?' \
    '{messages:[{role:"user",content:$content}],add_generation_prompt:true}')"
post_json "apply-template" "$apply_body" "$temporary_dir/template.json"
oracle_prompt="$(jq -er '.prompt | select(type == "string")' \
    "$temporary_dir/template.json")" || {
    printf 'parity-check: malformed apply-template response\n' >&2
    exit 1
}

tokenize_body="$(jq -cn --arg content "$oracle_prompt" \
    '{content:$content,add_special:true,parse_special:true}')"
post_json "tokenize" "$tokenize_body" "$temporary_dir/tokenize.json"
reference_prompt_ids="$(jq -er '
    .tokens
    | select(type == "array" and length > 0)
    | select(all(.[]; type == "number" and . >= 0 and floor == .))
    | map(tostring)
    | join(",")
' "$temporary_dir/tokenize.json")" || {
    printf 'parity-check: malformed tokenize response\n' >&2
    exit 1
}

completion_body="$(jq -cn --argjson prompt "[$reference_prompt_ids]" '{
    prompt:$prompt,n_predict:16,temperature:0,top_k:1,top_p:1,min_p:0,
    repeat_penalty:1,stream:false,return_tokens:true
}')"
post_json "completion" "$completion_body" "$temporary_dir/completion.json"
reference_completion_ids="$(jq -er '
    .tokens
    | select(type == "array" and length == 16)
    | select(all(.[]; type == "number" and . >= 0 and floor == .))
    | map(tostring)
    | join(",")
' "$temporary_dir/completion.json")" || {
    printf 'parity-check: malformed completion response\n' >&2
    exit 1
}

if ! (
    cd "$project_dir"
    GH_ZERO_MODEL="$model" \
    GH_ZERO_CONTEXT="$context" \
    GH_ZERO_KV="$kv" \
    GH_ZERO_VRAM_WEIGHTS_PERCENT="$weights_percent" \
    GH_ZERO_REFERENCE_PROMPT_IDS="$reference_prompt_ids" \
    GH_ZERO_REFERENCE_COMPLETION_IDS="$reference_completion_ids" \
        cargo test --locked -p gh_zero_engine --no-default-features \
            --features "$backend" real_reasoning_parity -- --ignored --nocapture
) >"$temporary_dir/test.log" 2>&1; then
    cat "$temporary_dir/test.log" >&2
    exit 1
fi

if grep -Fq "external verification: Vulkan backend unavailable" "$temporary_dir/test.log"; then
    external "Vulkan backend unavailable"
fi
local_result="$(grep -F 'reasoning-parity:' "$temporary_dir/test.log" | tail -n 1)"
if [[ -z "$local_result" ]]; then
    printf 'parity-check: missing local parity result\n' >&2
    exit 1
fi
printf 'parity-check: pass profile=%s backend=%s kv=%s prompt_ids=%s completion_ids=%s %s\n' \
    "$profile" "$backend" "$kv" "$reference_prompt_ids" "$reference_completion_ids" "$local_result"
