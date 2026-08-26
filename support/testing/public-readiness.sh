#!/usr/bin/env bash
# Authenticates one catalog model, tests installed backends, and renders a bounded report.

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; catalog="$project_dir/support/models.tsv"
model_id=""; model=""; benchmark_backend=""; report=""; id_seen=false
model_seen=false; backend_seen=false; report_seen=false
temp_root=""; server_pid=""

usage() { echo "usage: public-readiness.sh --model-id ID --model /absolute/model.gguf --benchmark-backend cpu|vulkan|vulkan-hybrid|metal|metal-hybrid --report /absolute/report.md"; }
fail() { printf 'public-readiness: %s\n' "$1" >&2; exit "${2:-1}"; }

stop_server() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; server_pid=""
    fi
}

cleanup() {
    stop_server
    # The owned root is accepted only in this exact mktemp namespace.
    [[ "$temp_root" =~ ^/tmp/graph-horizon-readiness\.[A-Za-z0-9]+$ && -d "$temp_root" ]] && rm -rf -- "$temp_root" || true
}

trap cleanup EXIT; trap 'exit 130' HUP INT TERM

while (($#)); do
    case "$1" in
        --model-id) (($# >= 2)) || fail "missing --model-id value" 2; $id_seen && fail "duplicate --model-id" 2; model_id="$2"; id_seen=true; shift 2 ;;
        --model) (($# >= 2)) || fail "missing --model value" 2; $model_seen && fail "duplicate --model" 2; model="$2"; model_seen=true; shift 2 ;;
        --benchmark-backend) (($# >= 2)) || fail "missing --benchmark-backend value" 2; $backend_seen && fail "duplicate --benchmark-backend" 2; benchmark_backend="$2"; backend_seen=true; shift 2 ;;
        --report) (($# >= 2)) || fail "missing --report value" 2; $report_seen && fail "duplicate --report" 2; report="$2"; report_seen=true; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) fail "invalid arguments" 2 ;;
    esac
done
$id_seen && $model_seen && $backend_seen && $report_seen || fail "required arguments are missing" 2
[[ "$model_id" =~ ^[a-z0-9][a-z0-9-]{0,63}$ ]] || fail "invalid model ID" 2
case "$benchmark_backend" in cpu|vulkan|vulkan-hybrid|metal|metal-hybrid) ;; *) fail "invalid benchmark backend" 2 ;; esac
for path in "$model" "$report"; do
    [[ "$path" == /* && "$path" != *[$'\001'-$'\037'$'\177']* ]] || fail "invalid path" 2
    case "$path/" in */./*|*/../*) fail "invalid path" 2 ;; esac
done
[[ -f "$model" && -r "$model" ]] || fail "model is missing or unreadable" 2
report_parent="${report%/*}"; [[ -n "${report##*/}" && -d "$report_parent" && -w "$report_parent" && ! -L "$report" ]] || fail "invalid report destination" 2

for tool in awk bash cargo cat curl find git grep install lscpu ln mkdir mktemp npm rm rustc script sed sleep stat stty timeout uname wc; do command -v "$tool" >/dev/null 2>&1 || fail "required tool is unavailable"; done
source "$project_dir/support/artifact.sh"
row="$(awk -F '\t' -v id="$model_id" '$1 == id { print; found++ } END { if (found != 1) exit 1 }' "$catalog")" \
    || fail "model catalog is invalid"
IFS=$'\t' read -r selected_id _ model_name expected_bytes expected_hash _ <<<"$row"
[[ "$selected_id" == "$model_id" && "${model##*/}" == "$model_name" ]] || fail "model is not authenticated"
actual_bytes="$(artifact_size "$model")" || fail "model authentication failed"; actual_hash="$(artifact_sha256 "$model")" || fail "model authentication failed"
[[ "$actual_bytes" == "$expected_bytes" && "$actual_hash" == "$expected_hash" ]] || fail "model is not authenticated"

commit="$(git -C "$project_dir" rev-parse --verify 'HEAD^{commit}' 2>/dev/null)" || fail "Git commit cannot be resolved"; [[ "$commit" =~ ^[0-9a-f]{40,64}$ ]] || fail "Git commit is invalid"
os="$(uname -s)"; arch="$(uname -m)"; kernel="$(uname -r)"
case "$os/$arch" in Linux/x86_64|Darwin/arm64) ;; *) fail "unsupported platform" ;; esac
cpu="$(LC_ALL=C lscpu | awk -F: '$1 ~ /^Model name/ { sub(/^[ \t]+/, "", $2); print $2; exit }')"; [[ -n "$cpu" && "$cpu" != *['`|'$'\r\n']* ]] || fail "CPU identity is unavailable"

vulkan_status="external verification"; vulkan_hybrid_status="external verification"; metal_status="external verification"; metal_hybrid_status="external verification"
available=(cpu); gpu="external verification"; driver="external verification"; vulkan_runtime="external verification"
temp_root="$(mktemp -d /tmp/graph-horizon-readiness.XXXXXX)" || fail "temporary state cannot be created"
[[ "$temp_root" =~ ^/tmp/graph-horizon-readiness\.[A-Za-z0-9]+$ && -d "$temp_root" ]] || fail "temporary state is invalid"
vulkan_file="$temp_root/vulkan"
if command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo --summary >"$vulkan_file" 2>/dev/null; then
    device="$(awk -F= '
        /^[[:space:]]*GPU[0-9]+:/ { name=""; type=""; api=""; drv=""; info="" }
        /deviceName[[:space:]]*=/ { name=$2; sub(/^[ \t]+/, "", name) }
        /deviceType[[:space:]]*=/ { type=$2 }
        /apiVersion[[:space:]]*=/ { api=$2; sub(/^[ \t]+/, "", api) }
        /driverName[[:space:]]*=/ { drv=$2; sub(/^[ \t]+/, "", drv) }
        /driverInfo[[:space:]]*=/ { info=$2; sub(/^[ \t]+/, "", info) }
        /conformanceVersion[[:space:]]*=/ && name != "" {
            low=tolower(name " " drv); if (type !~ /CPU/ && low !~ /llvmpipe|lavapipe|software/) { print name "\t" drv " " info "\t" api; exit }
        }' "$vulkan_file")"
    if [[ -n "$device" ]]; then
        IFS=$'\t' read -r gpu driver device_api <<<"$device"
        vulkan_runtime="$(awk -F: '/Vulkan Instance Version/ { sub(/^[ \t]+/, "", $2); print $2; exit }' "$vulkan_file"); device API $device_api"
        available+=(vulkan vulkan-hybrid)
    fi
fi
rm -f -- "$vulkan_file"
if [[ "$os/$arch" == Darwin/arm64 ]] && command -v xcrun >/dev/null 2>&1 \
    && xcrun -f metal >/dev/null 2>&1 && xcrun -f metallib >/dev/null 2>&1; then
    available+=(metal metal-hybrid)
fi
for value in "$gpu" "$driver" "$vulkan_runtime" "$kernel"; do
    [[ "$value" != *['`|'$'\r\n']* && "$value" != */* ]] || fail "unsafe system identity"
done
selected=false
for backend in "${available[@]}"; do [[ "$backend" == "$benchmark_backend" ]] && selected=true; done
$selected || fail "selected benchmark backend requires external verification"

clone="$temp_root/source"; state="$temp_root/state"; mkdir -m 0700 "$state"
git clone --quiet --no-hardlinks --no-checkout -- "$project_dir" "$clone" >"$state/clone.log" 2>&1 || fail "clean clone failed"
git -C "$clone" checkout --quiet --detach "$commit" >>"$state/clone.log" 2>&1 || fail "commit checkout failed"
[[ "$(git -C "$clone" rev-parse HEAD)" == "$commit" && -z "$(git -C "$clone" status --porcelain)" ]] \
    || fail "clean clone identity failed"
[[ -z "$(find "$clone" -type d \( -name target -o -name node_modules \) -print -quit)" ]] || fail "clone contains generated state"
ln -s -- "$model" "$state/model.gguf"
if (exec 3<>/dev/tcp/127.0.0.1/18082) 2>/dev/null; then fail "Web smoke port is occupied"; fi

version=""
for backend in "${available[@]}"; do
    prefix="$temp_root/prefix-$backend"; log="$state/$backend.log"
    (cd "$clone" && ./support/install.sh --backend "$backend" --profile release --prefix "$prefix") >"$log" 2>&1 \
        || fail "installation failed for $backend"
    binary="$prefix/bin/graph-horizon"
    current_version="$($binary --version 2>>"$log")" || fail "version check failed for $backend"
    [[ "$current_version" =~ ^graph-horizon\ [0-9]+\.[0-9]+\.[0-9]+([-.+][A-Za-z0-9.-]+)?$ ]] || fail "version output is invalid"
    [[ -z "$version" || "$version" == "$current_version" ]] || fail "installed versions differ"
    version="$current_version"
    "$binary" --help >>"$log" 2>&1 || fail "help check failed for $backend"
    grep -q '^Usage: graph-horizon' "$log" || fail "help output is invalid"
    (cd "$state" && exec "$binary" --mode web --model "$state/model.gguf" --host 127.0.0.1 --port 18082 \
        --context-tokens 2048 --kv-quant f16 --max-tokens 2) >>"$log" 2>&1 & server_pid=$!
    ready=false
    for _ in {1..180}; do
        kill -0 "$server_pid" 2>/dev/null || break
        if curl --fail --silent --max-time 2 http://127.0.0.1:18082/internal/context >"$state/context.json" 2>/dev/null; then ready=true; break; fi
        sleep 1
    done
    $ready || fail "Web startup failed for $backend"
    curl --fail --silent --max-time 5 http://127.0.0.1:18082/ >"$state/index.html" 2>>"$log" \
        || fail "Web UI check failed for $backend"
    grep -qi '<html' "$state/index.html" || fail "installed Web UI is invalid"
    curl --fail --silent --max-time 5 http://127.0.0.1:18082/internal/runtime >"$state/runtime.json" 2>>"$log" \
        || fail "runtime check failed for $backend"
    grep -Fq "\"backend\":\"$backend\"" "$state/runtime.json" || fail "backend substitution detected for $backend"
    if [[ "$backend" == *-hybrid ]]; then
        grep -Eq '"mode":"(all-gpu|all-metal|mixed)"' "$state/runtime.json" || fail "hybrid backend used CPU-only placement"
    fi
    curl --fail --silent --max-time 300 -H 'content-type: application/json' -H 'x-graph-horizon-cache: 00112233445566778899aabbccddeeff' \
        --data-binary '{"messages":[{"role":"user","content":"Ciao"}]}' \
        http://127.0.0.1:18082/internal/chat >"$state/chat.sse" 2>>"$log" \
        || fail "generation failed for $backend"
    grep -Fq '"stats":' "$state/chat.sse" && grep -Fq '"done":true' "$state/chat.sse" \
        && ! grep -Fq '"error":' "$state/chat.sse" || fail "generation result failed for $backend"
    stop_server
    case "$backend" in vulkan) vulkan_status=PASS ;; vulkan-hybrid) vulkan_hybrid_status=PASS ;; metal) metal_status=PASS ;; metal-hybrid) metal_hybrid_status=PASS ;; esac
done

cpu_binary="$temp_root/prefix-cpu/bin/graph-horizon"; cli_log="$state/cli.log"
cli_command="stty cols 120 rows 40; exec $cpu_binary --model $state/model.gguf --context-tokens 2048 --kv-quant f16 --max-tokens 2"
{ for _ in {1..180}; do grep -aq 'Graph Horizon' "$cli_log" 2>/dev/null && break; sleep 1; done; printf 'Ciao\r'; for _ in {1..180}; do grep -aqE '[1-9][0-9]* out' "$cli_log" 2>/dev/null && break; sleep 1; done; printf '\033'; } \
    | timeout 420 script -qefc "$cli_command" "$cli_log" >/dev/null 2>&1 \
    || fail "installed CLI generation failed"
grep -aqE '[1-9][0-9]* out' "$cli_log" || fail "installed CLI generation was incomplete"

prompt='Spiega in una frase che cosa misura un benchmark riproducibile.'
(cd "$clone" && cargo run --locked --release --no-default-features --features "$benchmark_backend" \
    --example bench -- "$state/model.gguf" --context 2048 --kv f16 --prompt "$prompt" \
    --max-tokens 64 --warmup 1 --reps 3) >"$state/bench.out" 2>"$state/bench.log" \
    || fail "benchmark failed"
[[ "$(wc -l <"$state/bench.out")" == 1 ]] || fail "benchmark output is invalid"
prompt_tokens=""; p_median=""; p_sd=""; p_cv=""; t_median=""; t_sd=""; t_cv=""; m_median=""; m_sd=""; m_cv=""
read -ra fields <"$state/bench.out"
for field in "${fields[@]}"; do
    case "$field" in
        prompt_tokens=*) prompt_tokens="${field#*=}" ;; prompt_tps_median=*) p_median="${field#*=}" ;;
        prompt_tps_stddev=*) p_sd="${field#*=}" ;; prompt_tps_cv=*) p_cv="${field#*=}" ;;
        ttft_ms_median=*) t_median="${field#*=}" ;; ttft_ms_stddev=*) t_sd="${field#*=}" ;; ttft_cv=*) t_cv="${field#*=}" ;;
        model_decode_tps_median=*) m_median="${field#*=}" ;; model_decode_tps_stddev=*) m_sd="${field#*=}" ;; model_decode_tps_cv=*) m_cv="${field#*=}" ;;
    esac
done
[[ "$prompt_tokens" =~ ^[1-9][0-9]*$ ]] || fail "benchmark token count is invalid"
for value in "$p_median" "$p_sd" "$t_median" "$t_sd" "$m_median" "$m_sd"; do [[ "$value" =~ ^[0-9]+\.[0-9]{2}$ ]] || fail "benchmark metric is invalid"; done
for value in "$p_cv" "$t_cv" "$m_cv"; do [[ "$value" =~ ^[0-9]+\.[0-9]{4}$ ]] || fail "benchmark dispersion is invalid"; done

cat >"$state/report.md" <<EOF
# Graph Horizon Public Readiness Report

Overall result: **PASS**

| Field | Value |
|---|---|
| Commit | \`$commit\` |
| Graph Horizon version | \`$version\` |
| Model ID | \`$model_id\` |
| Model | \`$model_name\` |
| Model SHA-256 | \`$actual_hash\` |
| Operating system / architecture | \`$os $kernel / $arch\` |
| CPU | $cpu |
| GPU | $gpu |
| GPU driver | $driver |
| Vulkan runtime | $vulkan_runtime |
| Backend | \`$benchmark_backend\` |
| KV / context | \`f16 / 2048\` |
| Prompt | “$prompt” |
| Prompt tokens | $prompt_tokens |
| Maximum tokens / warm-up / repetitions | \`64 / 1 / 3\` |

| Metric | Median | Sample standard deviation | CV |
|---|---:|---:|---:|
| TTFT (ms) | $t_median | $t_sd | $t_cv |
| Prompt throughput (tokens/s) | $p_median | $p_sd | $p_cv |
| Model decode throughput (tokens/s) | $m_median | $m_sd | $m_cv |

## Backend Verification

| Backend | Result |
|---|---|
| cpu | PASS |
| vulkan | $vulkan_status |
| vulkan-hybrid | $vulkan_hybrid_status |
| metal | $metal_status |
| metal-hybrid | $metal_hybrid_status |

## Limits

- This measurement is valid only for this machine, commit, model, and configuration.
- It does not generalize to other GPUs or platforms.
- This activity makes no comparison with llama.cpp.
- TTFT follows Graph Horizon's public boundary and includes tokenization and first sampling.
- The benchmark measures performance; by itself it qualifies neither correctness nor output quality.
EOF
install -m 0644 "$state/report.md" "$report" || fail "report cannot be written"
printf '%s\n' 'clone/install: PASS' 'backend cpu: PASS' "backend vulkan: $vulkan_status" \
    "backend vulkan-hybrid: $vulkan_hybrid_status" "backend metal: $metal_status" \
    "backend metal-hybrid: $metal_hybrid_status" "benchmark $benchmark_backend: PASS" 'report: PASS'
