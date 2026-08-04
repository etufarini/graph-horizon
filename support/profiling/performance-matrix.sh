#!/usr/bin/env bash
# GH Zero 3B local performance matrix runner.
# Owns strict tuple order, artifact authentication, immutable per-revision
# executables, bounded child output, and terminal JSONL normalization. Inputs are
# untrusted; model inference and A/B decisions remain in their dedicated tools.

set -u -o pipefail

usage_error() { printf 'performance-matrix: invalid arguments\n' >&2; exit 2; }
[[ $# -eq 8 ]] || usage_error
models= hardware= driver= cache=
while [[ $# -gt 0 ]]; do
    flag=$1; value=$2; shift 2
    case "$flag" in
        --models-dir) [[ -z "$models" ]] || usage_error; models=$value ;;
        --hardware-id) [[ -z "$hardware" ]] || usage_error; hardware=$value ;;
        --driver-id) [[ -z "$driver" ]] || usage_error; driver=$value ;;
        --binary-cache) [[ -z "$cache" ]] || usage_error; cache=$value ;;
        *) usage_error ;;
    esac
done
id_ok() { [[ ${#1} -ge 1 && ${#1} -le 96 && $1 != *[!A-Za-z0-9._-]* ]]; }
id_ok "$hardware" && id_ok "$driver" && [[ -d "$models" ]] || usage_error
[[ $cache == target/performance/* && $cache != *..* && $cache != *$'\n'* ]] || usage_error

root=$(cd "$(dirname "$0")/../.." && pwd -P) || usage_error
[[ $PWD == "$root" ]] || usage_error
source "$root/support/artifact.sh"
revision=$(git rev-parse HEAD 2>/dev/null) || usage_error
[[ ${#revision} -eq 40 && $revision != *[!0-9a-f]* ]] || usage_error
mkdir -p target/performance || usage_error
run_dir=$(mktemp -d target/performance/matrix.XXXXXX) || usage_error
cleanup() { [[ $run_dir == target/performance/matrix.* ]] && rm -rf -- "$run_dir"; }
trap cleanup EXIT HUP INT TERM

manifest=$cache/manifest
reuse=0
if [[ -e "$cache" ]]; then
    [[ -d "$cache" ]] || usage_error
    if [[ -f "$manifest" ]]; then reuse=1
    elif [[ -n $(find "$cache" -mindepth 1 -print -quit 2>/dev/null) ]]; then usage_error
    fi
else
    mkdir "$cache" || usage_error
fi

profiles=(cpu metal metal-hybrid)
prepare() {
    local profile=$1 state=pass binary size hash platform
    platform=$(uname -s 2>/dev/null || true)
    if [[ $profile == metal* && $platform != Darwin ]]; then state='platform unavailable'
    elif ! command -v cargo >/dev/null 2>&1 || ! artifact_hash_tool_available; then state='tool unavailable'
    elif ! cargo build --locked --release --no-default-features --features "$profile" --example phases >/dev/null 2>&1; then state=fail
    else
        binary=target/release/examples/phases
        [[ -x "$binary" ]] || state=fail
        if [[ $state == pass ]]; then
            mkdir "$cache/$profile" || usage_error
            cp "$binary" "$cache/$profile/phases" || usage_error
            size=$(artifact_size "$cache/$profile/phases") || usage_error
            hash=$(artifact_sha256 "$cache/$profile/phases") || usage_error
        fi
    fi
    printf '%s\n' "$state" > "$run_dir/$profile.status"
    printf '%s\t%s\t%s\t%s\n' "$profile" "$state" "${size:-null}" "${hash:-null}" >> "$run_dir/manifest"
}

if [[ $reuse -eq 0 ]]; then
    printf 'revision\t%s\n' "$revision" > "$run_dir/manifest"
    for profile in "${profiles[@]}"; do prepare "$profile"; done
    mv "$run_dir/manifest" "$manifest" || usage_error
else
    [[ $(wc -l < "$manifest") -eq 4 && $(sed -n '1p' "$manifest") == $'revision\t'"$revision" ]] || usage_error
    for profile in "${profiles[@]}"; do
        entry=$(awk -F '\t' -v p="$profile" '$1==p {print $2 "\t" $3 "\t" $4}' "$manifest")
        IFS=$'\t' read -r state size hash <<< "$entry"
        case "$state" in pass|fail|'tool unavailable'|'platform unavailable') ;; *) usage_error ;; esac
        if [[ $state == pass ]]; then
            binary=$cache/$profile/phases
            [[ -x "$binary" && $(artifact_size "$binary") == "$size" && $(artifact_sha256 "$binary") == "$hash" ]] || usage_error
            [[ $(find "$cache/$profile" -mindepth 1 -maxdepth 1 -print | wc -l) -eq 1 ]] || usage_error
        else [[ ! -e "$cache/$profile" ]] || usage_error
        fi
        printf '%s\n' "$state" > "$run_dir/$profile.status"
    done
fi

catalog() {
    local wanted=$1 found=0 id chat file bytes sha q8
    while IFS=$'\t' read -r id chat file bytes sha q8; do
        [[ -z "$id" || $id == \#* ]] && continue
        if [[ $id == "$wanted" ]]; then
            ((found += 1)); printf '%s\t%s\t%s\t%s\n' "$file" "$bytes" "$sha" "$chat"
        fi
    done < support/models.tsv
    [[ $found -eq 1 ]]
}

terminal() {
    local status=$1 reason=$2 profile=$3 model=$4 bytes=$5 sha=$6 kv=$7 fixture=$8
    local mode=pure percent=null
    [[ $profile == *-hybrid ]] && { mode=mixed; percent=25; }
    printf '{"schema_version":2,"status":"%s","reason":"%s","revision":"%s","backend_profile":"%s","family":"mistral3","model_id":"%s","variant":"instruct","artifact_bytes":%s,"artifact_sha256":"%s","kv":"%s","placement_mode":"%s","cpu_layers":null,"gpu_layers":null,"weights_percent":%s,"context":4096,"fixture":"%s","fixture_digest":null,"hardware_id":"%s","driver_id":"%s","warmup":1,"repetitions":3,"prompt_tokens":null,"decode_steps":null,"prefill_mean_ns":null,"prefill_tps_mean":null,"prefill_tps_stddev":null,"prefill_tps_cv":null,"first_sample_mean_ns":null,"decode_p50_mean_ns":null,"decode_p95_mean_ns":null,"decode_tps_mean":null,"decode_tps_stddev":null,"decode_tps_cv":null,"public_ttft_ms":null,"public_decode_tps":null,"cpu_memory_total":null,"gpu_memory_total":null}' "$status" "$reason" "$revision" "$profile" "$model" "$bytes" "$sha" "$kv" "$mode" "$percent" "$fixture" "$hardware" "$driver"
}

pass=0; fail=0; external=0; total=0
write_row() {
    local row=$1
    [[ ${#row} -le 32768 && $row == '{"schema_version":2,'* ]] || usage_error
    case "$row" in *'"status":"pass"'*) ((pass += 1)) ;;
        *'"status":"fail"'*) ((fail += 1)) ;;
        *'"status":"external verification"'*) ((external += 1)) ;; *) usage_error ;; esac
    ((total += 1)); printf '%s\n' "$row"
}

model=3b-instruct
meta=$(catalog "$model") || usage_error
IFS=$'\t' read -r file bytes sha chat <<< "$meta"
[[ $chat == instruct && -n $file && $file != */* && -n $bytes && $bytes != *[!0-9]*
    && ${#sha} -eq 64 && $sha != *[!0-9a-f]* ]] || usage_error
path=$models/$file; artifact=pass
if [[ ! -e "$path" ]]; then artifact='artifact unavailable'
elif [[ ! -f "$path" || ! artifact_hash_tool_available ]]; then artifact='tool unavailable'
elif [[ $(artifact_size "$path") != "$bytes" || $(artifact_sha256 "$path") != "$sha" ]]; then artifact='artifact mismatch'
fi
for profile in "${profiles[@]}"; do
    state=$(<"$run_dir/$profile.status")
    for kv in f16 int8; do for fixture in short long; do
        if [[ $artifact != pass ]]; then
            if [[ $artifact == 'artifact mismatch' ]]; then row=$(terminal fail "$artifact" "$profile" "$model" "$bytes" "$sha" "$kv" "$fixture")
            else row=$(terminal 'external verification' "$artifact" "$profile" "$model" "$bytes" "$sha" "$kv" "$fixture"); fi
        elif [[ $state != pass ]]; then
            if [[ $state == fail ]]; then row=$(terminal fail 'execution failed' "$profile" "$model" "$bytes" "$sha" "$kv" "$fixture")
            else row=$(terminal 'external verification' "$state" "$profile" "$model" "$bytes" "$sha" "$kv" "$fixture"); fi
        else
            binary=$cache/$profile/phases; weight_args=()
            [[ $profile == *-hybrid ]] && weight_args=(--weights-percent 25)
            # The + guard keeps an empty array truly argument-free under Bash 3.2 with set -u.
            "$binary" "$path" --context 4096 --kv "$kv" --fixture "$fixture" "${weight_args[@]+"${weight_args[@]}"}" \
                --model-id "$model" --variant instruct --artifact-bytes "$bytes" --artifact-sha256 "$sha" \
                --revision "$revision" --hardware-id "$hardware" --driver-id "$driver" \
                > "$run_dir/row" 2>/dev/null
            child=$?
            if [[ $child -ne 0 || $(wc -l < "$run_dir/row") -ne 1 ]]; then row=$(terminal fail 'execution failed' "$profile" "$model" "$bytes" "$sha" "$kv" "$fixture")
            else row=$(<"$run_dir/row"); fi
        fi
        write_row "$row"
    done; done
done
[[ $total -eq 12 ]] || usage_error
printf '{"schema_version":2,"summary":true,"total":12,"pass":%d,"fail":%d,"external_verification":%d,"revision":"%s"}\n' "$pass" "$fail" "$external" "$revision"
