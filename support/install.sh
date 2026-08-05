#!/usr/bin/env bash
#
# Builds the Web UI and one required explicit profile, then installs its binary.
# Arguments own the build/profile/prefix tuple; subprocesses stay synchronous.
# It installs no prerequisites and never retries another profile or toolchain.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
backend=""
profile="release"
prefix="${GRAPH_HORIZON_INSTALL_PREFIX:-${HOME}/.local}"

usage() {
    printf '%s\n' \
        "usage: install.sh --backend cpu|vulkan|vulkan-hybrid|metal|metal-hybrid [--profile release|fast] [--prefix PATH]"
}

fail() {
    printf 'install: %s\n' "$*" >&2
    exit 2
}

while (($#)); do
    case "$1" in
        --backend)
            (($# >= 2)) || fail "missing value for --backend"
            backend="$2"
            shift 2
            ;;
        --profile)
            (($# >= 2)) || fail "missing value for --profile"
            profile="$2"
            shift 2
            ;;
        --prefix)
            (($# >= 2)) || fail "missing value for --prefix"
            prefix="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            fail "unknown argument: $1"
            ;;
    esac
done

[[ -n "${backend}" ]] || fail "--backend is required"
case "${backend}" in
    cpu|vulkan|vulkan-hybrid|metal|metal-hybrid) ;;
    *) fail "invalid backend: ${backend}" ;;
esac
case "${profile}" in
    release|fast) ;;
    *) fail "invalid build profile: ${profile}" ;;
esac
[[ -n "${prefix}" && "${prefix}" != "/" ]] || fail "invalid install prefix"
case "${backend}" in
    metal|metal-hybrid)
        [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] \
            || fail "Metal requires macOS on arm64"
        command -v xcrun >/dev/null 2>&1 || fail "Metal requires xcrun"
        xcrun -f metal >/dev/null 2>&1 || fail "Metal compiler is unavailable"
        xcrun -f metallib >/dev/null 2>&1 || fail "Metal library tool is unavailable"
        ;;
esac
command -v cargo >/dev/null 2>&1 || fail "cargo is required"
command -v npm >/dev/null 2>&1 || fail "npm is required"

(
    cd "${project_dir}/web/frontend"
    npm ci
    npm run build
)

profile_args=(--profile "${profile}")
(
    cd "${project_dir}"
    cargo build --locked --no-default-features --features "${backend}" \
        "${profile_args[@]}" -p graph-horizon
)

binary="${project_dir}/target/${profile}/graph-horizon"
[[ -f "${binary}" ]] || fail "build completed without the expected binary"
bindir="${prefix}/bin"
install -d "${bindir}"
install -m 0755 "${binary}" "${bindir}/graph-horizon"
printf 'installed %s (backend=%s, build-profile=%s)\n' \
    "${bindir}/graph-horizon" "${backend}" "${profile}"
