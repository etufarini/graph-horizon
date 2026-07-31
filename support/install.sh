#!/usr/bin/env bash
#
# Builds the static Web UI and one explicitly selected backend, then installs
# the resulting binary. This script validates values before execution, adds no
# dependencies, and never changes model files or user configuration.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
backend="hybrid"
profile="release"
prefix="${GH_ZERO_INSTALL_PREFIX:-${HOME}/.local}"

usage() {
    printf '%s\n' \
        "usage: install.sh [--backend cpu|vulkan|hybrid] [--profile release|fast] [--prefix PATH]"
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

case "${backend}" in
    cpu|vulkan|hybrid) ;;
    *) fail "invalid backend: ${backend}" ;;
esac
case "${profile}" in
    release|fast) ;;
    *) fail "invalid build profile: ${profile}" ;;
esac
[[ -n "${prefix}" && "${prefix}" != "/" ]] || fail "invalid install prefix"
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
        "${profile_args[@]}" -p gh_zero_cli
)

binary="${project_dir}/target/${profile}/gh_zero_cli"
[[ -f "${binary}" ]] || fail "build completed without the expected binary"
bindir="${prefix}/bin"
install -d "${bindir}"
install -m 0755 "${binary}" "${bindir}/gh-zero-engine"
printf 'installed %s (backend=%s, build-profile=%s)\n' \
    "${bindir}/gh-zero-engine" "${backend}" "${profile}"
