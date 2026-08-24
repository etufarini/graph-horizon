#!/usr/bin/env bash
#
# Local source installer: validates one explicit host/backend/profile tuple,
# builds the Web UI and Rust binary, and installs the executable plus Web assets
# in one prefix. Remote acquisition and prerequisite installation stay excluded.

set -euo pipefail

script_source="${BASH_SOURCE[0]}"
script_parent="${script_source%/*}"
[[ "${script_parent}" != "${script_source}" ]] || script_parent="."
script_dir="$(cd -- "${script_parent}" && pwd)"
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

[[ -n "${backend}" ]] \
    || fail "--backend is required; accepted: cpu, vulkan, vulkan-hybrid, metal, metal-hybrid"
case "${backend}" in
    cpu|vulkan|vulkan-hybrid|metal|metal-hybrid) ;;
    *) fail "invalid backend: ${backend}; accepted: cpu, vulkan, vulkan-hybrid, metal, metal-hybrid" ;;
esac
case "${profile}" in
    release|fast) ;;
    *) fail "invalid build profile: ${profile}" ;;
esac
[[ -n "${prefix}" && "${prefix}" == /* ]] || fail "invalid install prefix"
[[ "${prefix}" != *[$'\001'-$'\037'$'\177']* ]] || fail "invalid install prefix"
case "${prefix}/" in
    */./*|*/../*) fail "invalid install prefix" ;;
esac
while [[ "${prefix}" != "/" && "${prefix}" == */ ]]; do
    prefix="${prefix%/}"
done
[[ "${prefix}" != "/" ]] || fail "invalid install prefix"

for prerequisite in bash uname install npm cargo rustc find curl; do
    command -v "${prerequisite}" >/dev/null 2>&1 || fail "${prerequisite} is required"
done

rust_version="$(rustc --version 2>/dev/null)" || fail "cannot determine Rust version"
rust_version="${rust_version#rustc }"
rust_version="${rust_version%% *}"
IFS=. read -r rust_major rust_minor rust_patch <<< "${rust_version}"
[[ "${rust_major}" =~ ^[0-9]+$ && "${rust_minor}" =~ ^[0-9]+$ && "${rust_patch}" =~ ^[0-9]+$ ]] \
    || fail "cannot determine Rust version"
((rust_major > 1 || (rust_major == 1 && rust_minor >= 88))) \
    || fail "Rust 1.88 or newer is required"

os="$(uname -s)"
arch="$(uname -m)"
case "${os}/${arch}/${backend}" in
    Darwin/arm64/cpu|Darwin/arm64/vulkan|Darwin/arm64/vulkan-hybrid|Darwin/arm64/metal|Darwin/arm64/metal-hybrid) ;;
    Linux/x86_64/cpu|Linux/x86_64/vulkan|Linux/x86_64/vulkan-hybrid) ;;
    *) fail "unsupported platform/backend: ${os}/${arch}/${backend}; Metal requires macOS on arm64" ;;
esac
if [[ "${backend}" == metal || "${backend}" == metal-hybrid ]]; then
    command -v xcrun >/dev/null 2>&1 || fail "Metal requires xcrun"
    xcrun -f metal >/dev/null 2>&1 || fail "Metal compiler is unavailable"
    xcrun -f metallib >/dev/null 2>&1 || fail "Metal library tool is unavailable"
fi

(
    cd "${project_dir}/web/frontend"
    npm ci
    npm run build
)

web_dist="${project_dir}/web/frontend/dist"
[[ -f "${web_dist}/index.html" ]] || fail "frontend build completed without index.html"
# The install tree must never dereference a link emitted by build tooling.
[[ -z "$(find "${web_dist}" ! -type d ! -type f -print -quit)" ]] \
    || fail "frontend build contains an unsupported file"

profile_args=(--profile "${profile}")
(
    cd "${project_dir}"
    cargo build --locked --no-default-features --features "${backend}" \
        "${profile_args[@]}" -p graph-horizon
)

binary="${project_dir}/target/${profile}/graph-horizon"
[[ -f "${binary}" ]] || fail "build completed without the expected binary"
bindir="${prefix}/bin"
assetdir="${prefix}/share/graph-horizon/web"
legacy="${bindir}/gh-zero-engine"
[[ ! -d "${legacy}" || -L "${legacy}" ]] \
    || fail "legacy command path is a directory: ${legacy}"
install -d "${assetdir}"
while IFS= read -r -d '' source; do
    relative="${source#"${web_dist}"}"
    install -d "${assetdir}${relative}"
done < <(find "${web_dist}" -type d -print0)
while IFS= read -r -d '' source; do
    relative="${source#"${web_dist}"}"
    install -m 0644 "${source}" "${assetdir}${relative}"
done < <(find "${web_dist}" -type f -print0)
install -d "${bindir}"
install -m 0755 "${binary}" "${bindir}/graph-horizon"
# One relative link keeps the legacy command on the exact installed artifact.
ln -sfn "graph-horizon" "${legacy}"
printf 'installed graph-horizon (prefix=%s, backend=%s, profile=%s)\n' \
    "${prefix}" "${backend}" "${profile}"
case ":${PATH:-}:" in
    *":${bindir}:"*) ;;
    *) printf 'install: %s is not in PATH; add it manually\n' "${bindir}" ;;
esac
