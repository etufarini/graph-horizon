#!/usr/bin/env bash
#
# Remote acquisition boundary: downloads and authenticates the v0.1.3 source
# release, validates its shape, and delegates every build/install decision
# unchanged to support/install.sh. It owns no backend/profile/prefix policy.

set -euo pipefail

release_url="https://github.com/etufarini/graph-horizon/releases/download/v0.1.3"
archive_name="graph-horizon-0.1.3.tar.gz"
archive_url="${release_url}/${archive_name}"
checksum_url="${release_url}/${archive_name}.sha256"
expected_root="graph-horizon-0.1.3"
temp_dir=""

fail() {
    printf 'bootstrap: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    [[ -z "${temp_dir}" ]] || rm -rf -- "${temp_dir}"
}

validate_archive() {
    local member component
    [[ -s "${temp_dir}/members" ]] || fail "source archive is empty"
    while IFS= read -r member; do
        case "${member}" in
            "${expected_root}"|"${expected_root}"/*) ;;
            *) fail "source archive has an unsafe member" ;;
        esac
        IFS='/' read -r -a components <<< "${member}"
        for component in "${components[@]}"; do
            [[ "${component}" != "." && "${component}" != ".." ]] \
                || fail "source archive has an unsafe member"
        done
    done < "${temp_dir}/members"
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

for prerequisite in bash curl tar mktemp find awk; do
    command -v "${prerequisite}" >/dev/null 2>&1 || {
        printf 'bootstrap: %s is required\n' "${prerequisite}" >&2
        exit 2
    }
done
if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
    fail "sha256sum or shasum is required"
fi

temp_dir="$(mktemp -d)" || fail "cannot create temporary directory"
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

archive="${temp_dir}/source.tar.gz"
checksum_file="${temp_dir}/source.sha256"
curl --fail --location --silent --show-error --output "${archive}" "${archive_url}"
curl --fail --location --silent --show-error --output "${checksum_file}" "${checksum_url}"
read -r expected_hash expected_name extra < "${checksum_file}" \
    || fail "source checksum record is invalid"
[[ "${expected_hash}" =~ ^[0-9a-f]{64}$ && "${expected_name}" == "${archive_name}" && -z "${extra:-}" ]] \
    || fail "source checksum record is invalid"
[[ "$(sha256_file "${archive}")" == "${expected_hash}" ]] \
    || fail "source checksum mismatch"
tar -tzf "${archive}" > "${temp_dir}/members"
validate_archive
mkdir "${temp_dir}/source"
tar -xzf "${archive}" -C "${temp_dir}/source"
[[ -z "$(find "${temp_dir}/source" -type l -print -quit)" ]] \
    || fail "source archive contains a symbolic link"

source_root="${temp_dir}/source/${expected_root}"
for required in support/install.sh Cargo.toml web/frontend/package.json web/frontend/package-lock.json; do
    [[ -f "${source_root}/${required}" ]] || fail "source archive is incomplete"
done

status=0
bash "${source_root}/support/install.sh" "$@" || status=$?
exit "${status}"
