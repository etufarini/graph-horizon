#!/usr/bin/env bash
#
# Remote acquisition boundary: downloads one disposable main-branch snapshot,
# validates its shape, and delegates every build/install decision unchanged to
# support/install.sh. It owns no backend, profile, prefix, or build policy.

set -euo pipefail

archive_url="https://github.com/etufarini/graph-horizon/archive/refs/heads/main.tar.gz"
expected_root="graph-horizon-main"
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

for prerequisite in bash curl tar mktemp find; do
    command -v "${prerequisite}" >/dev/null 2>&1 || {
        printf 'bootstrap: %s is required\n' "${prerequisite}" >&2
        exit 2
    }
done

temp_dir="$(mktemp -d)" || fail "cannot create temporary directory"
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

archive="${temp_dir}/source.tar.gz"
curl --fail --location --silent --show-error --output "${archive}" "${archive_url}"
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
