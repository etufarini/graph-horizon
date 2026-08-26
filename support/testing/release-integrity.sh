#!/usr/bin/env bash
#
# Published-release identity verifier: compares one immutable annotated version
# tag with its anonymous remote tag, source archive, checksum, and archive root.
# It deliberately has no relationship to HEAD or the moving main branch.

set -euo pipefail

repository=""
tag=""
repository_seen=false
tag_seen=false
temp_dir=""

usage() {
    echo "usage: release-integrity.sh --repository owner/repository --tag vMAJOR.MINOR.PATCH"
}

fail() {
    printf 'release-integrity: %s\n' "$1" >&2
    exit "${2:-1}"
}

cleanup() {
    # Cleanup is permitted only for the exact namespace created below.
    if [[ "$temp_dir" =~ ^/tmp/graph-horizon-release-integrity\.[A-Za-z0-9]+$ && -d "$temp_dir" ]]; then
        rm -rf -- "$temp_dir"
    fi
}

trap cleanup EXIT
trap 'exit 130' HUP INT TERM

while (($#)); do
    case "$1" in
        --repository)
            (($# >= 2)) || fail "missing --repository value" 2
            $repository_seen && fail "duplicate --repository" 2
            repository="$2"; repository_seen=true; shift 2
            ;;
        --tag)
            (($# >= 2)) || fail "missing --tag value" 2
            $tag_seen && fail "duplicate --tag" 2
            tag="$2"; tag_seen=true; shift 2
            ;;
        --help|-h) usage; exit 0 ;;
        *) fail "invalid arguments" 2 ;;
    esac
done
$repository_seen && $tag_seen || fail "required arguments are missing" 2

[[ "$repository" =~ ^[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || fail "invalid repository" 2
owner="${repository%%/*}"
name="${repository#*/}"
[[ ${#owner} -le 39 && ${#name} -le 100 && "$owner" =~ [A-Za-z0-9]$ \
    && "$name" =~ [A-Za-z0-9]$ && "$owner" != *--* ]] \
    || fail "invalid repository" 2
[[ "$tag" =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] \
    || fail "invalid version tag" 2

for tool in curl git gzip mktemp rm tar; do
    command -v "$tool" >/dev/null 2>&1 || fail "$tool is required"
done
if command -v sha256sum >/dev/null 2>&1; then
    sha256_tool=sha256sum
elif command -v shasum >/dev/null 2>&1; then
    sha256_tool=shasum
else
    fail "sha256sum or shasum is required"
fi

[[ "$(git cat-file -t "refs/tags/$tag" 2>/dev/null || true)" == tag ]] \
    || fail "local tag is missing or is not annotated"
local_commit="$(git rev-parse --verify "$tag^{commit}" 2>/dev/null)" \
    || fail "local tag commit cannot be resolved"
[[ "$local_commit" =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]] \
    || fail "local tag commit is invalid"

remote_url="https://github.com/$repository.git"
remote_output="$(GIT_TERMINAL_PROMPT=0 git -c credential.helper= ls-remote \
    "$remote_url" "refs/tags/$tag" "refs/tags/$tag^{}" 2>/dev/null)" \
    || fail "remote tag cannot be resolved anonymously"
remote_object=""
remote_commit=""
while IFS=$'\t' read -r object reference extra; do
    [[ -n "$object$reference${extra:-}" ]] || continue
    [[ "$object" =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ && -z "${extra:-}" ]] \
        || fail "remote tag response is invalid"
    case "$reference" in
        "refs/tags/$tag") [[ -z "$remote_object" ]] || fail "remote tag response is invalid"; remote_object="$object" ;;
        "refs/tags/$tag^{}") [[ -z "$remote_commit" ]] || fail "remote tag response is invalid"; remote_commit="$object" ;;
        *) fail "remote tag response is invalid" ;;
    esac
done <<< "$remote_output"
[[ -n "$remote_object" && -n "$remote_commit" ]] \
    || fail "remote annotated tag is missing"
[[ "$remote_commit" == "$local_commit" ]] \
    || fail "local and remote tag commits differ"

version="${tag#v}"
archive_name="graph-horizon-$version.tar.gz"
expected_root="graph-horizon-$version"
asset_url="https://github.com/$repository/releases/download/$tag"
temp_dir="$(mktemp -d /tmp/graph-horizon-release-integrity.XXXXXX)" \
    || fail "temporary directory cannot be created"
[[ "$temp_dir" =~ ^/tmp/graph-horizon-release-integrity\.[A-Za-z0-9]+$ && -d "$temp_dir" ]] \
    || fail "temporary directory is invalid"
archive_path="$temp_dir/$archive_name"
checksum_path="$temp_dir/$archive_name.sha256"
members_path="$temp_dir/members"

curl --fail --location --silent --show-error --output "$archive_path" \
    "$asset_url/$archive_name" || fail "source archive download failed"
curl --fail --location --silent --show-error --output "$checksum_path" \
    "$asset_url/$archive_name.sha256" || fail "checksum download failed"

checksum_line=""
checksum_lines=0
while IFS= read -r line || [[ -n "$line" ]]; do
    ((checksum_lines += 1))
    checksum_line="$line"
done < "$checksum_path"
expected_hash="${checksum_line%% *}"
[[ $checksum_lines -eq 1 && "$expected_hash" =~ ^[0-9a-f]{64}$ \
    && "$checksum_line" == "$expected_hash  $archive_name" ]] \
    || fail "checksum record is invalid"
if [[ "$sha256_tool" == sha256sum ]]; then
    hash_output="$(sha256sum -- "$archive_path")" || fail "source checksum cannot be computed"
else
    hash_output="$(shasum -a 256 "$archive_path")" || fail "source checksum cannot be computed"
fi
actual_hash="${hash_output%%[[:space:]]*}"
[[ "$actual_hash" == "$expected_hash" ]] || fail "source checksum mismatch"

gzip -t "$archive_path" 2>/dev/null || fail "source archive is invalid"
archive_commit="$(git get-tar-commit-id < <(gzip -dc "$archive_path"))" \
    || fail "archive commit cannot be resolved"
[[ "$archive_commit" =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]] \
    || fail "archive commit is invalid"
[[ "$archive_commit" == "$local_commit" ]] \
    || fail "tag and archive commits differ"

LC_ALL=C tar -tzf "$archive_path" > "$members_path" 2>/dev/null \
    || fail "source archive member list is invalid"
root_seen=false
member_seen=false
while IFS= read -r member || [[ -n "$member" ]]; do
    member_seen=true
    [[ -n "$member" && "$member" != *[$'\001'-$'\037'$'\177']* \
        && "$member" != *\\* && "$member" != *//* \
        && "$member" =~ ^[A-Za-z0-9_./@+%=-]+$ ]] \
        || fail "source archive has a malformed member"
    case "$member" in
        "$expected_root/") root_seen=true ;;
        "$expected_root/"*) ;;
        *) fail "source archive has an unexpected root" ;;
    esac
    path="${member%/}"
    IFS='/' read -r -a components <<< "$path"
    for component in "${components[@]}"; do
        [[ -n "$component" && "$component" != . && "$component" != .. ]] \
            || fail "source archive has an unsafe member"
    done
done < "$members_path"
$member_seen && $root_seen || fail "source archive root is missing"

printf 'tag: %s\nversion: %s\nlocal tag commit: %s\nremote tag commit: %s\n' \
    "$tag" "$version" "$local_commit" "$remote_commit"
printf 'archive commit: %s\narchive SHA-256: %s\narchive root: %s/\n' \
    "$archive_commit" "$actual_hash" "$expected_root"
echo "release-integrity: PASS"
