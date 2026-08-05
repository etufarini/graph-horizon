#!/usr/bin/env bash
#
# Supplies portable, source-only artifact size and SHA-256 functions. Callers
# own all files and subprocess policy; sourcing this file performs no checks,
# output, mutation, or fallback beyond choosing an available compatible tool.

artifact_size() {
    local file="$1" value=""
    [[ -f "$file" && -r "$file" ]] || return 1
    if command -v stat >/dev/null 2>&1; then
        value="$(stat -c %s -- "$file" 2>/dev/null)" || value=""
        if [[ ! "$value" =~ ^[0-9]+$ ]]; then
            value="$(stat -f %z -- "$file" 2>/dev/null)" || value=""
        fi
    fi
    [[ "$value" =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "$value"
}

artifact_sha256() {
    local file="$1" output="" digest=""
    [[ -f "$file" && -r "$file" ]] || return 1
    if command -v sha256sum >/dev/null 2>&1; then
        output="$(sha256sum -- "$file" 2>/dev/null)" || return 1
    elif command -v shasum >/dev/null 2>&1; then
        output="$(shasum -a 256 -- "$file" 2>/dev/null)" || return 1
    else
        return 1
    fi
    read -r digest _ <<<"$output"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || return 1
    printf '%s\n' "$digest"
}

artifact_size_tool_available() {
    command -v stat >/dev/null 2>&1
}

artifact_hash_tool_available() {
    command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1
}
