---
name: release
description: >-
  Prepare, qualify, tag, and optionally publish a new Graph Horizon version.
  Use when the user asks to release this repository or automate its version,
  source archive, checksum, Git tag, and GitHub Release workflow.
---

<!--
This skill owns Graph Horizon's release identity and qualification workflow;
feature implementation and post-release development remain separate.
-->

# Release

Always communicate in English. Follow `AGENTS.md` and preserve the repository's
simple, explicit release model: one source commit, one immutable annotated tag,
one deterministic source archive, and one adjacent SHA-256 record.

## Authority boundary

A request to prepare or automate a release authorizes local version edits,
validation, a release commit, and local artifact generation. Creating the
immutable tag, pushing commits or tags, and creating a GitHub Release are
separate mutations. Perform them only when the current request explicitly
authorizes those actions. Immediately before the first remote mutation, show
the exact commit, tag, asset names, release title, and destination remote. Never
force-push, move a tag, replace a published asset, or reuse a released version.

Do not print or inspect credentials. Use the existing authenticated Git and
`gh` configuration only when publication is authorized; missing authentication
is a blocker for publication, not for local release preparation.

## Inputs and identity

Accept a target version in strict `MAJOR.MINOR.PATCH` form and derive:

```text
version: MAJOR.MINOR.PATCH
tag: vMAJOR.MINOR.PATCH
archive: graph-horizon-MAJOR.MINOR.PATCH.tar.gz
root: graph-horizon-MAJOR.MINOR.PATCH/
checksum: graph-horizon-MAJOR.MINOR.PATCH.tar.gz.sha256
```

If the user omits the version, inspect the latest reachable version tag and the
unreleased changes. Select a patch for compatible fixes, a minor version for
compatible features, and never select a major version without explicit approval.
Ask only when the change set makes the bump genuinely ambiguous. Reject a
version that is not greater than every existing local or remote release tag.

## Preflight

1. Read `AGENTS.md`, `docs/release-notes.md`, `docs/validation.md`, the package
   manifests, `install.sh`, and the changes since the previous release tag.
2. Require a clean working tree before release edits. Identify the current
   branch, upstream, `HEAD`, previous tag, and remote. Fetch tags read-only when
   network access is available, then prove the target tag is absent locally and
   remotely. Do not release a feature branch as the canonical release branch
   unless the user explicitly chose it.
3. Classify every change since the previous tag. Separate user-facing changes,
   internal changes, compatibility changes, and qualification claims. A past
   model/backend result belongs only to its recorded commit and must not be
   silently attributed to the new release.
4. Record the invariant, smallest viable release diff, and principal risk.
   Stop if tracked changes, an existing target tag, or an unqualified required
   claim makes the identity ambiguous.

## Prepare the release commit

Update only current-version surfaces. At minimum inspect and, where applicable,
update:

- `Cargo.toml`, `crates/graph_horizon_engine/Cargo.toml`, and the two workspace
  package entries in `Cargo.lock`;
- `web/frontend/package.json` and its root entries in `package-lock.json`;
- the release URL, archive name, and expected root in `install.sh`;
- bootstrap fixtures or assertions that intentionally model the current release;
- installation examples and current-version statements in `README.md`,
  `support/README.md`, `docs/README.md`, and other current-state documents;
- `docs/release-notes.md` and `docs/validation.md`.

Use `rg` to classify every occurrence of the old version before changing it.
Preserve historical tags, archives, evidence, comparisons, and immutable release
records. Do not perform a repository-wide replacement.

Move the actual user-facing `Unreleased` content into a new release section and
restore an empty `Unreleased` section. Write release notes from the diff and
verified behavior, not commit-message wording alone. In the validation registry,
record only gates that were actually run; unavailable hardware, models, tools,
or networks remain `external verification`, never `PASS`.

Regenerate lock metadata with the repository's package managers when needed,
using no install scripts for a version-only update. Reject unrelated dependency
or lockfile churn. Re-run `rg` for the old and new versions and explain every
remaining old-version occurrence.

## Qualification gate

Run the documented release gates without weakening tests, features, thresholds,
or support claims. The minimum local gate is:

```sh
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --no-default-features \
  --features cpu -- -D warnings
cargo test --locked --workspace --all-targets --no-default-features \
  --features cpu
cargo build --locked --release --workspace --no-default-features --features cpu
cargo test --locked -p graph_horizon_engine --no-default-features \
  --features cpu --test family_agnostic source_structure -- --exact
cd web/frontend
npm ci
npm test
npm run check
npm run build
npm audit
```

Also syntax-check every tracked shell script, run the minimum Rust toolchain and
every locally available affected backend gate documented in `docs/validation.md`,
and execute applicable model, parity, semantic, installer, and product-smoke
checks. Return to the repository root, run `git diff --check`, inspect the full
diff, and confirm the tree contains no generated frontend output, model, secret,
personal path, or unrelated change.

Commit the verified preparation as one focused commit, normally
`chore(release): prepare vMAJOR.MINOR.PATCH`. Do not include release artifacts in
Git. Confirm the commit is clean and rerun any check whose input changed while
recording evidence.

## Candidate archive

Before creating the immutable tag, generate and test an archive from the exact
release commit in a temporary directory:

```sh
git archive --format=tar --prefix="graph-horizon-${version}/" HEAD |
  gzip -n > "${archive}"
sha256sum "${archive}" > "${checksum}"
```

Inspect the member list, reject absolute paths, traversal, symlinks, unexpected
roots, or missing required files, extract into a new directory, and run
`support/install.sh --backend cpu --prefix <new-absolute-prefix>`. From outside
the checkout, verify the installed binary reports the target version and smoke
the installed CLI and Web assets. Keep artifacts outside the repository.

## Tag and publish

When local tag creation is authorized and every applicable gate is green, create
exactly one annotated tag with a concise release annotation. Regenerate the
archive and checksum from that tag, then prove:

- `git rev-parse "${tag}^{commit}"` equals the release commit;
- `git get-tar-commit-id` from the archive equals that commit;
- the archive root and filename contain the target version;
- the checksum record contains one lowercase SHA-256 and the exact archive name;
- the tag-derived archive passes the same clean-prefix product smoke.

Never delete or move the tag if a post-tag check fails. Stop publication and fix
the defect in a later version.

If remote publication is explicitly authorized, require the clean release
commit to be present on the intended remote, push the annotated tag without
force, and create one GitHub Release from that existing tag with the archive and
checksum as the only generated assets. Use the release-notes section for its
body. Read back the remote tag, release metadata, and asset names; download the
published assets anonymously into a temporary directory, verify the checksum,
and smoke the public bootstrap. Never edit or replace immutable published
identity to hide a failed check.

## Final report

Report the previous and new versions, release commit and tag identity, exact
assets, every gate with `PASS`, `FAIL`, or `external verification`, local and
remote mutations performed, and any remaining publication step. A release is
complete only when every action explicitly requested by the user has current,
direct evidence.
