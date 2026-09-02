<!--
This index owns navigation for repository documentation. It separates user
tasks, stable technical contracts, maintainer processes, current project
status, and historical investigation records.
-->

# Documentation

Graph Horizon documentation is organized by reader goal. Start with installation
and the interface you intend to use; consult development processes and evidence
only when maintaining or qualifying the project.

The implementation currently integrates Ministral 3, but the Graph Horizon
project name and model-neutral architecture do not imply a separate edition.

## Start Here

| Document | Contents |
|---|---|
| [Main README](../README.md) | Project overview and shortest supported path |
| [Installation](installation.md) | Quick install, prerequisites, backend selection, prefix, and verification |
| [Supported models and formats](supported-models-and-formats.md) | Accepted model family, sizes, and GGUF profile |
| [Command-line interface](command-line/README.md) | Runtime options, terminal controls, slash commands, and attachments |
| [Local Web interface](web-interface/README.md) | Browser chat, persistence, Markdown files, search, and privacy |

## Engine Reference

| Document | Contents |
|---|---|
| [Architecture](engine/architecture.md) | Workspace boundaries and application flows |
| [Backend support status](engine/backend-support-status.md) | Build profiles and current support labels |
| [Hybrid placement and memory](engine/hybrid-placement-and-memory.md) | Immutable placement, budgets, KV, and failure policy |
| [Context capacity and admission](engine/context-capacity-and-admission.md) | Shared estimate and request-admission arithmetic |
| [Engine crate](../crates/graph_horizon_engine/README.md) | Public Rust API and engine contract |
| [Engine module ownership](../crates/graph_horizon_engine/src/module-ownership.md) | Maintainer-facing source responsibility map |

## Development

| Document | Contents |
|---|---|
| [Contributing](../CONTRIBUTING.md) | Change discipline and verification expectations |
| [Model-support addition](development/model-support-addition.md) | Process for adding a family or profile |
| [Compute-backend addition](development/compute-backend-addition.md) | Process for adding a hardware backend |
| [Model-support validation](development/model-support-validation.md) | Technical and semantic qualification process |
| [Oracle comparison](development/oracle-comparison.md) | Numeric and external-oracle evidence |
| [Ministral KV-cache validation](development/ministral-kv-cache-validation.md) | Repeatable f16/int8 KV comparison |
| [Performance investigation](development/performance-investigation.md) | Correctness-gated optimization process |
| [End-to-end throughput benchmark](development/end-to-end-throughput-benchmark.md) | Public-event benchmark interface |
| [Support script command reference](../support/script-command-reference.md) | Exact profiling and validation-script arguments |
| [Implementation decision log](development/implementation-decision-log.md) | Durable repository implementation decisions |

## Current Project Status

| Document | Contents |
|---|---|
| [Project-status index](project-status/README.md) | Authority and freshness boundaries |
| [Validation evidence](project-status/validation-evidence.md) | Artifact identities and reviewed qualification results |
| [Current performance status](project-status/current-performance-status.md) | Retained optimizations and remaining bottlenecks |
| [Release notes](project-status/release-notes.md) | Released user-facing changes and limitations |
| [v0.1.0 release qualification](project-status/v0.1.0-release-qualification.md) | Historical candidate and tag-derived gates |

## Historical Investigations

Detailed performance and cleanup reports preserve measurements and decisions at
their recorded revisions. They do not override current support or release
status. Browse the [investigation report index](investigation-reports/README.md).

## Repository Policies

- [Support](../SUPPORT.md)
- [Security](../SECURITY.md)
- [Contributing](../CONTRIBUTING.md)
- [License](../LICENSE)
