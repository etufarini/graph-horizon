<!--
This document owns the interactive console lifecycle, controls, rendering and
status indicators. Commands and pruning algorithms are documented separately.
-->

# Console: TUI, Streaming, And Status Bar

The TUI lives under `src/gh_zero_cli/console/` and alternates between input and
streaming. It uses the same text-only format with both the local engine and the
HTTP provider.

## Session Cycle

1. **Input**: editing, history, completion, and scrolling. Empty input is ignored.
2. **Streaming**: request opening, incremental text consumption, and periodic rendering.

A completed turn enters history as a user/assistant pair. An error or empty
response remains visible but is not sent back to the model. Local commands
produce notices that are excluded from context.

## Keyboard During Input

| Key | Effect |
|---|---|
| `Tab` | Completes attachments, commands, or arguments |
| `Enter` | Resolves a pending completion; otherwise submits |
| `Up`/`Down` | Navigates the last ten prompts; scrolls output at the boundaries |
| `Left`/`Right` | Moves the cursor or candidate window |
| `PageUp`/`PageDown`/`Home`/`End` | Navigates output |
| `Esc` | Closes the application |

Typing after recalling a prompt separates the draft from history. Completion is
described in [commands.md](commands.md).

## Keyboard During Streaming

| Key | Effect |
|---|---|
| `Up`/`Down` | Scrolls one line |
| `PageUp`/`PageDown` | Scrolls one page |
| `Home` | Moves to the beginning of output |
| `End` | Moves to the bottom and restores auto-follow |
| `Esc` | Closes the application and discards the partial turn |
| `Ctrl+C` | Cancels an open stream and restores the prompt |

While only waiting for the connection, `Ctrl+C` has no effect; `Esc` still
closes the application.

## Scrolling And Rendering

`ViewportState` distinguishes auto-follow from manual scrolling. A navigation
command enters manual mode; `End`, or explicitly reaching the bottom, restores
following. A resize repositions the viewport without changing that choice.

The cache rebuilds lines only when the content revision changes. Wrapping uses
Unicode display widths. The prompt cursor always occupies a visible cell and
wraps when the last line is full.

The `generating` spinner indicates an unbounded wait and does not enter history.
The palette has four centralized roles: input, secondary content, response, and
hint.

## Status Bar

Metrics are updated by the runtime and frozen when a turn ends:

| Indicator | Meaning |
|---|---|
| `↑` / `↓` `tok/s` | Prefill and output throughput; values below 1 are hidden |
| `a N` / `Σ N tok` | Estimated response tokens and total turn tokens |
| `~X.Xk tok` | Estimated full conversation size |
| `ctx X.Xk` | Known context used for pruning |
| `∞` / `pruning off` | Pruning active or unavailable |

After `/import`, metrics from the previous turn are reset.
