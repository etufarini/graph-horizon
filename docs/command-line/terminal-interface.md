<!--
This document owns the interactive terminal lifecycle, controls, rendering,
derived THINK section, and compact status. Capacity arithmetic is delegated to
the canonical context contract.
-->

# Terminal Interface

The TUI lives under `src/graph_horizon_cli/console/`, loads the selected model
in-process, and alternates between input and streaming.

## Session Cycle

1. **Input**: editing, history, completion, and scrolling. Empty input is ignored.
2. **Admission**: attachment expansion, complete request assembly, and the
   checked capacity gate from
   [context-capacity-and-admission.md](../engine/context-capacity-and-admission.md).
3. **Streaming**: request opening, incremental text consumption, and periodic rendering.

A completed non-empty turn enters model history as the raw user prompt and raw
assistant response. An error or empty response remains visible but is not sent
back to the model. Local commands produce notices that are excluded from
context.

Attachments are expanded only in the outgoing copy before admission. Their
contents contribute to that request's estimate. If admission fails, the typed
prompt is restored unchanged and no generation starts; a successful turn
commits the raw `@path` spelling.

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
described in
[slash-commands-and-file-attachments.md](slash-commands-and-file-attachments.md).

## Keyboard During Streaming

| Key | Effect |
|---|---|
| `Up`/`Down` | Scrolls one line |
| `PageUp`/`PageDown` | Scrolls one page |
| `Home` | Moves to the beginning of output |
| `End` | Moves to the bottom and restores auto-follow |
| `Esc` | Closes the application and discards the partial turn |
| `Ctrl+C` | Cancels an open stream and restores the prompt |

Before the first engine event, `Ctrl+C` has no effect; `Esc` still closes the
application.

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

## Reasoning Presentation

The engine emits ordinary text-only messages. Rendering recognizes raw
Reasoning text only when, after leading whitespace, it starts with
the exact case-sensitive `[THINK]` marker. The first following `[/THINK]` closes
the THINK section. Its `[THINK]` label and content use the `Secondary` style;
the final answer remains in the `Response` style. This terminal section is not
collapsible.

While an opening prefix such as `[TH` is still undecided, the prefix stays hidden
and the existing generating spinner remains visible. A contradiction, or stream
completion with an incomplete prefix, renders the entire raw response as an
ordinary Response. A recognized opening without a close renders its remainder as
THINK and, only after completion, adds `[no response]` in Secondary. Empty THINK
keeps its label, and markers after the recognized boundaries remain literal.
Marker-shape fallback does not create a connection or runtime error.

Completed and imported turns derive the same presentation again from their raw
responses. Context estimation, model history, `/import`, and `/export` retain
that raw content with markers and leading whitespace. `Ctrl+C`
continues to cancel streaming and discard the complete partial turn; every other
keyboard behavior remains as documented above.

## Capacity And Status

The scrollable conversation header identifies the
loaded model, compile-time backend, retained model weights, full-context KV
capacity, and effective hybrid placement. The weight/KV summary is present for
every backend profile; hybrid profiles additionally show planned CPU and
accelerator budget totals, including the withheld reserve. These are immutable load-time values, not live
RSS or allocator telemetry. The header never includes the model path, device
identity, or physical-memory capacity.

The right-aligned status adapts to terminal width. While a request is active it
shows `waiting`, `prefill`, or `decode` with a monotonic phase-local timer. After
a Graph Horizon generation completes, wide terminals show prompt, actually
prefilled, and output counts plus separate phase durations and token rates:

```text
ctx ~1.2k/32.8k tok | 1.2k in · 96 pf · 42 out | pf 0.80s 120.0/s · dec 1.40s 30.0/s
```

Medium widths retain output and both rates; narrow widths retain output and
decode rate. Counts below 1000 are integers; larger counts use one decimal and
`k`. Only the occupied estimate has `~`. A zero-duration rate is `—`. The CLI
has no percentage and no graphical progress bar.

Occupancy includes the raw draft while editing, the expanded submitted prompt
and partial assistant response while streaming, then the committed raw pair on
success. A failed or empty turn remains outside model occupancy. `/import`
recomputes occupancy and clears the prior duration.

The live phase timer measures client-perceived time from receipt of each phase
boundary; the completed durations and rates come only from engine usage. A new
admitted turn clears the previous result; interruption or failure leaves no new
metrics. A rejected prompt never starts the timer and therefore preserves the
previous successful status.

While a capacity error is present, the status row instead reports estimated
messages, reserved tokens, and safe budget. The next edit, recall, completion,
or submission clears that transient error without clearing the restored prompt.

The CLI always requests greedy sampling. The Web UI instead uses the loaded
model profile's default sampling policy.
