<!--
This reference owns browser conversation state, generation lifecycle, controls,
capacity presentation, Reasoning rendering, responsive layout, and bounded
error behavior. Persistence, files, and search are documented separately.
-->

# Browser Chat Interface

The browser keeps the active conversation in memory and sends only `system`,
`user`, and `assistant` messages to the private local transport. Each chat owns
one editable system prompt.

## Request And Stream Contract

An ordinary request has the strict body `{"messages":[]}`. The page also sends
one random lowercase-hex `x-graph-horizon-cache` key for its lifetime. On
standalone Vulkan and Metal, one KV allocation can reuse only an exact rendered
token prefix with the same key. CPU and hybrid profiles do not reuse prefixes.

At startup the browser requests context and runtime facts with three-second
timeouts. Send remains disabled until the context limit and advertised search
reserve are safe positive integers. Invalid configuration shows `Context
configuration unavailable`; optional runtime metadata never weakens admission.

The bundled frontend accepts only the expected search-provenance, text,
prefill/decode, statistics, and terminal frames. Statistics are non-negative,
prefill tokens cannot exceed prompt tokens, and nothing may follow the terminal
statistics. Invalid JSON, unknown fields, error objects, or invalid ordering
interrupt the response without exposing payload details.

A five-minute inactivity watchdog starts before the request and resets on every
non-empty body chunk. It is not a total-generation timeout. Stop aborts the
request and retains the active pair, including partial or empty assistant text.
The composer remains editable during streaming, but Send, system-prompt edits,
history mutation, file mutation, and turn controls remain disabled.

An empty retained assistant is valid browser history but invalid prior engine
history. Before the next request, the user must regenerate, edit, or delete that
turn; otherwise the response is interrupted while the pair remains intact.

`--max-tokens` bounds Web generation and defaults to the resolved context
limit. Context, KV, threads, and placement configure the same in-process engine
used by the terminal. Instruct sampling is greedy; supported Reasoning profiles
use the qualified `temperature=0.7` policy.

## Chat And Turn Controls

`New chat` creates an empty chat unless the active chat is already empty and has
no system prompt. A first stable turn derives its title from the first 48
Unicode code points of the trimmed user message. History is ordered by newest
`updatedAt`, then ID. Rename accepts 1–80 Unicode code points; deletion requires
confirmation and always leaves one active chat.

Every complete user message supports Edit. Only the final pair supports Delete
and Regenerate. Editing an earlier prompt asks before removing its response and
all later turns. Regeneration excludes the previous assistant response;
edit-and-restart uses only messages before the selected pair. Admission runs
before visible mutation or transport.

A failed new request removes its uncommitted pair. A failed regeneration or
edit restores the exact previous transcript and successors. Stop commits the
candidate pair, clears timing, updates recency, and saves only after abort has
settled. Stream deltas are never persisted individually.

Search-aware edit and regeneration use a fresh assistant identity and discard
stale provenance. Stream events are accepted only while chat and assistant IDs
still match, preventing late events from overwriting a replacement turn.

## Responsive Workspace

History and Markdown-file panels start closed, use compact counted edge tabs,
and never create duplicate header toggles. Only one panel opens at a time. At
1200 CSS pixels or wider it docks; below that it is an accessible drawer closed
by its button, Escape, selection, or backdrop. Focus returns to the originating
tab, the obscured chat becomes inert, and coarse pointers use 44-pixel targets.

Panel expansion is page-session state and is not persisted. Transcript,
history, file list, and file preview own independent bounded scrolling while
the dynamic viewport and composer remain fixed. Eligible controls remain
keyboard reachable and visibly focused.

## Context And Generation Status

The canonical arithmetic is defined in
[context capacity and admission](../engine/context-capacity-and-admission.md).
Idle occupancy includes the trimmed system prompt, committed raw messages,
trimmed draft, and current Markdown framing. Streaming adds the submitted user
message and partial raw response. Active search additionally reserves the full
2,800-character result framing ceiling.

Admission occurs before creating a message pair, `AbortController`, or `fetch`.
Rejection therefore preserves the transcript and draft. Imported oversized
conversations remain readable, but their next oversized submission is rejected.
The assembled UTF-8 request must also fit the backend's 4 MiB body limit.

The status rail shows `Context ≈N / M · P%` and an accessible progress bar.
The visible percentage may exceed 100%; graphical and ARIA values clamp to 100.
Color changes at 80% and 100%.

Runtime identity shows a bounded GGUF display name or `Local model`, the static
backend, retained weights, full-context KV capacity, and optional hybrid
placement. It never exposes model paths, device names, or physical-memory
capacity. Placement details divide weights, KV, scratch, fixed, staging,
crossing, reserve, and totals by owner; they are load-time plans, not live RSS.

An admitted request moves through `Waiting`, `Prefill`, and `Decode`. Exact
engine statistics provide prompt, actually-prefilled, completion, and phase
timing. Stop, timeout, invalid frames, or failure clear request telemetry. A
capacity rejection preserves the previous successful metrics because no
generation began.

## Reasoning Presentation

Reasoning output remains ordinary assistant text. Only an exact case-sensitive
`[THINK]` after optional leading whitespace opens a THINK section; the first
`[/THINK]` closes it. The browser renders THINK in a collapsed native
disclosure and sanitizes both sections through the normal Markdown pipeline.

An incomplete possible prefix remains hidden while streaming. Contradiction or
completion with an incomplete prefix falls back to ordinary text. A recognized
opening without a close shows `Incomplete response`; empty completed THINK
shows `No response`. Other marker spellings or positions remain literal text.
Raw content, markers, and leading whitespace are preserved in history and
transfer formats.

## Errors And Limits

Transport failures use short fixed messages such as `Request failed` and
`Response interrupted`; backend, parser, storage, search, and filesystem details
are not exposed. The interface has no accounts, authentication, cross-tab merge,
response branching, chat search, model selection, tool calling, workspace,
separate Reasoning channel, or advanced sampling controls.
