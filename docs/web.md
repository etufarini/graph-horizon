<!--
This document owns the current local browser UI, static routes, chat behavior,
browser persistence, Reasoning presentation, and transfer format. It does not
own server or model policy or describe removed tool/workspace capabilities.
-->

# Local Web UI

`--mode web` loads a supported model into the local engine, serves the built
Svelte frontend, and delegates chat to the server's text-only pipeline.

```sh
graph-horizon --mode web --model /path/to/model.gguf \
  --host 127.0.0.1 --port 8080
```

The default bind is `127.0.0.1:8080`. There is no authentication or TLS: binding
outside loopback explicitly exposes the service to the network.

## Frontend Build

The frontend uses Node.js 22.12 or newer, Svelte, Vite, and TypeScript. The
installer automatically runs:

```sh
cd web/frontend
npm ci
npm run build
```

With `cargo run`, assets must already exist in `web/frontend/dist`. If
`index.html` is missing, startup terminates with `web frontend build missing`.

`npm run dev`, `npm run check`, `npm test`, and `npm run build` are available
for frontend development. `npm test` runs every direct `src/chat/*.test.ts`
suite with Node's TypeScript stripping.

## Routes

| Method And Path | Effect |
|---|---|
| `GET /` | Serves `index.html` |
| `GET /index.html` | Serves `index.html` |
| `GET /assets/*` | Serves a built asset |
| `GET /props` | Returns positive context and generation limits |
| `POST /v1/chat/completions` | Streams chat with SSE |

There is no SPA fallback for other paths and there are no workspace, tool, or
confirmation routes. Asset paths are percent-decoded and accept only normal
relative components; traversal, absolute paths, and invalid UTF-8 receive `404`.

## Browser Chat

The UI keeps the conversation in browser memory and sends `system`, `user`, and
`assistant` messages. The system prompt is editable and saved in `localStorage`
under `graph-horizon.system-prompt`; storage errors do not block the session.

Every request uses `fetch` with:

```json
{
  "messages": [],
  "max_tokens": 1024,
  "stream": true
}
```

On page initialization the browser requests exact `/props` with a three-second
timeout. Send remains disabled until positive safe integer values for `n_ctx`
and `max_tokens` are known. A failed, timed-out, malformed, non-integer, unsafe,
or non-positive response shows `Configurazione del contesto non disponibile`; a
window whose reserve leaves no prompt space shows `max_tokens non lascia spazio
al prompt`.

The browser consumes only `delta.content` and the `[DONE]` terminal sentinel.
Usage frames are tolerated but do not drive presentation; missing or malformed
usage does not affect the client timer. `reasoning_content` and tool frames
remain protocol errors. The stop button aborts the fetch and retains the active
pair, including partial raw text, to preserve alternating history.

The composer remains editable while a response is streaming, so the next
message can be prepared without starting a concurrent request. Send stays
disabled until streaming ends, while Stop remains available. A failed request
restores its submitted prompt only when no newer draft has been entered.

`--provider` is ignored. `--context-tokens`, KV, threads, and placement configure
the local engine. Web `max_tokens` defaults to `1024`; `/props` propagates an
explicit `--max-tokens` to both browser admission and every request body.
Instruct sampling remains greedy, while a loaded Reasoning profile uses the
qualified temperature `0.7` policy.

### Conversation Persistence

The browser restores one conversation synchronously during UI initialization.
It uses the origin-scoped `localStorage` key `graph-horizon.conversation`; a
different scheme, host, or port therefore selects a different conversation.
An absent key starts an empty chat without a warning.

The compact record is distinct from import/export and has exactly this shape:

```json
{
  "version": 1,
  "messages": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

Messages must be empty or complete, strictly alternating user/assistant pairs.
Runtime IDs are regenerated on restore. The record excludes the system prompt,
draft, status, errors, timing, context settings, and presentation-only
Reasoning state. The system prompt remains separately stored under
`graph-horizon.system-prompt`.

Successful `[DONE]`, a settled stop, and a valid import are the only save
checkpoints. Stop saves an empty or partial assistant response as the completed
pair. `Nuova chat` is the only clear checkpoint: it retains the system prompt,
clears chat errors and generation timing, asks
`Iniziare una nuova chat? La conversazione corrente verrà eliminata.` only when
messages exist, and clears an already empty idle chat without confirmation.
Draft edits, system-prompt edits, admission rejection, generation start,
assistant deltas, timing updates, and failed-request rollback never write the
conversation record. Closing or reloading during streaming therefore leaves
the previous stable record unchanged.

Stored text and new records are limited to 4 MiB measured as UTF-8. Corrupt,
oversized, unknown-version, or structurally invalid records are ignored and
removed when possible. Successful removal shows `Conversazione salvata non
valida: avvio con una chat vuota`. If obtaining, reading, writing, or removing
`localStorage` fails, the in-memory chat continues and the UI shows
`Persistenza non disponibile: la conversazione resterà solo in memoria`. A
failed save preserves the previous stable record. A failed `Nuova chat` removal
still clears memory, but the older record may return on reload. A later
successful stable operation clears the warning.

There is no storage-event listener, merge, or synchronization. With multiple
tabs, the last successful stable writer wins. Persistence is browser-only and
stores one conversation: the CLI and server have no corresponding persistence,
database, filesystem write, or HTTP route.

## Context And Generation Status

The canonical estimate and capacity gate are defined in
[context.md](context.md). The browser uses the positive `max_tokens` reported by
`/props` for both that gate and the request body. Idle occupancy includes the
trimmed system prompt, every committed raw message, and the trimmed draft.
Streaming occupancy includes the submitted user message and current partial raw
assistant response.

Admission runs before creating the user/assistant pair, `AbortController`, or
chat `fetch`. Rejection therefore leaves messages and draft unchanged and shows
the estimate, reserve, and safe budget. Imported conversations may display over
100%; import remains valid, but the next oversized submission is rejected.

Whenever configuration is valid, `Contesto N%` appears above a horizontal
progress bar. The visible percentage is not capped. The fill is normal below
80%, warning from 80% through 99%, and error at 100% or above; its width and
ARIA current value are capped at 100. The element exposes `role="progressbar"`,
minimum 0, maximum 100, and the accessible name `Occupazione del contesto`.

An admitted request starts a `performance.now()` timer immediately before chat
transport. `Generazione N.Ns` refreshes every 250 ms and measures connection
wait, prefill, transport, and decode. Successful `[DONE]` completion freezes the
exact elapsed duration. Stop, missing `[DONE]`, or failure clears timing without
publishing a final value; a capacity rejection preserves the previous final
duration because no generation started. Errors render in addition to a valid
context bar.

## Reasoning Presentation

Raw Reasoning text is recognized only when the assistant response begins, after
leading whitespace, with the exact case-sensitive marker `[THINK]`. The first
following `[/THINK]` ends the THINK section. The browser renders that section
before the final answer in a native `THINK` disclosure that is initially closed.
Both sections use the existing sanitized Markdown renderer.

During streaming, an incomplete possible opening prefix such as `[TH` remains
hidden while it is undecided. A contradicting character immediately falls back
to showing the entire raw response as an ordinary answer. If streaming ends on
an incomplete prefix, that raw response is also ordinary. A recognized opening
without a close displays all following text in THINK; after completion its final
position reads `Risposta incompleta`. Empty completed THINK still reads `Nessuna
risposta`. Empty THINK remains a visible disclosure, and additional markers
remain literal content. Lowercase, mixed-case, misplaced, and otherwise malformed
markers are ordinary text, not runtime errors.

Stopping a partial web turn retains its raw content and derives the same view on
the next render. Browser memory and version-1 import/export likewise retain raw
assistant content, including markers and leading whitespace; presentation is not
stored as a separate Reasoning channel.

## Import And Export

The web UI exports a versioned JSON object:

```json
{
  "version": 1,
  "systemPrompt": "",
  "messages": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

Import requires `version: 1`, strings for the system prompt and message content,
and complete `user`/`assistant` pairs. An invalid file leaves the current
conversation intact. If history exists, the UI asks for confirmation before
reading and applying the file.

Unlike the browser conversation record, this file includes `systemPrompt`.
Both formats share the same complete-pair transcript validation, but neither is
the JSON array used by TUI `/export` and `/import`.

## Errors And Limits

The HTTP chat body and the separate browser conversation record are each
limited to 4 MiB under their respective contracts. HTTP and stream errors
become short UI messages; parser, engine, storage, and filesystem details are
not shown. The Web gate rejects over-budget requests before transport.

The current UI does not include multiple saved chats, login, API keys, tool
calling, a workspace, a separate Reasoning protocol/state channel, or advanced
sampling controls.
