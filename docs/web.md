<!--
This document owns the current local browser UI, static routes, multi-chat
history, last-turn controls, browser persistence, Reasoning presentation, and
transfer format. It does not own server or model policy or describe removed
tool/workspace capabilities.
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
  "max_tokens": 32768,
  "stream": true
}
```

The page also sends one random lowercase-hex `x-graph-horizon-cache` key for
its lifetime. On the Vulkan profile, the server retains one KV allocation and
reuses only an exact rendered-token prefix with the same key. Another page may
replace that slot; requests without the header keep the uncached behavior.

On page initialization the browser requests exact `/props` with a three-second
timeout. Send remains disabled until `n_ctx` and `max_tokens` are equal positive
safe integers whose 90% prompt budget is non-zero. A failed, timed-out,
malformed, unequal, non-integer, unsafe, or non-positive response shows
`Configurazione del contesto non disponibile`.

The browser consumes only non-empty string `delta.content` and requires `[DONE]`
for successful completion. It stops reading immediately at that sentinel. Usage
and final-stop frames are tolerated but do not drive presentation. Invalid JSON,
server error objects, `reasoning_content`, and tool frames interrupt the response
without exposing payload details. A 60-second inactivity watchdog starts before
the request and resets on every non-empty body chunk; it is not a total
generation timeout. The stop button aborts voluntarily and retains the active
pair, including partial raw text, to preserve alternating history.

The composer remains editable while a response is streaming, so the next
message can be prepared without starting a concurrent request. Send stays
disabled until streaming ends, while Stop remains available. A failed request
restores its submitted prompt only when no newer draft has been entered.

`--provider` and `--max-tokens` are ignored. `--context-tokens`, KV, threads, and
placement configure the local engine. The Web wrapper publishes the loaded
engine's `n_ctx` as both capacity fields.
The browser sends `max_tokens` equal to `n_ctx`.
Instruct sampling remains greedy, while a loaded Reasoning profile uses the
qualified temperature `0.7` policy.

### Saved Chat Persistence

The browser restores one chat collection synchronously from the origin-scoped
`localStorage` key `graph-horizon.conversation`. A different scheme, host, or
port therefore selects a different archive. The private compact record is
distinct from import/export and has exactly this version-2 shape:

```json
{
  "version": 2,
  "activeChatId": "00000000-0000-4000-8000-000000000001",
  "chats": [
    {
      "id": "00000000-0000-4000-8000-000000000001",
      "title": "Titolo",
      "messages": [
        { "role": "user", "content": "..." },
        { "role": "assistant", "content": "..." }
      ],
      "updatedAt": 0
    }
  ]
}
```

The list is never empty, IDs are unique UUIDs, and `activeChatId` identifies
exactly one member. Every transcript is empty or consists of complete, strictly
alternating `user`/`assistant` pairs. Titles contain 1–80 Unicode code points;
`updatedAt` is a non-negative safe Unix time in milliseconds. Runtime message
IDs are regenerated on restore and never enter the archive.

The serialized archive is limited to 4,194,304 UTF-8 bytes. A missing key
creates, activates, and immediately attempts to save one empty `Nuova chat`.
Malformed JSON, unknown versions, extra or missing fields, oversize data, and
any invariant violation cause the entire archive to be ignored and removed
when possible. Successful cleanup shows `Archivio chat non valido: avvio con
una chat vuota`; a cleanup, acquisition, read, or write failure shows
`Persistenza non disponibile: le chat resteranno solo in memoria`.

The system prompt remains global and separately stored under
`graph-horizon.system-prompt`. The archive excludes it along with drafts,
status, errors, generation timing, context settings, and presentation-only
Reasoning state.

Stable transcript checkpoints are successful `[DONE]`, settled Stop,
regenerate, edit-and-regenerate, and deletion of the final turn. They update
that chat's `updatedAt` and save the complete collection once after idle state.
Creating, selecting, renaming, or deleting a chat and importing a chat also
save immediately; selection and rename do not change `updatedAt`. Generation
start, assistant deltas, draft/system-prompt edits, capacity rejection, and
transport rollback never save the collection.

A failed stable save keeps the in-memory result and preserves the previous
stored value. Reload may therefore restore the older archive; this is not a
recovery merge. A later successful checkpoint clears the warning. There is no
`storage` listener or cross-tab merge: the last successful stable writer wins.
The CLI and server have no corresponding persistence, database, filesystem
write, or HTTP route.

### Legacy Conversation Migration

The loader recognizes only the exact legacy private record
`{"version":1,"messages":[...]}` with a valid complete transcript. It migrates
the transcript unchanged into one active chat, derives the title from the
first user message, and attempts one version-2 `setItem` on the same key.

The replacement is atomic from the application's perspective: the legacy value
is replaced only when the version-2 write succeeds. If serialization or the
write fails, the migrated collection remains usable in memory, the exact
version-1 value remains stored, and persistence is reported unavailable. A
later reload may therefore attempt migration again. Invalid legacy input follows
the invalid-archive cleanup path rather than returning a valid prefix.

### Chat History Controls

`NUOVA CHAT` creates and activates an empty chat only when the current chat is
non-empty; requesting it from an already empty active chat is a no-op. A new
chat receives an unused UUID, `Nuova chat`, and the current timestamp. Its first
stable transcript derives a title from the first 48 Unicode code points of the
trimmed first user message after collapsing whitespace, without an ellipsis.

The history list sorts by descending `updatedAt`, then ascending ID. Selection
persists only `activeChatId`. The always-visible row menu offers `RINOMINA` and
`ELIMINA`; rename trims outer whitespace, preserves inner whitespace, and
accepts 1–80 Unicode code points. Deletion requires the displayed irreversible
confirmation. Deleting the active chat selects the most recently updated
remaining chat; deleting the only chat creates one empty active replacement.
Deleting a final turn does not reset an already derived title.

History creation, selection, rename, and deletion are disabled during
streaming. Import and all final-turn controls are disabled at the same time;
Stop remains available.

### Last-Turn Controls

Only the final complete pair exposes controls: `MODIFICA` and `ELIMINA` below
the user message, and `RIGENERA` below the assistant response. Earlier turns
cannot be edited. Edit uses the current prompt as a local multiline draft;
empty trimmed edits cannot be saved, and cancel or Escape discards the draft.
Deletion asks `Eliminare l’ultimo turno? Il messaggio e la risposta verranno
rimossi.` and removes the entire pair after confirmation.

Regenerate sends the unchanged final user prompt with the system prompt and all
messages before the final pair; the previous assistant response is excluded.
Edit-and-regenerate substitutes the trimmed edited prompt. Capacity admission
runs against that candidate and prior context before fetch or visible mutation.
A rejected candidate leaves the stable transcript and archive unchanged.

A new-send transport failure removes its uncommitted appended pair. A failed
regenerate or edit-and-regenerate instead restores the exact prior user and
assistant pair. An unsuccessful or bodyless HTTP response shows `Richiesta non
riuscita`; timeout, missing `[DONE]`, or protocol/transport failure shows
`Risposta interrotta`. Stop commits the candidate prompt and current replacement
response, including an empty assistant response, clears final timing, updates
recency, and saves once after abort settles. Stream deltas are never persisted.

### Responsive Chat History

Above 720 CSS pixels, the bounded 1320-pixel application shows a collapsible
264-pixel history column beside the chat surface; closing it lets the chat use
the released width. At 720 pixels or narrower, history starts closed and opens
as a fixed left drawer over a backdrop. Selection, backdrop activation, and
Escape close the mobile drawer and return focus to the accessible history
toggle. The chat list and transcript scroll independently while the document
body and composer remain fixed in the viewport. Collapse state is not stored.

All history and final-turn controls are keyboard reachable, use visible focus
treatment, and remain visible when eligible rather than depending on hover.

## Context And Generation Status

The canonical estimate and capacity gate are defined in
[context.md](context.md). The browser compares estimated prompt occupancy alone
with the 90% safe prompt budget and sends the full context limit as request
`max_tokens`. Idle occupancy includes the
trimmed system prompt, every committed raw message, and the trimmed draft.
Streaming occupancy includes the submitted user message and current partial raw
assistant response.

Admission runs before creating the user/assistant pair, `AbortController`, or
chat `fetch`. Rejection therefore leaves messages and draft unchanged and shows
the estimate and safe prompt budget. Imported conversations may display over
100%; import remains valid, but the next oversized submission is rejected.

Whenever configuration is valid, `Contesto N%` appears above a horizontal
progress bar. The visible percentage is not capped. The fill is normal below
80%, warning from 80% through 99%, and error at 100% or above; its width and
ARIA current value are capped at 100. The element exposes `role="progressbar"`,
minimum 0, maximum 100, and the accessible name `Occupazione del contesto`.

An admitted request starts a `performance.now()` timer immediately before chat
transport. `Generazione N.Ns` refreshes every 250 ms and measures connection
wait, prefill, transport, and decode. Successful `[DONE]` completion freezes the
exact elapsed duration. Stop, inactivity, missing `[DONE]`, or failure clears
timing without publishing a final value; a capacity rejection preserves the
previous final duration because no generation started. Errors render in addition
to a valid context bar.

## Reasoning Presentation

For a supported Reasoning model, the release-owned internal system instruction
is always rendered first. Non-empty explicit system text follows it after one
blank line; empty explicit text adds no separator. User text that resembles a
Reasoning or structural marker remains ordinary tokenizer input.

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
the next render, so an unclosed leading `[THINK]` still shows `Risposta
incompleta`. Browser memory and version-1 import/export likewise retain raw
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
and complete `user`/`assistant` pairs. A valid file creates and activates a new
chat, derives its title, persists the collection, and applies its system prompt
globally. An imported empty transcript still creates a new empty chat. Invalid
or unreadable files leave the collection, active selection, archive, and global
system prompt unchanged. No replacement confirmation is shown.

Export serializes only the active chat. Unlike the private version-2 browser
archive, the public file remains version 1 and includes `systemPrompt`; private
chat IDs, titles, timestamps, and inactive chats never enter it. Both formats
share complete-pair transcript validation, but neither is the JSON array used by
TUI `/export` and `/import`.

## Errors And Limits

The HTTP chat body and the separate browser conversation record are each
limited to 4 MiB under their respective contracts. HTTP and stream errors
become short UI messages; parser, engine, storage, and filesystem details are
not shown. The Web gate rejects over-budget requests before transport.

The Web UI has no server persistence, accounts, authentication, account or
cross-tab synchronization/merge, response-version history, earlier-turn
editing, chat search, pins, archive folders, bulk deletion, or model selection
for regenerate. It also has no API-key flow, tool calling, workspace, separate
Reasoning protocol/state channel, or advanced sampling controls.
