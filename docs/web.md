<!--
This document owns the current local browser UI, static routes, chat behavior and
transfer format. It does not describe removed tool or workspace capabilities.
-->

# Local Web UI

`--mode web` loads a supported model into the local engine, serves the built
Svelte frontend, and delegates chat to the server's text-only pipeline.

```sh
gh-zero-engine --mode web --model /path/to/model.gguf \
  --host 127.0.0.1 --port 8080
```

The default bind is `127.0.0.1:8080`. There is no authentication or TLS: binding
outside loopback explicitly exposes the service to the network.

## Frontend Build

The frontend uses Svelte, Vite, and TypeScript. The installer automatically runs:

```sh
cd web/frontend
npm ci
npm run build
```

With `cargo run`, assets must already exist in `web/frontend/dist`. If
`index.html` is missing, startup terminates with `web frontend build missing`.

`npm run dev`, `npm run check`, and `npm run build` are available for frontend
development.

## Routes

| Method And Path | Effect |
|---|---|
| `GET /` | Serves `index.html` |
| `GET /index.html` | Serves `index.html` |
| `GET /assets/*` | Serves a built asset |
| `POST /v1/chat/completions` | Streams chat with SSE |

There is no SPA fallback for other paths and there are no workspace, tool, or
confirmation routes. Asset paths are percent-decoded and accept only normal
relative components; traversal, absolute paths, and invalid UTF-8 receive `404`.

## Browser Chat

The UI keeps the conversation in browser memory and sends `system`, `user`, and
`assistant` messages. The system prompt is editable and saved in `localStorage`
under `gh-zero.system-prompt`; storage errors do not block the session.

Every request uses `fetch` with:

```json
{
  "messages": [],
  "max_tokens": 1024,
  "stream": true
}
```

The browser displays only `delta.content` and final statistics. Tool or reasoning
frames are protocol errors for the current UI. The stop button aborts the fetch
and retains the active pair, including partial text, to preserve alternating
history.

`--provider` is ignored. `--context-tokens`, KV, threads, and placement configure
the local engine. The UI always sends `max_tokens: 1024`, so the corresponding
flag does not change requests from the included frontend.

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

This container differs from the JSON array used by TUI `/export` and `/import`.

## Errors And Limits

The chat body is limited to 4 MiB. HTTP and stream errors become short UI
messages; parser, engine, and filesystem details are not shown.

The current UI does not include multiple sessions, login, API keys, tool calling,
a workspace, a separate reasoning channel, or advanced sampling controls.
