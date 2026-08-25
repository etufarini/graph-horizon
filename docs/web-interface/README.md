<!--
This index owns startup, frontend-build, and navigation information for the
integrated local Web UI. Detailed browser behavior lives in the linked pages.
-->

# Local Web Interface

Web mode loads one supported model into the local engine and serves the bundled
Svelte frontend through a private same-origin transport.

```sh
graph-horizon --mode web --model /absolute/path/to/model.gguf \
  --host 127.0.0.1 --port 8080
```

The default address is `http://127.0.0.1:8080`. Graph Horizon does not add TLS
or authentication; binding outside loopback deliberately exposes the interface
to that network.

| Document | Contents |
|---|---|
| [Browser chat interface](browser-chat-interface.md) | Conversation lifecycle, controls, capacity, streaming, status, and errors |
| [Browser files and persistence](browser-files-and-persistence.md) | Saved chats, Markdown files, browser storage, import, and export |
| [Web search and privacy](web-search-and-privacy.md) | Built-in providers, JSON override, result bounds, citations, and data disclosure |

## Frontend Build

The frontend requires Node.js 22.12 or newer, Svelte, Vite, and TypeScript. The
installer builds it automatically. From a source checkout:

```sh
cd web/frontend
npm ci
npm run check
npm test
npm run build
```

`npm run dev` starts the frontend development server.

`cargo run` expects `web/frontend/dist/index.html`. An installed executable
loads assets from `<prefix>/share/graph-horizon/web`, independent of the current
directory.

## Private Transport

The Rust host serves the entry document, compiled assets, immutable runtime
facts, and streamed local generation required by the bundled frontend. Paths,
payloads, and stream frames are private implementation details, not a public
HTTP API. Unknown routes have no SPA fallback. Static paths reject traversal,
absolute components, invalid UTF-8, and unsafe percent-decoding.
