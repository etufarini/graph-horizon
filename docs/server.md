<!--
This document owns the current headless HTTP route, request validation, SSE
contract, concurrency and security boundaries. It is model-family neutral.
-->

# Server Mode

`--mode server` loads a supported model into the local engine and exposes one
text-only, OpenAI-compatible endpoint. It does not forward requests to external
providers.

```sh
gh-zero-engine --mode server --model /path/to/model.gguf \
  --host 127.0.0.1 --port 8080
```

The default bind is `127.0.0.1:8080`. `--host 0.0.0.0` exposes the service to the
network without adding authentication or TLS.

## Startup

Before binding, the process validates common flags, requires `--model`, loads
the GGUF, and builds shared state. Missing files, incompatible metadata, and
allocation errors therefore surface before the first request.

Concrete model support is defined in the
[crate README](../crates/gh_zero_engine/README.md).

## Route

### `POST /v1/chat/completions`

The body reads only:

- `messages`, a non-empty array of text objects with a `system`, `user`, or
  `assistant` role;
- `max_tokens`, an optional positive integer.

`tools`, `tool_choice`, `tool`/`function` roles, `tool_calls`, `function_call`,
and non-string content receive `400`. Unknown fields such as `model`,
`temperature`, and `stream` are ignored to tolerate existing OpenAI clients and
do not change execution. Sampling remains greedy.

When `max_tokens` is absent, the server uses `--max-tokens`, whose server default
is `1024`. A positive value in the request is retained even when it exceeds that
default: the flag is not a global ceiling.

Every other route receives `404`; a different method on
`/v1/chat/completions` receives `405`. The server does not expose `/props`.

## SSE Streaming

A successful response uses `Content-Type: text/event-stream` and produces:

1. zero or more `chat.completion.chunk` frames with `delta.content`;
2. a `usage` frame with token counts and prefill/decode timings;
3. a frame with `finish_reason: "stop"`;
4. `data: [DONE]`.

There is no `delta.reasoning_content`. If the engine fails after opening the
stream, the server sends a generic error and `[DONE]`, without usage or internal
details.

## Concurrency And Cancellation

A global mutex serializes generations on the single engine. A semaphore admits
at most eight requests, including waiting requests; further requests receive
`429` without waiting. Bodies and JSON are validated before occupying a slot.

Generation runs in a blocking task and sends ready frames through a channel.
The lock and permit are released as soon as `generate` ends. If the client drops
the stream, the sink cancels decoding and releases resources.

## Server Flags

| Flag | Default | Effect |
|---|---|---|
| `--host <host>` | `127.0.0.1` | Bind address |
| `--port <n>` | `8080` | Port; a non-numeric value falls back to the default |
| `--max-tokens <n>` | `1024` | Default used when the body omits `max_tokens` |
| `--context-tokens <n>` | engine policy | Explicit local context |

KV, CPU, and placement options are described in
[configuration.md](configuration.md).

## Security

- maximum body size of 4 MiB, enforced before full collection;
- generic JSON errors without paths, stack traces, or OS details;
- loopback bind by default;
- no authentication, authorization, TLS, or per-identity rate limit;
- no tool calling or filesystem access through HTTP.
