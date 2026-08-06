<!--
This document owns the current headless HTTP route, request validation, SSE
contract, concurrency and security boundaries. It is model-family neutral.
-->

# Server Mode

`--mode server` loads a supported model into the local engine and exposes a
text-only OpenAI-compatible chat endpoint plus read-only capacity properties. It
does not forward requests to external providers.

```sh
graph-horizon --mode server --model /path/to/model.gguf \
  --host 127.0.0.1 --port 8080
```

The default bind is `127.0.0.1:8080`. `--host 0.0.0.0` exposes the service to the
network without adding authentication or TLS.

## Startup

Before binding, the process validates common flags, requires `--model`, loads
the GGUF, and builds shared state. Missing files, incompatible metadata, and
allocation errors therefore surface before the first request.

Concrete model support is defined in the
[crate README](../crates/graph_horizon_engine/README.md).

## Routes

### `GET /props`

This observational route returns the engine-resolved positive context and the
configured generation limit:

```json
{
  "default_generation_settings": {
    "n_ctx": 32768,
    "max_tokens": 1024
  }
}
```

The public fields are `default_generation_settings.n_ctx` and
`default_generation_settings.max_tokens`. The response contains no model path,
GGUF metadata, backend, memory, generation statistic, or internal error. It does
not read a request body, acquire a chat permit or lock, allocate inference
buffers, or start generation.

### `POST /v1/chat/completions`

The body reads only:

- `messages`, a non-empty array of text objects with a `system`, `user`, or
  `assistant` role;
- `max_tokens`, an optional positive integer.

`tools`, `tool_choice`, `tool`/`function` roles, `tool_calls`, `function_call`,
and non-string content receive `400`. Unknown fields such as `model`,
`temperature`, and `stream` are ignored to tolerate existing OpenAI clients and
do not change execution. Sampling is selected from the loaded model profile:
Instruct is greedy, while Reasoning uses the qualified temperature `0.7` policy.

When `max_tokens` is absent, the server uses `--max-tokens`, whose server default
is `1024`. A positive value in the request is retained even when it exceeds that
default: the flag is not a global ceiling.

A different method on either exact public route receives the existing
client-safe `405` JSON error. `/props/` and every other unknown route receive
`404`; no prefix or trailing-slash normalization is applied.

The server does not add a capacity preflight to chat requests from external API
clients. CLI and Web admission are client-side; the engine remains the final
validation boundary for direct calls.

## SSE Streaming

A successful response uses `Content-Type: text/event-stream` and produces:

1. zero or more `chat.completion.chunk` frames with `delta.content`;
2. a `usage` frame with numeric `prompt_tokens`, `completion_tokens`,
   `prefill_ms`, and `decode_ms`;
3. a frame with `finish_reason: "stop"`;
4. `data: [DONE]`.

There is no `delta.reasoning_content`. If the engine fails after opening the
stream, the server sends a generic error and `[DONE]`, without usage or internal
details. The CLI and browser measure total client-perceived duration
independently; removing rate presentation does not change these SSE fields.

## Concurrency And Cancellation

A global mutex serializes generations on the single engine. A semaphore admits
at most eight requests, including waiting requests; further requests receive
`429` without waiting. Bodies and JSON are validated before occupying a slot.

Generation runs in a blocking task and sends ready frames through a channel
bounded to 32 entries. A slow client applies backpressure instead of allowing
unbounded buffering; the lock and permit remain tied to that generation. If the
client drops the stream, the closed receiver cancels decoding and releases the
lock, permit, and request resources.

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
- at most 32 queued SSE frames per admitted generation;
- generic JSON errors without paths, stack traces, or OS details;
- loopback bind by default;
- no authentication, authorization, TLS, or per-identity rate limit;
- no tool calling or filesystem access through HTTP.
