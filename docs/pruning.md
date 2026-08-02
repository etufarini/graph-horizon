<!--
This document owns the TUI token estimate, pruning threshold and protected-turn
invariants. It does not define model tokenizer or server-side context behavior.
-->

# Prompt Pruning

Pruning keeps TUI requests within an estimated budget by removing the oldest
turn pairs. The Rust server and web backend do not retain history: each request
contains the complete `messages` array from the client. The web UI keeps its own
conversation in the browser.

## Token Estimate

The TUI does not load a tokenizer to estimate history. It counts the Unicode
code points in all messages, sums the total first, and applies this once:

```text
estimated_tokens = total_characters / 4
```

Dividing each message separately would introduce repeated truncation and
underestimate many short conversations. The estimate remains heuristic and does
not replace exact checks by the engine or provider.

## Context And Reserve

The window comes from one of these sources:

- `--context-tokens`, which takes precedence;
- the context resolved by the engine with the local provider;
- the HTTP provider's `GET /props`, when available.

The output reserve is `--max-tokens`, with TUI default `2048`. There are no
runtime response or reasoning profiles.

For a valid window, the threshold is:

```text
usable = context_window - max_tokens
threshold = usable * 90 / 100
```

The 10% margin absorbs part of the `characters / 4` estimation error. The
threshold is disabled when context is missing or zero, or when the reserve
leaves no positive space for history.

## Turn Removal

Before every submission, `prune_to_limit` protects:

- the initial `system` message, when present;
- the final new `user` message.

While the estimate exceeds the threshold, it removes the oldest complete
`user`/`assistant` pair from between them. It never splits a turn. If the system
prompt and new prompt alone exceed the threshold, both remain; the engine or
provider then decides whether the actual request fits the context.

## Flow

```text
--max-tokens ----------------------------> output reserve
--context-tokens / engine / GET /props --> window
                                               |
                                               v
                                 pruning_threshold (once)
                                               |
                                               v
                                 prune_to_limit (each submission)
```

## Invariants

- The system prompt and new prompt are never removed.
- History is removed only as complete pairs.
- An active threshold is always greater than zero.
- An absent threshold leaves the request unchanged.
- The count divides the total by four exactly once.
