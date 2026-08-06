<!--
This document owns the canonical cross-surface context estimate and admission
contract. Console and Web pages own presentation details and link back here.
-->

# Context Capacity And Admission

Graph Horizon preserves every message supplied to a request. It never removes,
truncates, summarizes, compresses, splits, or rewrites conversation history to
make a request fit. A client either sends the complete assembled request or
rejects it before local generation or HTTP transport.

The estimate is deliberately dependency-free and approximate. It is not a
tokenizer and does not replace the engine or provider's exact validation.

## Canonical Arithmetic

The values are:

- `context_limit`: the positive effective model window in tokens;
- `max_tokens`: the reserved response capacity;
- `estimated_messages`: the estimated occupancy of messages only;
- `safe_total_budget`: the context window after the 10% safety margin;
- `required_budget`: message occupancy plus the response reserve.

Both clients use exactly these formulas:

```text
total_code_points = sum(Unicode code points in every outgoing message content)
estimated_messages = floor(total_code_points / 4)
safe_total_budget = floor(context_limit * 90 / 100)
required_budget = estimated_messages + max_tokens
admitted = required_budget <= safe_total_budget
```

Characters are summed before the single division. Rust counts Unicode scalar
values with `chars()`; JavaScript iterates strings by Unicode code point rather
than UTF-16 code unit. All additions, percentage calculations, and budget
calculations are checked. An overflow is rejected as out of budget.

The equality boundary is admitted. `max_tokens` participates in admission but
is not included in the displayed occupancy.

## Included Messages

The estimate covers the exact content a surface would send:

- the trimmed Web system prompt or the configured CLI system prompt;
- every committed raw user and assistant message, including raw Reasoning
  markers in assistant content;
- the current trimmed Web draft while idle;
- the current raw CLI draft while editing;
- the attachment-expanded CLI prompt after submission and before admission;
- the submitted user message and partial raw assistant response while streaming.

Failed CLI turns and rolled-back Web transport errors have no model messages and
remain outside occupancy. A stopped Web partial pair remains in browser history
and therefore stays included. An imported oversized conversation is retained;
only its next attempted submission is rejected.

CLI attachment expansion affects the submitted request estimate. A rejection
restores the original typed spelling, and a successful turn commits that raw
spelling rather than embedded file contents.

## Context Discovery And Reserves

An explicit positive `--context-tokens` always wins. The local CLI otherwise
uses the loaded engine's resolved context. The HTTP CLI and bundled Web UI read
`GET /props`, whose public value is
`default_generation_settings.n_ctx`.

The CLI reserve is its configured `--max-tokens`. The bundled Web UI always
uses 1024 tokens for both admission and the request body.

The HTTP CLI exits before terminal initialization when context discovery is
unavailable, unless an explicit override was supplied. The Web composer remains
disabled and shows `Configurazione del contesto non disponibile`. When the
reserve leaves no prompt capacity, both surfaces report
`max_tokens non lascia spazio al prompt`.

An out-of-budget rejection reports estimated message tokens, reserved response
tokens, and the safe budget. It preserves CLI input, Web draft, conversation
state, and the previous successful generation duration because no generation
started.

## Presentation Edge Cases

Empty message occupancy is zero. The CLI displays occupied and limit counts but
no percentage or progress bar. The Web UI displays
`ceil(estimated_messages * 100 / context_limit)` as text: it may exceed 100%,
while the accessible graphical value and fill remain clamped to 100.

See [console.md](console.md) for the terminal label and [web.md](web.md) for the
browser bar, colors, and timing presentation.
