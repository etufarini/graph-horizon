<!--
This document owns the canonical cross-surface context estimate and admission
contract. Console and Web pages own presentation details and link back here.
-->

# Context Capacity And Admission

Graph Horizon applies the capacity gate to the complete assembled request before
generation or HTTP transport. The dependency-free estimate is approximate, not
a tokenizer or a replacement for engine request validation.

## Canonical Arithmetic

`context_limit` is the positive effective model window; `estimated_messages` is
message occupancy; and `safe_prompt_budget` is the window after the 10% margin.
For the CLI, `max_tokens` remains the response reserve and `required_budget` is
occupancy plus that reserve.

Both clients share the estimate and safe-budget formulas:

```text
total_code_points = sum(Unicode code points in every outgoing message content)
estimated_messages = floor(total_code_points / 4)
safe_prompt_budget = floor(context_limit * 90 / 100)
```

Characters are summed before the single division. Rust uses `chars()` and
JavaScript iterates by Unicode code point, not UTF-16 code unit. Arithmetic is
checked; overflow is rejected as out of budget. Equality is admitted.

CLI admission remains:

```text
required_budget = estimated_messages + max_tokens
admitted = required_budget <= safe_prompt_budget
```

Web admission does not reserve generation tokens. It admits exactly when
`estimated_messages <= safe_prompt_budget`; the engine enforces the exact limit
after real rendering and tokenization.

## Included Messages

The estimate covers the exact content a surface would send:

- the Web's trimmed or CLI's configured system prompt;
- every committed raw user and assistant message, including Reasoning markers;
- the trimmed Web or raw CLI draft while idle;
- the attachment-expanded CLI prompt between submission and admission;
- the Web's current per-chat Markdown-file framing and request heading;
- the submitted user and partial raw assistant messages while streaming.

Failed CLI turns and rolled-back Web transport errors remain outside occupancy.
A stopped Web partial pair remains in history and stays included. Imported
oversized history is retained; only its next submission is rejected.

CLI attachments affect the submitted estimate. Rejection restores the original
spelling; success commits that spelling rather than embedded file contents.
Web Markdown files follow the same raw-transcript rule but remain active across
requests until deleted. Idle occupancy includes their complete deterministic
framing even with an empty draft. Upload admission also rejects a candidate
file collection whose full framing plus the current chat already exceeds the
safe prompt budget. Generation expands only the candidate outgoing user copy;
edit and regenerate use the file snapshots active at that later operation.

## Context Discovery And Reserves

An explicit positive `--context-tokens` wins. Otherwise the local CLI uses the
engine's resolved context, while HTTP CLI and Web read `GET /props` at
`default_generation_settings.n_ctx`.

The CLI reserve is configured by `--max-tokens`. In Web mode the wrapper exposes
equal `n_ctx` and `max_tokens` values from the loaded engine. The browser uses
`n_ctx` for the request allowance and uses only the 90% prompt budget for local
admission.

Without an override, HTTP CLI discovery failure exits before terminal setup.
Web disables its composer and shows `Context configuration unavailable`
for invalid capacity properties or a zero safe prompt budget. The CLI retains
`max_tokens leaves no room for the prompt` when its reserve leaves no capacity.

An admission rejection reports estimate and safe budget; the CLI also reports
its reserve. It preserves CLI input, Web draft, conversation, and the previous
successful duration.

## Presentation Edge Cases

Empty occupancy is zero. CLI shows counts without a percentage or bar. Web text
shows the estimated occupancy, immutable context limit, and
`ceil(estimated_messages * 100 / context_limit)` percentage together. The
approximation marker remains visible because the estimate is not tokenizer
output. The percentage may exceed 100%; its accessible graphical value and fill
are clamped to 100.

See [console.md](console.md) for the terminal label and [web.md](web.md) for the
browser bar, colors, and timing presentation.
