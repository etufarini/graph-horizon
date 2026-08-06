<!--
This document owns the canonical cross-surface context estimate and admission
contract. Console and Web pages own presentation details and link back here.
-->

# Context Capacity And Admission

Graph Horizon applies the capacity gate to the complete assembled request before
generation or HTTP transport. The dependency-free estimate is approximate, not
a tokenizer or a replacement for engine and provider validation.

## Canonical Arithmetic

`context_limit` is the positive effective model window; `max_tokens` is the
response reserve; `estimated_messages` is message occupancy;
`safe_total_budget` is the window after the 10% margin; and `required_budget`
is occupancy plus reserve.

Both clients use exactly these formulas:

```text
total_code_points = sum(Unicode code points in every outgoing message content)
estimated_messages = floor(total_code_points / 4)
safe_total_budget = floor(context_limit * 90 / 100)
required_budget = estimated_messages + max_tokens
admitted = required_budget <= safe_total_budget
```

Characters are summed before the single division. Rust uses `chars()` and
JavaScript iterates by Unicode code point, not UTF-16 code unit. Arithmetic is
checked; overflow is rejected as out of budget. Equality is admitted.
`max_tokens` participates in admission but not displayed occupancy.

## Included Messages

The estimate covers the exact content a surface would send:

- the Web's trimmed or CLI's configured system prompt;
- every committed raw user and assistant message, including Reasoning markers;
- the trimmed Web or raw CLI draft while idle;
- the attachment-expanded CLI prompt between submission and admission;
- the submitted user and partial raw assistant messages while streaming.

Failed CLI turns and rolled-back Web transport errors remain outside occupancy.
A stopped Web partial pair remains in history and stays included. Imported
oversized history is retained; only its next submission is rejected.

CLI attachments affect the submitted estimate. Rejection restores the original
spelling; success commits that spelling rather than embedded file contents.

## Context Discovery And Reserves

An explicit positive `--context-tokens` wins. Otherwise the local CLI uses the
engine's resolved context, while HTTP CLI and Web read `GET /props` at
`default_generation_settings.n_ctx`.

The CLI reserve is configured `--max-tokens`; Web always uses 1024 for both
admission and the request body.

Without an override, HTTP CLI discovery failure exits before terminal setup.
Web disables its composer and shows `Configurazione del contesto non disponibile`.
With no prompt capacity, both report `max_tokens non lascia spazio al prompt`.

An admission rejection reports estimate, reserve, and safe budget. It preserves
CLI input, Web draft, conversation, and the previous successful duration.

## Presentation Edge Cases

Empty occupancy is zero. CLI shows counts without a percentage or bar. Web text
uses `ceil(estimated_messages * 100 / context_limit)` and may exceed 100%; its
accessible graphical value and fill are clamped to 100.

See [console.md](console.md) for the terminal label and [web.md](web.md) for the
browser bar, colors, and timing presentation.
