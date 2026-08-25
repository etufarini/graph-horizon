<!--
This reference owns optional Web-search selection, provider and override
contracts, result validation, citation provenance, and the external-data privacy
boundary. It does not own general chat transport or model history.
-->

# Web Search And Privacy

Search is available without configuration but remains an explicit choice for
one browser request. Starting Web mode never sends a query, and the model cannot
initiate search.

## Built-In Providers

The labelled globe control reports `Search off`, `Search on`, or `Search
unavailable`. A `web` request uses DuckDuckGo Lite; a `news` request uses Google
News RSS. Graph Horizon never reroutes one category to the other, retries a
hidden provider, or falls back to an unsearched answer after search fails.

The browser displays the exact query that will leave the process. Empty search
terms select the visible user message; otherwise the explicit query is used.
System text, earlier messages, Markdown files, and other prompt framing never
enter the query. Terms are trimmed and limited to 512 Unicode code points.

The request carries the chosen category, language hint, browser-local reference
date, and optional historical publication range. Current requests send
`published: null`; dates expressed in the user message remain natural-language
intent. Results retain usable timestamps so the model can apply that intent and
report insufficient evidence. Category and terms are never inferred or
rewritten.

When enabled, the browser adds this strict object beside `messages` in the
private generation request:

```json
{
  "search": {
    "terms": "current information requested by the user",
    "category": "news",
    "language": "it-IT",
    "reference_date": "2026-08-23",
    "published": null
  }
}
```

## JSON Provider Override

`--search-url` replaces both built-in routes with one JSON endpoint.
`--search-key-file` optionally supplies a bearer token and is valid only with
the override. Endpoint validation is defined in the
[runtime option reference](../command-line/runtime-options.md).

Graph Horizon sends one POST with `content-type: application/json`:

```json
{
  "query": "Rust 1.97 release",
  "category": "news",
  "language": "it-IT",
  "reference_date": "2026-08-24",
  "published": null
}
```

The response must have exactly this shape:

```json
{
  "results": [{
    "title": "Rust 1.97 released",
    "url": "https://example.com/rust-1-97",
    "excerpt": "A bounded factual excerpt.",
    "publisher": "Example",
    "published_at_ms": 1787565600000
  }]
}
```

Unknown fields reject the complete override response. All providers converge on
the same bounded internal representation. Graph Horizon retains at most five
distinct HTTP(S) URLs, limits every string, and never visits a result URL. When
an exact interval is present, only timestamped results inside that half-open
interval survive.

Provider challenge, HTTP 429, timeout, invalid response, unavailable provider,
and no usable results have distinct fixed UI errors.

## Transport Bounds

The host invokes `curl` without a shell, user configuration, cookies, proxy, or
redirects. An override bearer token is passed through curl configuration on
standard input, not process arguments. Each request has an eight-second transfer
timeout, nine-second process ceiling, and 512 KiB response limit. At most one
search runs at a time.

Validated excerpts are framed as untrusted data only in the current outgoing
user-message copy. Framing requires `[S1]`-style citations, omits result URLs,
and is limited to 2,800 Unicode characters; the browser reserves that maximum
before fetching.

Provenance arrives in a distinct strict stream event and is rendered outside
assistant Markdown. Sources referenced by the visible answer are labelled
`Cited`; the rest remain `Search result`. Provenance is persisted and exported
with its assistant message, but later engine history projects only role and raw
assistant content.

Edit and regeneration create a fresh assistant identity and search report. A
failed replacement restores the previous transcript; Stop keeps the replacement
without stale provenance. Late events from obsolete requests cannot overwrite
the current turn.

## Privacy Boundary

Enabling search discloses the displayed query, selected category, language hint,
local reference date, and public source IP to DuckDuckGo, Google News, or the
configured override. Inference, system prompt, chat history, Markdown files,
and stored archives remain local.

Public provider markup, availability, rate limits, ranking, and terms can change
independently. Fixed adapters fail closed when input no longer matches. Search
excerpts are provider snippets, not downloaded source pages; citation provenance
does not independently prove every generated claim. Consequential claims should
be checked by opening the cited source.
