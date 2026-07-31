<script lang="ts">
  /*
   * Status.svelte — the generation status bar, four exclusive states in
   * priority order: error > generating (status is 'streaming') > stats
   * (status is 'idle' and stats present) > hidden. Presentational only:
   * everything arrives via props. While generating, the figures are
   * client-side estimates (chars/4 heuristic on a ~250 ms tick owned by
   * this component); the exact prefill/decode rates come from `stats`,
   * measured by the backend.
   */
  import { onDestroy } from 'svelte';
  import type { ChatStatus, GenerationStats } from '../chat/types';

  export let status: ChatStatus;
  export let error: string | null;
  export let stats: GenerationStats | null = null;
  export let streamChars = 0;

  let elapsedMs = 0;
  // Epoch ms of the 'streaming' transition; 0 while not streaming.
  let startTime = 0;
  // Epoch ms of the first received char (streamChars 0 → >0); 0 = none yet.
  let firstCharTime = 0;
  // Live decode estimate (tok/s); null before the first char.
  let liveRate: number | null = null;
  let timer: ReturnType<typeof setInterval> | null = null;

  // Token-estimate heuristic (same as the TUI): ~4 chars per token.
  function estimateTokens(chars: number): number {
    return Math.ceil(chars / 4);
  }

  function tick(): void {
    const now = Date.now();
    elapsedMs = now - startTime;
    if (firstCharTime > 0 && streamChars > 0) {
      const seconds = Math.max(0.001, (now - firstCharTime) / 1000);
      liveRate = Math.round(estimateTokens(streamChars) / seconds);
    } else {
      liveRate = null;
    }
  }

  // Timer runs only while streaming: each 'streaming' transition fully
  // resets timer, first-char time and live estimate — no leakage between
  // consecutive generations.
  $: if (status === 'streaming' && timer === null) {
    startTime = Date.now();
    firstCharTime = 0;
    elapsedMs = 0;
    liveRate = null;
    timer = setInterval(tick, 250);
  } else if (status !== 'streaming' && timer !== null) {
    clearInterval(timer);
    timer = null;
  }

  // First-char detection: streamChars is monotone within one generation.
  $: if (timer !== null && streamChars > 0 && firstCharTime === 0) {
    firstCharTime = Date.now();
  }

  onDestroy(() => {
    if (timer !== null) {
      clearInterval(timer);
    }
  });

  // Exact rates from the backend measurements; max(1, ms) guards the
  // division, rounding happens only here at display time.
  $: prefillRate = stats ? Math.round(stats.promptTokens / (Math.max(1, stats.prefillMs) / 1000)) : 0;
  $: decodeRate = stats
    ? Math.round(stats.completionTokens / (Math.max(1, stats.decodeMs) / 1000))
    : 0;
  $: elapsedLabel = `${(elapsedMs / 1000).toFixed(1)}s`;
</script>

{#if status === 'error' && error}
  <div class="status-error">{error}</div>
{:else if status === 'streaming'}
  <div class="status-bar status-streaming">
    <span class="dot" aria-hidden="true"></span>
    <span class="label">Generazione</span>
    <span class="figure">{elapsedLabel}</span>
    {#if liveRate !== null}
      <span class="figure">~↓ <span class="rate">{liveRate}</span> tok/s</span>
    {/if}
  </div>
{:else if status === 'idle' && stats}
  <div class="status-bar status-completed fade-in" title="Prefill / Decode (ultima generazione)">
    <span class="figure">↑ <span class="rate">{prefillRate}</span></span>
    <span class="figure">↓ <span class="rate">{decodeRate}</span></span>
    <span class="figure">tok/s</span>
  </div>
{/if}

<style lang="scss">
  /* Error strip preserved verbatim: it always wins over the other states. */
  .status-error {
    border: var(--gn-border-width) solid var(--gn-error-border);
    border-radius: var(--gn-radius-sm);
    padding: var(--gn-space-sm) var(--gn-space-md);
    font-size: var(--gn-text-sm);
    font-weight: 600;
    color: var(--gn-error-fg);
    background: var(--gn-error-bg);
  }

  /* Same mono/uppercase idiom as the panel headers. */
  .status-bar {
    display: flex;
    align-items: center;
    gap: var(--gn-space-sm);
    font-family: var(--gn-font-mono);
    font-size: var(--gn-text-xs);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--gn-text-muted);
  }

  .figure {
    white-space: nowrap;
  }

  .rate {
    color: var(--gn-accent-ink);
  }

  .status-completed .rate {
    color: var(--gn-ready-ink);
  }

  .dot {
    width: var(--gn-space-xs);
    height: var(--gn-space-xs);
    border-radius: var(--gn-radius-sm);
    background: var(--gn-streaming);
    animation: pulse var(--gn-motion-pulse) ease-in-out infinite;
  }

  @keyframes pulse {
    0%,
    100% {
      opacity: 1;
      transform: scale(1);
    }
    50% {
      opacity: var(--gn-motion-low-opacity);
      transform: scale(0.75);
    }
  }

  .fade-in {
    animation: fade-in var(--gn-motion-fast) ease-out;
  }

  @keyframes fade-in {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .dot,
    .fade-in {
      animation: none;
    }
  }
</style>
