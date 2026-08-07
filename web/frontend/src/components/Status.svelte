<script lang="ts">
  /*
   * Status.svelte
   * Single responsibility: present an additive error strip, accessible context
   * progress, and live/final client-perceived generation duration.
   */
  import { onDestroy } from 'svelte';
  import type { ContextUsage } from '../chat/types';

  export let error: string | null;
  export let usage: ContextUsage | null;
  export let generationStartedAt: number | null;
  export let generationMs: number | null;

  let now = performance.now();
  let timer: ReturnType<typeof setInterval> | null = null;

  function stopTimer(): void {
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  }

  $: if (generationStartedAt !== null && timer === null) {
    now = performance.now();
    timer = setInterval(() => (now = performance.now()), 250);
  } else if (generationStartedAt === null) {
    stopTimer();
  }

  $: elapsedMs = generationStartedAt === null
    ? generationMs
    : Math.max(0, now - generationStartedAt);
  $: fillClass = !usage || usage.percent < 80
    ? 'fill-normal'
    : usage.percent < 100
      ? 'fill-warning'
      : 'fill-error';

  onDestroy(stopTimer);
</script>

{#if error}
  <div class="status-error">{error}</div>
{/if}

{#if usage}
  <div class="status-panel">
    <div class="status-labels">
      <span>Contesto {usage.percent}%</span>
      {#if elapsedMs !== null}
        <span>Generazione {(elapsedMs / 1000).toFixed(1)}s</span>
      {/if}
    </div>
    <div
      class="context-track"
      role="progressbar"
      aria-label="Occupazione del contesto"
      aria-valuemin="0"
      aria-valuemax="100"
      aria-valuenow={usage.progress}
    >
      <div class="context-fill {fillClass}" style:width={`${usage.progress}%`}></div>
    </div>
  </div>
{/if}

<style lang="scss">
  .status-error {
    border: var(--gn-border-width) solid var(--gn-error-border);
    padding: var(--gn-space-sm) var(--gn-space-md);
    font-size: var(--gn-text-sm);
    font-weight: 600;
    color: var(--gn-error-fg);
    background: var(--gn-error-bg);
  }

  .status-panel {
    display: grid;
    gap: var(--gn-space-xs);
    color: var(--gn-text-muted);
    font-family: var(--gn-font-mono);
    font-size: var(--gn-text-xs);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .status-labels {
    display: flex;
    justify-content: space-between;
    gap: var(--gn-space-sm);
  }

  .context-track {
    height: var(--gn-space-sm);
    border: var(--gn-rule-width) solid var(--gn-border);
    background: var(--gn-bg-panel-raised);
    overflow: hidden;
  }

  .context-fill {
    height: 100%;
    transition: width var(--gn-motion-fast) ease-out;
  }

  .fill-normal {
    background: var(--gn-ready);
  }

  .fill-warning {
    background: var(--gn-streaming);
  }

  .fill-error {
    background: var(--gn-error-vivid);
  }

  @media (prefers-reduced-motion: reduce) {
    .context-fill {
      transition: none;
    }
  }
</style>
