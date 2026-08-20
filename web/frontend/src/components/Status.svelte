<script lang="ts">
  /*
   * Status.svelte
   * Single responsibility: present independent persistence warning/error
   * strips and accessible context progress. Inference telemetry is presented by
   * Metrics so estimated capacity never mixes with measured engine values.
   */
  import type { ContextUsage, PersistenceWarning } from '../chat/types';

  export let warning: PersistenceWarning | null;
  export let error: string | null;
  export let usage: ContextUsage | null;
  $: fillClass = !usage || usage.percent < 80
    ? 'fill-normal'
    : usage.percent < 100
      ? 'fill-warning'
      : 'fill-error';
  $: warningText = warning === 'invalid-record'
    ? 'Conversazione salvata non valida: avvio con una chat vuota'
    : warning === 'unavailable' ? 'Persistenza non disponibile: la conversazione resterà solo in memoria' : null;
</script>

{#if warningText}
  <div class="status-error status-warning" role="status">{warningText}</div>
{/if}

{#if error}
  <div class="status-error">{error}</div>
{/if}

{#if usage}
  <div class="status-panel">
    <div class="status-labels">
      <span>Contesto {usage.percent}%</span>
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

  .status-warning {
    border-color: var(--gn-streaming);
    color: var(--gn-text);
    background: var(--gn-bg-panel);
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
