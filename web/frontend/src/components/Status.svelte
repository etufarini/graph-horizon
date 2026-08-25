<script lang="ts">
  /*
   * Status.svelte
   * Presents persistence warnings, request errors, and accessible context
   * progress as one compact status group. Exact engine metrics remain separate.
   */
  import type { ContextUsage, PersistenceWarning } from '../chat/types';

  export let warning: PersistenceWarning | null;
  export let error: string | null;
  export let usage: ContextUsage | null;
  const tokens = new Intl.NumberFormat('en-US');
  $: fillClass = !usage || usage.percent < 80
    ? 'fill-normal'
    : usage.percent < 100 ? 'fill-warning' : 'fill-error';
  $: warningText = warning === 'invalid-record'
    ? 'Invalid saved conversation: starting with an empty chat'
    : warning === 'unavailable' ? 'Persistence unavailable: the conversation will remain in memory only' : null;
</script>

{#if warningText || error || usage}
  <div class="status-stack">
    {#if warningText}<div class="status-message status-warning" role="status">{warningText}</div>{/if}
    {#if error}<div class="status-message status-error" role="alert">{error}</div>{/if}
    {#if usage}
      <div class="status-panel">
        <span>Context ≈{tokens.format(usage.estimatedTokens)} / {tokens.format(usage.contextLimit)} · {usage.percent}%</span>
        <div
          class="context-track"
          role="progressbar"
          aria-label="Context usage"
          aria-valuemin="0"
          aria-valuemax="100"
          aria-valuenow={usage.progress}
        >
          <div class="context-fill {fillClass}" style:width={`${usage.progress}%`}></div>
        </div>
      </div>
    {/if}
  </div>
{/if}

<style lang="scss">
  .status-stack { min-width: 0; display: grid; gap: var(--gn-space-xs); }
  .status-message { border: var(--gn-border-width) solid currentColor; border-radius: var(--gn-radius-sm); padding: var(--gn-space-sm); font-size: var(--gn-text-sm); font-weight: 600; }
  .status-warning { color: var(--gn-warning); background: var(--gn-warning-bg); }
  .status-error { color: var(--gn-error-fg); background: var(--gn-error-bg); }
  .status-panel { min-height: var(--gn-control-height); box-sizing: border-box; display: flex; align-items: center; gap: var(--gn-space-sm); border: var(--gn-border-width) solid var(--gn-border); border-radius: var(--gn-radius-sm); background: var(--gn-bg-panel); box-shadow: var(--gn-shadow-small); padding: var(--gn-space-xs) var(--gn-space-sm); color: var(--gn-text-muted); font: 650 var(--gn-text-xs) var(--gn-font-mono); font-variant-numeric: tabular-nums; }
  .status-panel span { min-width: 0; flex: 1 1 auto; overflow-wrap: anywhere; }
  .context-track { width: clamp(72px, 24%, 180px); height: 6px; flex: 0 0 auto; border: var(--gn-rule-width) solid var(--gn-border); border-radius: 0; background: var(--gn-bg-panel-raised); overflow: hidden; }
  .context-fill { height: 100%; transition: width var(--gn-motion-fast) ease-out; }
  .fill-normal { background: var(--gn-ready); }
  .fill-warning { background: var(--gn-warning); }
  .fill-error { background: var(--gn-error-vivid); }
  @media (prefers-reduced-motion: reduce) { .context-fill { transition: none; } }
</style>
