<script lang="ts">
  /*
   * Metrics.svelte
   * Presents the current engine phase or the last exact generation statistics
   * in one compact adaptive strip. Stream ordering, formulas, and lifecycle
   * state remain in the chat modules.
   */
  import { onDestroy } from 'svelte';
  import { liveTelemetry, tokensPerSecond } from '../chat/telemetry';
  import type { GenerationTelemetry } from '../chat/types';

  export let telemetry: GenerationTelemetry | null;

  let now = performance.now();
  let timer: ReturnType<typeof setInterval> | null = null;

  $: active = liveTelemetry(telemetry);
  $: if (active !== null && timer === null) {
    now = performance.now();
    timer = setInterval(() => (now = performance.now()), 250);
  } else if (active === null && timer !== null) {
    clearInterval(timer);
    timer = null;
  }
  $: elapsed = active ? Math.max(0, now - active.phaseStartedAt!) : 0;
  $: phaseLabel = active?.phase === 'waiting'
    ? 'Waiting'
    : active?.phase === 'prefill' ? 'Prefill' : 'Decode';
  $: stats = telemetry?.stats;
  $: prefillRate = stats ? tokensPerSecond(stats.prefillTokens, stats.prefillMs) : null;
  $: decodeRate = stats ? tokensPerSecond(stats.completionTokens, stats.decodeMs) : null;

  const seconds = (milliseconds: number) => `${(milliseconds / 1000).toFixed(2)} s`;
  const rate = (value: number | null) => value === null ? '—' : `${value.toFixed(1)} tok/s`;
  onDestroy(() => { if (timer !== null) clearInterval(timer); });
</script>

{#if active}
  <section class="metrics metrics-live" aria-label="Generation phase">
    <span class="phase-dot phase-{active.phase}" aria-hidden="true"></span>
    <span class="phase-label" aria-live="polite" aria-atomic="true">{phaseLabel}</span>
    <span aria-hidden="true">· {seconds(elapsed)}</span>
  </section>
{:else if stats}
  <dl class="metrics metrics-final" aria-label="Latest generation metrics">
    <div><dt>Prompt</dt><dd><strong>{stats.promptTokens} tok</strong></dd></div>
    <div><dt>Prefill</dt><dd><strong>{stats.prefillTokens} tok</strong><small>{seconds(stats.prefillMs)} · {rate(prefillRate)}</small></dd></div>
    <div class="metric-primary"><dt>Output</dt><dd><strong>{stats.completionTokens} tok</strong></dd></div>
    <div class="metric-primary">
      <dt>Decode</dt>
      <dd><strong aria-label={decodeRate === null ? 'Decode rate unavailable' : undefined}>{rate(decodeRate)}</strong><small>{seconds(stats.decodeMs)}</small></dd>
    </div>
  </dl>
{/if}

<style lang="scss">
  .metrics { min-width: 0; box-sizing: border-box; border: var(--gn-rule-width) solid var(--gn-border-subtle); border-radius: var(--gn-radius-sm); background: var(--gn-bg-panel); color: var(--gn-text-muted); font: var(--gn-text-xs) var(--gn-font-mono); font-variant-numeric: tabular-nums; }
  .metrics-live { min-height: var(--gn-control-height); display: flex; align-items: center; gap: var(--gn-space-xs); padding: var(--gn-space-xs) var(--gn-space-sm); font-weight: 700; }
  .phase-dot { width: 8px; height: 8px; flex: 0 0 auto; border-radius: 50%; background: var(--gn-text-muted); }
  .phase-prefill { background: var(--gn-streaming); }
  .phase-decode { background: var(--gn-accent); }
  .phase-label { color: var(--gn-text-primary); }
  .metrics-final { margin: 0; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); padding: var(--gn-space-xs) 0; }
  .metrics-final div { min-width: 0; display: grid; grid-template-columns: auto minmax(0, 1fr); align-items: baseline; gap: 2px var(--gn-space-sm); padding: 2px var(--gn-space-sm); border-left: var(--gn-rule-width) solid var(--gn-border-subtle); }
  .metrics-final div:first-child { border-left: 0; }
  dt { font-weight: 650; }
  dd { min-width: 0; margin: 0; display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 0 var(--gn-space-xs); text-align: right; overflow-wrap: anywhere; }
  strong { color: var(--gn-text-primary); font-size: var(--gn-text-sm); }
  .metric-primary strong { color: var(--gn-accent-ink); }
  small { font: inherit; white-space: nowrap; }
  @media (min-width: 901px) and (max-width: 1050px), (max-width: 640px) {
    .metrics-final { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .metrics-final div:nth-child(odd) { border-left: 0; }
    .metrics-final div:nth-child(n + 3) { border-top: var(--gn-rule-width) solid var(--gn-border-subtle); }
  }
</style>
