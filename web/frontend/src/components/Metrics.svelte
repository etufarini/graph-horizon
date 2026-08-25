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
  <details class="metrics metrics-final">
    <summary>
      <span><small>Output</small><strong>{stats.completionTokens} tok</strong></span>
      <span><small>Decode</small><strong aria-label={decodeRate === null ? 'Decode rate unavailable' : undefined}>{rate(decodeRate)}</strong><em>{seconds(stats.decodeMs)}</em></span>
      <b>Details</b>
    </summary>
    <dl>
      <div><dt>Prompt</dt><dd><strong>{stats.promptTokens} tok</strong></dd></div>
      <div>
        <dt>Prefill</dt>
        <dd><strong>{stats.prefillTokens} tok</strong><small class="rate-details"><span>{seconds(stats.prefillMs)}</span><span>· {rate(prefillRate)}</span></small></dd>
      </div>
    </dl>
  </details>
{/if}

<style lang="scss">
  .metrics { min-width: 0; box-sizing: border-box; color: var(--gn-text-muted); font: var(--gn-text-xs) var(--gn-font-mono); font-variant-numeric: tabular-nums; }
  .metrics-live { min-height: var(--gn-control-height); display: flex; align-items: center; gap: var(--gn-space-xs); padding: var(--gn-space-xs) var(--gn-space-sm); font-weight: 700; }
  .phase-dot { width: 8px; height: 8px; flex: 0 0 auto; border-radius: 0; background: var(--gn-text-muted); }
  .phase-prefill { background: var(--gn-streaming); }
  .phase-decode { background: var(--gn-accent); }
  .phase-label { color: var(--gn-text-primary); }
  .metrics-final { container-type: inline-size; }
  summary { min-height: var(--gn-control-height); display: flex; align-items: center; gap: var(--gn-space-sm); padding: var(--gn-space-xs) var(--gn-space-sm); cursor: pointer; list-style: none; }
  summary::-webkit-details-marker { display: none; }
  summary > span { min-width: 0; display: flex; align-items: baseline; gap: var(--gn-space-xs); }
  summary > span + span { border-left: var(--gn-rule-width) solid var(--gn-border-subtle); padding-left: var(--gn-space-sm); }
  summary small, summary em { font: inherit; }
  summary small { font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }
  summary em { font-style: normal; white-space: nowrap; }
  summary b { margin-left: auto; color: var(--gn-accent-ink); font-weight: 650; }
  summary b::after { content: " ▾"; }
  details[open] summary b::after { content: " ▴"; }
  .metrics-final dl { margin: 0; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); border-top: var(--gn-rule-width) solid var(--gn-border-subtle); padding: var(--gn-space-xs) 0; }
  .metrics-final div { min-width: 0; display: grid; grid-template-columns: auto minmax(0, 1fr); align-items: baseline; gap: var(--gn-space-2xs) var(--gn-space-sm); padding: var(--gn-space-2xs) var(--gn-space-sm); border-left: var(--gn-rule-width) solid var(--gn-border-subtle); }
  .metrics-final div:first-child { border-left: 0; }
  dt { font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }
  dd { min-width: 0; max-width: 100%; margin: 0; display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 0 var(--gn-space-xs); text-align: right; }
  strong { color: var(--gn-text-primary); font-size: var(--gn-text-sm); white-space: nowrap; }
  summary strong { color: var(--gn-accent-ink); }
  small { min-width: 0; max-width: 100%; font: inherit; white-space: nowrap; }
  .rate-details { display: inline-flex; flex-wrap: wrap; justify-content: flex-end; column-gap: var(--gn-space-xs); white-space: normal; }
  .rate-details span { white-space: nowrap; }
  @container (max-width: 419px) {
    summary { flex-wrap: wrap; }
    summary b { margin-left: 0; }
    .metrics-final dl { grid-template-columns: 1fr; }
    .metrics-final div { grid-template-columns: 1fr; border-left: 0; }
    .metrics-final div:nth-child(n + 2) { border-top: var(--gn-rule-width) solid var(--gn-border-subtle); }
    dd, .rate-details { justify-content: flex-start; text-align: left; }
  }
</style>
