<script lang="ts">
  /*
   * Metrics.svelte
   * Presents the current engine phase or the last exact generation statistics
   * above the composer. It owns only display timing and safe rate formatting;
   * stream ordering and lifecycle state remain in the chat modules.
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
    ? 'Attesa'
    : active?.phase === 'prefill' ? 'Prefill' : 'Decode';
  $: stats = telemetry?.stats;
  $: prefillRate = stats ? tokensPerSecond(stats.prefillTokens, stats.prefillMs) : null;
  $: decodeRate = stats ? tokensPerSecond(stats.completionTokens, stats.decodeMs) : null;

  const seconds = (milliseconds: number) => `${(milliseconds / 1000).toFixed(2).replace('.', ',')} s`;
  const rate = (value: number | null) => value === null ? '—' : `${value.toFixed(1).replace('.', ',')} tok/s`;
  onDestroy(() => { if (timer !== null) clearInterval(timer); });
</script>

{#if active}
  <section class="metrics metrics-live" aria-label="Fase di generazione">
    <span class="phase-dot phase-{active.phase}" aria-hidden="true"></span>
    <span class="phase-label" aria-live="polite">{phaseLabel}</span>
    <span aria-hidden="true">{seconds(elapsed)}</span>
  </section>
{:else if stats}
  <section class="metrics metrics-final" aria-label="Metriche dell'ultima generazione">
    <div><span>Prompt</span><strong>{stats.promptTokens} tok</strong></div>
    <div><span>Prefill</span><strong>{stats.prefillTokens} tok</strong><small>{seconds(stats.prefillMs)} · {rate(prefillRate)}</small></div>
    <div><span>Output</span><strong>{stats.completionTokens} tok</strong></div>
    <div><span>Decode</span><strong>{rate(decodeRate)}</strong><small>{seconds(stats.decodeMs)}</small></div>
  </section>
{/if}

<style lang="scss">
  .metrics { border: var(--gn-rule-width) solid var(--gn-border); background: var(--gn-bg-panel); padding: var(--gn-space-sm) var(--gn-space-md); color: var(--gn-text-muted); font-family: var(--gn-font-mono); font-size: var(--gn-text-xs); }
  .metrics-live { display: flex; align-items: center; gap: var(--gn-space-sm); font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }
  .phase-dot { width: 9px; height: 9px; flex: 0 0 auto; background: var(--gn-text-muted); }
  .phase-prefill { background: var(--gn-streaming); }
  .phase-decode { background: var(--gn-accent); }
  .phase-label { color: var(--gn-text-primary); }
  .metrics-final { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: var(--gn-space-md); }
  .metrics-final div { min-width: 0; display: grid; gap: 2px; }
  .metrics-final span { font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }
  .metrics-final strong { color: var(--gn-text-primary); font-size: var(--gn-text-sm); overflow-wrap: anywhere; }
  .metrics-final small { font-size: inherit; }
  @media (max-width: 720px) { .metrics-final { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
</style>
