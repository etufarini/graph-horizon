<script lang="ts">
  /*
   * Runtime.svelte
   * Presents model and backend identity at a glance, with immutable placement
   * and planned allocation facts available through one compact disclosure.
   */
  import { formatBytes } from '../chat/telemetry';
  import type { RuntimeInfo, RuntimeMemory } from '../chat/types';

  export let info: RuntimeInfo;

  $: placement = info.placement;
  $: accelerator = info.backend.startsWith('metal') ? 'Metal' : 'GPU';
  $: mode = placement?.mode ?? 'homogeneous';

  function rows(memory: RuntimeMemory): Array<[string, string]> {
    return [
      ['Weights', formatBytes(memory.weights)], ['KV max', formatBytes(memory.kv)],
      ['Scratch', formatBytes(memory.scratch)], ['Fixed', formatBytes(memory.fixed)],
      ['Staging', formatBytes(memory.staging)], ['Crossing', formatBytes(memory.crossing)],
      ['Reserve', formatBytes(memory.reserve)]
    ];
  }
</script>

<section class="runtime" aria-label="Inference runtime">
  <div class="runtime-summary">
    <strong>{info.modelName}</strong><span>{info.backend}</span>
  </div>
  <details>
    <summary>Runtime details</summary>
    <dl class="runtime-overview">
      <div><dt>Mode</dt><dd>{mode}</dd></div>
      <div><dt>Weights</dt><dd>{formatBytes(info.memory.weights)}</dd></div>
      <div><dt>KV max</dt><dd>{formatBytes(info.memory.kv)}</dd></div>
      {#if placement}<div><dt>Placement</dt><dd>CPU {placement.cpuLayers}L · {accelerator} {placement.acceleratorLayers}L</dd></div>{/if}
    </dl>
    {#if placement}
      <div class="memory-owners">
        <section>
          <h2>CPU · budget {formatBytes(placement.cpu.total)}</h2>
          <dl>{#each rows(placement.cpu) as row}<div><dt>{row[0]}</dt><dd>{row[1]}</dd></div>{/each}</dl>
        </section>
        <section>
          <h2>{accelerator} · budget {formatBytes(placement.accelerator.total)}</h2>
          <dl>{#each rows(placement.accelerator) as row}<div><dt>{row[0]}</dt><dd>{row[1]}</dd></div>{/each}</dl>
        </section>
      </div>
    {/if}
  </details>
</section>

<style lang="scss">
  .runtime { min-width: 0; display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--gn-space-xs) var(--gn-space-sm); color: var(--gn-text-muted); font: var(--gn-text-xs) var(--gn-font-mono); font-variant-numeric: tabular-nums; }
  .runtime-summary { min-width: 0; display: flex; flex-wrap: wrap; gap: var(--gn-space-xs); }
  .runtime-summary strong { min-width: 0; color: var(--gn-text-primary); overflow-wrap: anywhere; }
  .runtime-summary span::before { content: "·"; margin-right: var(--gn-space-xs); }
  details { min-width: 0; }
  details[open] { flex-basis: 100%; }
  summary { width: fit-content; border-radius: var(--gn-radius-sm); color: var(--gn-accent-ink); cursor: pointer; font-weight: 650; }
  .runtime-overview { margin: var(--gn-space-sm) 0 0; display: flex; flex-wrap: wrap; gap: var(--gn-space-xs) var(--gn-space-lg); border-top: var(--gn-rule-width) solid var(--gn-border-subtle); padding-top: var(--gn-space-sm); }
  dl div { min-width: 0; }
  dt { color: var(--gn-text-muted); }
  dd { margin: 0; color: var(--gn-text-primary); overflow-wrap: anywhere; }
  .memory-owners { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--gn-space-md); margin-top: var(--gn-space-sm); }
  .memory-owners section { min-width: 0; border-top: var(--gn-rule-width) solid var(--gn-border-subtle); padding-top: var(--gn-space-sm); }
  h2 { margin: 0 0 var(--gn-space-xs); color: var(--gn-text-primary); font-size: var(--gn-text-xs); }
  .memory-owners dl { margin: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(84px, 1fr)); gap: var(--gn-space-xs) var(--gn-space-md); }
  @media (max-width: 640px) {
    .runtime { display: grid; gap: 2px; }
    summary { min-height: var(--gn-touch-height); display: flex; align-items: center; }
    .memory-owners { grid-template-columns: 1fr; }
  }
</style>
