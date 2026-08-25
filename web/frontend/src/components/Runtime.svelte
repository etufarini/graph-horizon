<script lang="ts">
  /*
   * Runtime.svelte
   * Presents model and backend identity at a glance, with immutable placement
   * and planned allocation facts available through one compact disclosure.
   */
  import { formatBytes } from '../chat/telemetry';
  import type { RuntimeInfo, RuntimeMemory } from '../chat/types';

  export let info: RuntimeInfo;
  let open = false;
  let summary: HTMLElement;

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

  function keydown(event: KeyboardEvent): void {
    if (event.key === 'Escape' && open) {
      event.preventDefault();
      open = false;
      summary.focus();
    }
  }
</script>

<section class="runtime" aria-label="Inference runtime">
  <div class="runtime-summary">
    <strong>{info.modelName}</strong><span>{info.backend}</span>
  </div>
  <details bind:open>
    <summary bind:this={summary} on:keydown={keydown}>Runtime details</summary>
    <div class="runtime-details">
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
    </div>
  </details>
</section>

<style lang="scss">
  .runtime { position: relative; min-width: 0; display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--gn-space-xs) var(--gn-space-sm); color: var(--gn-text-muted); font: var(--gn-text-xs) var(--gn-font-mono); font-variant-numeric: tabular-nums; }
  .runtime-summary { min-width: 0; display: flex; flex-wrap: wrap; gap: var(--gn-space-xs); }
  .runtime-summary strong { min-width: 0; color: var(--gn-text-primary); overflow-wrap: anywhere; }
  .runtime-summary span::before { content: "·"; margin-right: var(--gn-space-xs); }
  details { min-width: 0; }
  summary { width: fit-content; border-radius: var(--gn-radius-sm); color: var(--gn-accent-ink); cursor: pointer; font-weight: 650; }
  .runtime-details { position: absolute; z-index: 12; top: calc(100% + var(--gn-space-sm)); right: 0; width: min(560px, calc(100vw - var(--gn-space-xl))); max-height: min(60dvh, 480px); overflow: auto; border: var(--gn-border-width) solid var(--gn-border); background: var(--gn-bg-panel); box-shadow: var(--gn-shadow-hard); padding: var(--gn-space-md); }
  .runtime-overview { margin: 0; display: flex; flex-wrap: wrap; gap: var(--gn-space-xs) var(--gn-space-lg); }
  dl div { min-width: 0; }
  dt { color: var(--gn-text-muted); }
  dd { margin: 0; color: var(--gn-text-primary); overflow-wrap: anywhere; }
  .memory-owners { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--gn-space-md); margin-top: var(--gn-space-sm); }
  .memory-owners section { min-width: 0; border-top: var(--gn-rule-width) solid var(--gn-border-subtle); padding-top: var(--gn-space-sm); }
  h2 { margin: 0 0 var(--gn-space-xs); color: var(--gn-text-primary); font-size: var(--gn-text-xs); }
  .memory-owners dl { margin: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(84px, 1fr)); gap: var(--gn-space-xs) var(--gn-space-md); }
  @media (max-width: 640px) {
    .runtime { display: grid; gap: var(--gn-space-2xs); }
    summary { min-height: var(--gn-touch-height); display: flex; align-items: center; }
    .runtime-details { position: fixed; top: calc(env(safe-area-inset-top) + var(--gn-space-sm)); right: calc(env(safe-area-inset-right) + var(--gn-space-sm)); left: calc(env(safe-area-inset-left) + var(--gn-space-sm)); width: auto; max-height: calc(100dvh - var(--gn-space-lg)); }
    .memory-owners { grid-template-columns: 1fr; }
  }
</style>
