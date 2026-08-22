<script lang="ts">
  /*
   * Runtime.svelte
   * Presents immutable model, backend, placement, and planned allocation data
   * in one compact header disclosure. Loading, validation, and inference state
   * remain outside this component.
   */
  import { formatBytes } from '../chat/telemetry';
  import type { RuntimeInfo, RuntimeMemory } from '../chat/types';

  export let info: RuntimeInfo;

  $: placement = info.placement;
  $: accelerator = info.backend.startsWith('metal') ? 'Metal' : 'GPU';
  $: mode = placement?.mode ?? 'homogeneous';

  function rows(memory: RuntimeMemory): Array<[string, string]> {
    return [
      ['Weights', formatBytes(memory.weights)],
      ['KV max', formatBytes(memory.kv)],
      ['Scratch', formatBytes(memory.scratch)],
      ['Fixed', formatBytes(memory.fixed)],
      ['Staging', formatBytes(memory.staging)],
      ['Crossing', formatBytes(memory.crossing)],
      ['Reserve', formatBytes(memory.reserve)]
    ];
  }
</script>

<section class="runtime" aria-label="Inference runtime">
  <div class="runtime-summary">
    <strong class="model">{info.modelName}</strong>
    <span>{info.backend}</span>
    <span>{mode}</span>
    <span>Weights {formatBytes(info.memory.weights)}</span>
    <span>KV max {formatBytes(info.memory.kv)}</span>
    {#if placement}
      <span>CPU {placement.cpuLayers}L / {accelerator} {placement.acceleratorLayers}L</span>
    {/if}
  </div>
  {#if placement}
    <details>
      <summary>
        Budget CPU {formatBytes(placement.cpu.total)} · {accelerator} {formatBytes(placement.accelerator.total)}
      </summary>
      <div class="memory-owners">
        <section class="memory-owner">
          <h2>CPU · budget {formatBytes(placement.cpu.total)}</h2>
          <dl>
            {#each rows(placement.cpu) as row}
              <div><dt>{row[0]}</dt><dd>{row[1]}</dd></div>
            {/each}
          </dl>
        </section>
        <section class="memory-owner">
          <h2>{accelerator} · budget {formatBytes(placement.accelerator.total)}</h2>
          <dl>
            {#each rows(placement.accelerator) as row}
              <div><dt>{row[0]}</dt><dd>{row[1]}</dd></div>
            {/each}
          </dl>
        </section>
      </div>
    </details>
  {/if}
</section>

<style lang="scss">
  .runtime { min-width: 0; display: grid; gap: var(--gn-space-xs); color: var(--gn-text-muted); font: 700 var(--gn-text-xs) var(--gn-font-mono); letter-spacing: 0.06em; text-transform: uppercase; }
  .runtime-summary { min-width: 0; display: flex; flex-wrap: wrap; gap: var(--gn-space-xs) var(--gn-space-md); align-items: baseline; }
  .runtime-summary span::before { content: "·"; margin-right: var(--gn-space-md); color: var(--gn-border); }
  .model { min-width: 0; color: var(--gn-text-primary); overflow-wrap: anywhere; }
  details { border-left: var(--gn-rule-width) solid var(--gn-border); padding-left: var(--gn-space-sm); }
  summary { width: fit-content; cursor: pointer; color: var(--gn-accent-ink); }
  summary:focus-visible { outline: none; box-shadow: var(--gn-focus-ring); }
  .memory-owners { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--gn-space-md); margin-top: var(--gn-space-sm); }
  .memory-owner { border: var(--gn-rule-width) solid var(--gn-border); background: var(--gn-bg-panel); padding: var(--gn-space-sm); }
  h2 { margin: 0 0 var(--gn-space-sm); color: var(--gn-text-primary); font-size: var(--gn-text-xs); }
  dl { margin: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(92px, 1fr)); gap: var(--gn-space-xs) var(--gn-space-md); text-transform: none; letter-spacing: 0; }
  dl div { min-width: 0; }
  dt { color: var(--gn-text-muted); }
  dd { margin: 0; color: var(--gn-text-primary); }
  @media (max-width: 720px) {
    .runtime-summary { display: grid; gap: 2px; }
    .runtime-summary span::before { content: none; }
    .memory-owners { grid-template-columns: 1fr; }
  }
</style>
