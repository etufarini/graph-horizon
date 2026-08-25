<!--
PanelRail.svelte owns the visible closed state for one left or right workspace
panel. Open panel content, overlay coordination, and persistence remain outside.
-->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import CollapseControl from './CollapseControl.svelte';

  export let side: 'left' | 'right';
  export let label: string;
  export let text = label;
  export let controls: string;
  export let count: number | null = null;
  export let overlay = false;
  export let element: HTMLButtonElement | undefined = undefined;

  const dispatch = createEventDispatcher<{ toggle: void }>();
  $: direction = side === 'left' ? 'right' as const : 'left' as const;
</script>

<div class="rail {side}" class:overlay aria-label={`${label} collapsed`}>
  <CollapseControl
    bind:element
    expanded={false}
    {controls}
    openLabel={`Open ${label}`}
    closeLabel={`Close ${label}`}
    expandDirection={direction}
    on:toggle={() => dispatch('toggle')}
  />
  <span aria-hidden="true">{text}</span>
  {#if count !== null}<small aria-hidden="true">{count}</small>{/if}
</div>

<style lang="scss">
  .rail {
    width: 100%;
    height: 100%;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--gn-space-sm);
    border: var(--gn-rule-width) solid var(--gn-border-subtle);
    background: var(--gn-bg-panel);
    padding: var(--gn-space-xs);
  }
  span { writing-mode: vertical-rl; color: var(--gn-text-muted); font: 700 var(--gn-text-xs) var(--gn-font-mono); letter-spacing: 0.08em; text-transform: uppercase; }
  .left span { transform: rotate(180deg); }
  small { color: var(--gn-accent-ink); font: 700 var(--gn-text-xs) var(--gn-font-mono); }
  .overlay { position: fixed; z-index: 9; top: 50%; width: var(--gn-panel-rail-width); height: auto; transform: translateY(-50%); }
  .overlay.left { left: 0; border-left: 0; }
  .overlay.right { right: 0; border-right: 0; }
</style>
