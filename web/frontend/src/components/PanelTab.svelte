<!--
PanelTab.svelte owns the compact closed-state control for one workspace panel.
Open content, mutual exclusion, overlay behavior, and persistence stay outside.
-->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import CollapseControl from './CollapseControl.svelte';

  export let side: 'left' | 'right';
  export let label: string;
  export let controls: string;
  export let count: number | null = null;
  export let element: HTMLButtonElement | undefined = undefined;

  const dispatch = createEventDispatcher<{ toggle: void }>();
  $: direction = side === 'left' ? 'right' as const : 'left' as const;
</script>

<div class="panel-tab {side}">
  <CollapseControl
    bind:element
    expanded={false}
    {controls}
    openLabel={`Open ${label}`}
    closeLabel={`Close ${label}`}
    expandDirection={direction}
    on:toggle={() => dispatch('toggle')}
  />
  {#if count !== null}<small aria-hidden="true">{count}</small>{/if}
</div>

<style lang="scss">
  .panel-tab {
    position: fixed;
    z-index: 9;
    top: calc(env(safe-area-inset-top) + var(--gn-space-xs));
    width: var(--gn-panel-rail-width);
    height: var(--gn-panel-rail-width);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .left { left: env(safe-area-inset-left); }
  .right { right: env(safe-area-inset-right); }
  small {
    position: absolute;
    right: calc(-1 * var(--gn-space-2xs));
    bottom: calc(-1 * var(--gn-space-2xs));
    min-width: 18px;
    border: var(--gn-rule-width) solid var(--gn-border);
    background: var(--gn-accent-soft);
    padding: 0 var(--gn-space-2xs);
    color: var(--gn-accent-ink);
    font: 700 var(--gn-text-xs) var(--gn-font-mono);
    line-height: 16px;
    text-align: center;
  }
  .right small { right: auto; left: calc(-1 * var(--gn-space-2xs)); }
</style>
