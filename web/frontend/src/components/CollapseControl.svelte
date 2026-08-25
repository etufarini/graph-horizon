<!--
CollapseControl.svelte owns the shared accessible button, tooltip, and chevron
used to expand or collapse one spatial panel. Panel layout and state stay with
the component that contains this control.
-->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  type Direction = 'up' | 'right' | 'down' | 'left';

  export let expanded = false;
  export let controls: string;
  export let openLabel: string;
  export let closeLabel: string;
  export let expandDirection: Direction = 'down';
  export let collapseDirection: Direction = 'up';
  export let element: HTMLButtonElement | undefined = undefined;

  const dispatch = createEventDispatcher<{ toggle: void }>();
  $: label = expanded ? closeLabel : openLabel;
  $: direction = expanded ? collapseDirection : expandDirection;
</script>

<button
  bind:this={element}
  type="button"
  aria-label={label}
  aria-expanded={expanded}
  aria-controls={controls}
  title={label}
  on:click={() => dispatch('toggle')}
>
  <svg class="direction-{direction}" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
    <path d="m9 18 6-6-6-6" />
  </svg>
</button>

<style lang="scss">
  button {
    width: var(--gn-control-height);
    height: var(--gn-control-height);
    flex: 0 0 auto;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border: var(--gn-border-width) solid var(--gn-border);
    border-radius: var(--gn-radius-sm);
    background: var(--gn-bg-panel);
    box-shadow: var(--gn-shadow-small);
    color: var(--gn-text-primary);
    cursor: pointer;
  }

  button:hover {
    border-color: var(--gn-accent);
    background: var(--gn-accent-soft);
    color: var(--gn-accent-ink);
  }

  button:focus-visible {
    outline: none;
    box-shadow: var(--gn-focus-ring);
  }

  button:active {
    transform: translate(2px, 2px);
    box-shadow: none;
  }

  svg { transition: transform var(--gn-motion-fast) ease; }
  .direction-down { transform: rotate(90deg); }
  .direction-left { transform: rotate(180deg); }
  .direction-up { transform: rotate(270deg); }

  @media (max-width: 640px) {
    button { width: var(--gn-touch-height); height: var(--gn-touch-height); }
  }

  @media (pointer: coarse) {
    button { width: var(--gn-touch-height); height: var(--gn-touch-height); }
  }
</style>
