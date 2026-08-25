<!--
SessionActions.svelte presents active-chat import/export controls and owns only
exception-safe file selection. Parsing, collection mutation, persistence, and
replacement confirmation remain outside this props-in/events-out component.
-->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  export let importDisabled = false;
  const dispatch = createEventDispatcher<{ export: void; import: string }>();
  let picker: HTMLInputElement;
  let disclosure: HTMLDetailsElement;
  let summary: HTMLElement;

  function exportChat(): void {
    dispatch('export');
    disclosure.open = false;
    summary.focus();
  }

  async function selected(): Promise<void> {
    const file = picker.files?.[0];
    if (!file) return;
    try {
      dispatch('import', await file.text());
    } catch {
      // Read failure emits nothing; selecting the same file is the retry path.
    } finally {
      picker.value = '';
      disclosure.open = false;
      summary.focus();
    }
  }

  function keydown(event: KeyboardEvent): void {
    if (event.key === 'Escape' && disclosure.open) {
      event.preventDefault();
      disclosure.open = false;
      summary.focus();
    }
  }
</script>

<div class="session-actions">
  <details bind:this={disclosure}>
    <summary bind:this={summary} on:keydown={keydown}>Chat data</summary>
    <div class="action-menu">
      <button type="button" on:click={exportChat} on:keydown={keydown}>Export</button>
      <button type="button" disabled={importDisabled} on:click={() => picker.click()} on:keydown={keydown}>Import</button>
    </div>
  </details>
  <input type="file" accept=".json,application/json" bind:this={picker} on:change={selected} aria-hidden="true" tabindex="-1" />
</div>

<style lang="scss">
  .session-actions {
    display: flex;
    justify-content: flex-end;
  }
  details { position: relative; }
  summary, button {
    min-height: var(--gn-control-height);
    border: var(--gn-rule-width) solid var(--gn-border);
    background: var(--gn-bg-panel);
    padding: var(--gn-space-xs) var(--gn-space-sm);
    cursor: pointer;
    color: var(--gn-text-muted);
    font-family: var(--gn-font-sans);
    font-size: var(--gn-text-xs);
    font-weight: 650;
  }
  summary { list-style: none; border-width: var(--gn-border-width); box-shadow: var(--gn-shadow-small); font-family: var(--gn-font-mono); }
  summary::-webkit-details-marker { display: none; }
  summary::after { content: "▾"; margin-left: var(--gn-space-sm); }
  details[open] summary::after { content: "▴"; }
  summary:hover, button:hover:not(:disabled) { border-color: var(--gn-accent); background: var(--gn-accent-soft); color: var(--gn-accent-ink); }
  summary:focus-visible, button:focus-visible { outline: none; box-shadow: var(--gn-focus-ring); }
  .action-menu {
    position: absolute; z-index: 12; top: calc(100% + var(--gn-space-xs)); right: 0;
    display: flex; gap: var(--gn-space-xs);
    border: var(--gn-border-width) solid var(--gn-border); background: var(--gn-bg-panel);
    box-shadow: var(--gn-shadow-hard); padding: var(--gn-space-xs);
  }
  button:disabled {
    background: var(--gn-bg-panel-raised);
    color: var(--gn-text-muted);
    box-shadow: none;
    cursor: default;
  }
  input { display: none; }
  @media (max-width: 640px) {
    summary, button { min-height: var(--gn-touch-height); }
  }
</style>
