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

  async function selected(): Promise<void> {
    const file = picker.files?.[0];
    if (!file) return;
    try {
      dispatch('import', await file.text());
    } catch {
      // Read failure emits nothing; selecting the same file is the retry path.
    } finally {
      picker.value = '';
    }
  }
</script>

<div class="session-actions">
  <button type="button" on:click={() => dispatch('export')}>Esporta</button>
  <button type="button" disabled={importDisabled} on:click={() => picker.click()}>Importa</button>
  <input type="file" accept=".json,application/json" bind:this={picker} on:change={selected} aria-hidden="true" tabindex="-1" />
</div>

<style lang="scss">
  .session-actions {
    display: flex;
    justify-content: flex-end;
    gap: var(--gn-space-sm);
  }
  button {
    border: var(--gn-border-width) solid var(--gn-border);
    border-radius: var(--gn-radius-sm);
    background: var(--gn-bg-panel);
    padding: var(--gn-space-xs) var(--gn-space-sm);
    cursor: pointer;
    color: var(--gn-text-muted);
    box-shadow: var(--gn-shadow-small);
    font-family: var(--gn-font-mono);
    font-size: var(--gn-text-xs);
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  button:hover:not(:disabled) { border-color: var(--gn-accent-ink); color: var(--gn-accent-ink); }
  button:focus-visible { outline: none; box-shadow: var(--gn-focus-ring), var(--gn-shadow-small); }
  button:disabled {
    background: var(--gn-bg-panel-raised);
    color: var(--gn-text-muted);
    box-shadow: none;
    cursor: default;
  }
  input { display: none; }
</style>
