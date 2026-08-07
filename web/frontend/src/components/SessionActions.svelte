<script lang="ts">
  /*
   * SessionActions.svelte — presentational session action bar (props in,
   * events out): new-chat/export/import controls and their confirm/file-picker
   * choreography. Parsing and state/storage mutations remain outside; this
   * component never reads the store or localStorage.
   */
  import { createEventDispatcher } from 'svelte';

  export let importDisabled = false;
  export let confirmBeforeImport = false;
  export let hasMessages = false;

  const dispatch = createEventDispatcher<{ reset: void; export: void; import: string }>();

  let picker: HTMLInputElement;

  function reset(): void {
    if (hasMessages && !confirm('Iniziare una nuova chat? La conversazione corrente verrà eliminata.')) {
      return;
    }
    dispatch('reset');
  }

  async function selected(): Promise<void> {
    const file = picker.files?.[0];
    if (!file) {
      return;
    }
    try {
      // The confirm gate runs before the file is read: a declined confirm
      // means the file content is never touched. The prop value at
      // selection time decides whether the dialog appears.
      if (confirmBeforeImport && !confirm('Sostituire la conversazione corrente? Quella attuale andrà persa.')) {
        return;
      }
      const text = await file.text();
      dispatch('import', text);
    } catch {
      // Read failure: emit nothing; retry is the recovery path.
    } finally {
      // Always reset so re-selecting the same file fires change again.
      picker.value = '';
    }
  }
</script>

<div class="session-actions">
  <button type="button" disabled={importDisabled} on:click={reset}>Nuova chat</button>
  <button type="button" on:click={() => dispatch('export')}>Esporta</button>
  <button type="button" disabled={importDisabled} on:click={() => picker.click()}>Importa</button>
  <input
    type="file"
    accept=".json,application/json"
    bind:this={picker}
    on:change={selected}
    aria-hidden="true"
    tabindex="-1"
  />
</div>

<style lang="scss">
  .session-actions {
    display: flex;
    justify-content: flex-end;
    gap: var(--gn-space-sm);
  }

  /* Same mono/uppercase label idiom as the panel headers. */
  button {
    border: var(--gn-border-width) solid var(--gn-border);
    border-radius: var(--gn-radius-sm);
    background: var(--gn-bg-panel);
    padding: var(--gn-space-xs) var(--gn-space-sm);
    cursor: pointer;
    font-family: var(--gn-font-mono);
    font-size: var(--gn-text-xs);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--gn-text-muted);
    box-shadow: var(--gn-shadow-small);
  }

  button:hover:not(:disabled) {
    border-color: var(--gn-accent-ink);
    color: var(--gn-accent-ink);
  }

  button:focus-visible {
    outline: none;
    box-shadow: var(--gn-focus-ring), var(--gn-shadow-small);
  }

  button:disabled {
    background: var(--gn-bg-panel-raised);
    color: var(--gn-text-muted);
    box-shadow: none;
    cursor: default;
  }

  /* Visually hidden but still clickable programmatically. */
  input {
    display: none;
  }
</style>
