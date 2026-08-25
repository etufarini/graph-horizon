<script lang="ts">
  /*
   * Presentational collapsible editor for the active chat's system prompt.
   * Owns only open state; value and streaming availability flow through props,
   * and edits leave through one typed event. Store and persistence stay outside.
   */
  import { createEventDispatcher } from 'svelte';

  export let value = '';
  export let disabled = false;
  // Open state is bindable so the parent can give the editor the full toolbar row.
  export let open = false;

  const dispatch = createEventDispatcher<{ change: string }>();

</script>

<div class="system-prompt">
  <button type="button" class="panel-header" aria-expanded={open} aria-controls="system-prompt-editor" on:click={() => (open = !open)}>
    <span class="chevron" class:open aria-hidden="true"></span>
    <span class="panel-label">System prompt</span>
    {#if !open && value.trim() !== ''}
      <span class="prompt-state">Set</span>
    {/if}
  </button>
  {#if open}
    <div id="system-prompt-editor" class="panel-body">
      <textarea
        bind:value
        {disabled}
        on:input={() => dispatch('change', value)}
        rows="3"
        aria-label="System prompt"
        placeholder="Instructions for the model…"
      ></textarea>
    </div>
  {/if}
</div>

<style lang="scss">
  .system-prompt {
    min-width: 0;
    border: var(--gn-rule-width) solid var(--gn-border-subtle);
    border-radius: var(--gn-radius-sm);
    background: var(--gn-bg-panel);
  }

  .panel-header {
    display: flex;
    align-items: center;
    gap: var(--gn-space-xs);
    width: 100%;
    min-height: var(--gn-control-height);
    padding: var(--gn-space-xs) var(--gn-space-sm);
    border: none;
    background: none;
    cursor: pointer;
    border-radius: var(--gn-radius-sm);
    font-family: var(--gn-font-sans);
    font-size: var(--gn-text-xs);
    font-weight: 650;
    color: var(--gn-text-muted);
    text-align: left;
  }

  .panel-header:hover { background: var(--gn-bg-panel-raised); color: var(--gn-text-primary); }

  .panel-header:focus-visible {
    outline: none;
    box-shadow: var(--gn-focus-ring);
  }

  .chevron {
    width: 0;
    height: 0;
    border-top: 4px solid transparent;
    border-bottom: 4px solid transparent;
    border-left: 5px solid var(--gn-text-muted);
    transition: transform var(--gn-motion-fast) ease;
  }

  .chevron.open {
    transform: rotate(90deg);
  }

  .prompt-state { margin-left: auto; border-radius: 999px; background: var(--gn-accent-soft); padding: 1px var(--gn-space-xs); color: var(--gn-accent-ink); font-size: var(--gn-text-xs); }

  .panel-body {
    padding: 0 var(--gn-space-sm) var(--gn-space-sm);
  }

  textarea {
    display: block;
    width: 100%;
    resize: vertical;
    box-sizing: border-box;
    border: var(--gn-rule-width) solid var(--gn-border);
    border-radius: var(--gn-radius-sm);
    outline: none;
    background: var(--gn-bg-panel);
    padding: var(--gn-space-sm) var(--gn-space-md);
    color: var(--gn-text-primary);
    font-family: var(--gn-font-sans);
    font-size: var(--gn-text-md);
    line-height: var(--gn-line-height);
  }

  textarea:focus {
    border-color: var(--gn-accent);
    box-shadow: var(--gn-focus-inset);
  }

  @media (max-width: 640px) {
    .panel-header { min-height: var(--gn-touch-height); }
  }
</style>
