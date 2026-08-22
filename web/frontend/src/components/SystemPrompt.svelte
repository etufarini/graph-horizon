<script lang="ts">
  /*
   * Presentational collapsible editor for the active chat's system prompt.
   * Owns only open state; value and streaming availability flow through props,
   * and edits leave through one typed event. Store and persistence stay outside.
   */
  import { createEventDispatcher } from 'svelte';

  export let value = '';
  export let disabled = false;

  const dispatch = createEventDispatcher<{ change: string }>();

  // Collapsed by default on every load; open/closed state is not persisted.
  let open = false;
</script>

<div class="system-prompt">
  <button type="button" class="panel-header" aria-expanded={open} on:click={() => (open = !open)}>
    <span class="chevron" class:open aria-hidden="true"></span>
    <span class="panel-label">System prompt</span>
    {#if !open && value.trim() !== ''}
      <!-- Static dot (no pulse): signals a set prompt, not activity. -->
      <span class="dot" aria-hidden="true"></span>
    {/if}
  </button>
  {#if open}
    <div class="panel-body">
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
    border: var(--gn-border-width) solid var(--gn-border);
    border-radius: var(--gn-radius-sm);
    background: var(--gn-bg-panel);
    box-shadow: var(--gn-shadow-hard);
  }

  .panel-header {
    display: flex;
    align-items: center;
    gap: var(--gn-space-xs);
    width: 100%;
    padding: var(--gn-space-xs) var(--gn-space-sm);
    border: none;
    background: none;
    cursor: pointer;
    font-family: var(--gn-font-mono);
    font-size: var(--gn-text-xs);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--gn-text-muted);
    text-align: left;
  }

  .panel-header:focus-visible {
    outline: none;
    box-shadow: var(--gn-focus-inset);
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

  .dot {
    width: 0.5em;
    height: 0.5em;
    margin-left: var(--gn-space-xs);
    background: var(--gn-accent);
  }

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
    border-color: var(--gn-border);
    box-shadow: var(--gn-focus-inset);
  }
</style>
