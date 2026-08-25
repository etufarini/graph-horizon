<!--
SystemPrompt.svelte presents the collapsible editor for one chat. It owns only
open state; value, availability, persistence, and chat selection stay outside.
-->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import CollapseControl from './CollapseControl.svelte';

  export let value = '';
  export let disabled = false;
  // Open state is bindable so the parent can give the editor the full toolbar row.
  export let open = false;

  const dispatch = createEventDispatcher<{ change: string }>();

</script>

<div class="system-prompt">
  <div class="panel-header">
    <span class="panel-label">System prompt</span>
    {#if !open && value.trim() !== ''}
      <span class="prompt-state">Set</span>
    {/if}
    <CollapseControl
      expanded={open}
      controls="system-prompt-editor"
      openLabel="Open system prompt"
      closeLabel="Close system prompt"
      on:toggle={() => (open = !open)}
    />
  </div>
  <div id="system-prompt-editor" class="panel-body" hidden={!open}>
    <textarea
      bind:value
      {disabled}
      on:input={() => dispatch('change', value)}
      rows="3"
      aria-label="System prompt"
      placeholder="Instructions for the model…"
    ></textarea>
  </div>
</div>

<style lang="scss">
  .system-prompt {
    min-width: 0;
    border: var(--gn-border-width) solid var(--gn-border);
    border-radius: var(--gn-radius-sm);
    background: var(--gn-bg-panel);
    box-shadow: var(--gn-shadow-small);
  }

  .panel-header {
    display: flex;
    align-items: center;
    gap: var(--gn-space-xs);
    width: 100%;
    box-sizing: border-box;
    min-height: var(--gn-control-height);
    padding: var(--gn-space-xs) var(--gn-space-sm);
    font-family: var(--gn-font-mono);
    font-size: var(--gn-text-xs);
    font-weight: 650;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--gn-text-muted);
    text-align: left;
  }

  .panel-label { color: var(--gn-text-primary); }
  .prompt-state { margin-left: auto; border: var(--gn-rule-width) solid var(--gn-accent); border-radius: 0; background: var(--gn-accent-soft); padding: 1px var(--gn-space-xs); color: var(--gn-accent-ink); font-size: var(--gn-text-xs); }
  .panel-header :global(button) { margin-left: auto; }

  .panel-body {
    padding: 0 var(--gn-space-sm) var(--gn-space-sm);
  }

  [hidden] { display: none; }

  textarea {
    display: block;
    width: 100%;
    resize: vertical;
    box-sizing: border-box;
    border: var(--gn-border-width) solid var(--gn-border);
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
