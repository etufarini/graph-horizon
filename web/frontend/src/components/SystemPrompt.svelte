<!--
SystemPrompt.svelte presents the collapsible editor for one chat. It owns only
open state; value, availability, persistence, and chat selection stay outside.
-->
<script lang="ts">
  import { createEventDispatcher, tick } from 'svelte';

  export let value = '';
  export let disabled = false;
  // Open state is bindable so the parent can give the editor the full toolbar row.
  export let open = false;

  const dispatch = createEventDispatcher<{ change: string }>();
  let trigger: HTMLButtonElement;
  let editor: HTMLTextAreaElement;

  async function show(): Promise<void> {
    open = true;
    await tick();
    editor?.focus();
  }

  async function close(): Promise<void> {
    open = false;
    await tick();
    trigger?.focus();
  }

  function keydown(event: KeyboardEvent): void {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      void close();
    }
  }
</script>

<div class:open class="system-prompt">
  {#if open}
    <section id="system-prompt-editor" aria-label="System prompt editor">
      <header>
        <div class="panel-title">
          <strong>System prompt</strong>
          <small>Saved with this chat</small>
        </div>
        <button class="close" type="button" title="Close system prompt" aria-label="Close system prompt" on:click={close} on:keydown={keydown}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
            <path d="M3 3l10 10M13 3L3 13" />
          </svg>
        </button>
      </header>
      <div class="panel-body">
        <textarea
          bind:this={editor}
          bind:value
          {disabled}
          on:input={() => dispatch('change', value)}
          on:keydown={keydown}
          rows="3"
          aria-label="System prompt"
          placeholder="Instructions for the model…"
        ></textarea>
      </div>
    </section>
  {:else}
    <button bind:this={trigger} class="prompt-trigger" type="button" aria-expanded="false"
      aria-controls="system-prompt-editor" on:click={show}>
      <span class="panel-label">System prompt</span>
      {#if value.trim() !== ''}<span class="prompt-state">Set</span>{/if}
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
        <path d="m5 3 5 5-5 5" />
      </svg>
    </button>
  {/if}
</div>

<style lang="scss">
  .system-prompt { min-width: 0; }
  .system-prompt.open {
    border: var(--gn-border-width) solid var(--gn-border);
    background: var(--gn-bg-panel);
    box-shadow: var(--gn-shadow-small);
  }
  .prompt-trigger, header {
    width: 100%; min-width: 0; min-height: var(--gn-control-height);
    display: flex;
    align-items: center;
    gap: var(--gn-space-sm);
    padding: var(--gn-space-xs) var(--gn-space-sm);
  }
  .prompt-trigger {
    border: var(--gn-border-width) solid var(--gn-border);
    background: var(--gn-bg-panel);
    box-shadow: var(--gn-shadow-small);
    color: var(--gn-text-muted);
    cursor: pointer;
    font: 650 var(--gn-text-xs) var(--gn-font-mono);
    letter-spacing: 0.06em;
    text-align: left;
    text-transform: uppercase;
  }
  .prompt-trigger:hover { border-color: var(--gn-accent); background: var(--gn-accent-soft); color: var(--gn-accent-ink); }
  .prompt-trigger:focus-visible, .close:focus-visible { outline: none; box-shadow: var(--gn-focus-ring); }
  .panel-label { min-width: 0; flex: 1 1 auto; overflow: hidden; color: var(--gn-text-primary); text-overflow: ellipsis; white-space: nowrap; }
  .prompt-state {
    flex: 0 0 auto; border: var(--gn-rule-width) solid var(--gn-accent);
    background: var(--gn-accent-soft); padding: 1px var(--gn-space-xs);
    color: var(--gn-accent-ink); font-size: var(--gn-text-xs);
  }
  header { border-bottom: var(--gn-rule-width) solid var(--gn-border-subtle); }
  .panel-title { min-width: 0; flex: 1 1 auto; display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--gn-space-xs) var(--gn-space-sm); }
  strong { color: var(--gn-text-primary); font: 700 var(--gn-text-sm) var(--gn-font-mono); letter-spacing: 0.06em; text-transform: uppercase; }
  small { color: var(--gn-text-muted); font-size: var(--gn-text-xs); }
  .close {
    width: var(--gn-control-height); height: var(--gn-control-height); flex: 0 0 auto;
    display: inline-flex; align-items: center; justify-content: center;
    border: var(--gn-border-width) solid var(--gn-border); background: var(--gn-bg-panel);
    box-shadow: var(--gn-shadow-small); color: var(--gn-text-primary); cursor: pointer;
  }
  .close:hover { border-color: var(--gn-accent); background: var(--gn-accent-soft); color: var(--gn-accent-ink); }
  .close:active { transform: translate(2px, 2px); box-shadow: none; }
  .panel-body { padding: var(--gn-space-md); }
  textarea {
    display: block; width: 100%; min-height: 72px; max-height: 30dvh;
    resize: vertical; overflow: auto;
    border: var(--gn-rule-width) solid var(--gn-border); outline: none; background: var(--gn-bg-panel);
    padding: var(--gn-space-sm) var(--gn-space-md);
    color: var(--gn-text-primary); font: var(--gn-text-md)/var(--gn-line-height) var(--gn-font-sans);
  }
  textarea:focus { border-color: var(--gn-accent); box-shadow: var(--gn-focus-inset); }
  @media (max-width: 640px) {
    .prompt-trigger, header { min-height: var(--gn-touch-height); }
    .close { width: var(--gn-touch-height); height: var(--gn-touch-height); }
    .panel-title { display: grid; gap: 0; }
  }
</style>
