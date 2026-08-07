<!--
Turn.svelte presents one complete user/assistant pair through Bubble, owns only
the final-prompt edit draft and delete confirmation, and emits typed intents.
Transcript mutation, transport, persistence, and chat navigation are excluded.
-->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Bubble from './Bubble.svelte';
  import type { ChatMessage } from '../chat/types';

  export let user: ChatMessage;
  export let assistant: ChatMessage;
  export let final = false;
  export let streaming = false;

  const dispatch = createEventDispatcher<{
    regenerate: void;
    edit: string;
    delete: void;
  }>();
  let editing = false;
  let draft = '';
  $: validDraft = draft.trim().length > 0;

  function edit(): void {
    draft = user.content;
    editing = true;
  }

  function cancel(): void {
    draft = '';
    editing = false;
  }

  function save(): void {
    if (!streaming && validDraft) {
      dispatch('edit', draft);
      editing = false;
    }
  }

  function keydown(event: KeyboardEvent): void {
    if (event.key === 'Escape') {
      event.preventDefault();
      cancel();
    }
  }

  function remove(): void {
    if (confirm('Eliminare l’ultimo turno? Il messaggio e la risposta verranno rimossi.')) {
      dispatch('delete');
    }
  }
</script>

<div class="turn">
  <div class="message message-user">
    {#if final && editing}
      <article class="editor">
        <label for={`turn-${user.id}`}>Tu</label>
        <textarea id={`turn-${user.id}`} bind:value={draft} disabled={streaming} on:keydown={keydown}></textarea>
        <div class="actions">
          <button type="button" disabled={streaming || !validDraft} on:click={save}>Salva e rigenera</button>
          <button type="button" disabled={streaming} on:click={cancel}>Annulla</button>
        </div>
      </article>
    {:else}
      <Bubble message={user} />
      {#if final}
        <div class="actions actions-user">
          <button type="button" disabled={streaming} on:click={edit}>Modifica</button>
          <button class="destructive" type="button" disabled={streaming} on:click={remove}>Elimina</button>
        </div>
      {/if}
    {/if}
  </div>

  <div class="message message-assistant">
    <Bubble message={assistant} streaming={streaming && final} />
    {#if final}
      <div class="actions">
        <button type="button" disabled={streaming} on:click={() => dispatch('regenerate')}>Rigenera</button>
      </div>
    {/if}
  </div>
</div>

<style lang="scss">
  .turn {
    display: flex;
    flex-direction: column;
    gap: var(--gn-space-md);
  }

  .message {
    display: flex;
    flex-direction: column;
    gap: var(--gn-space-xs);
  }

  .message-user,
  .actions-user {
    align-items: flex-end;
  }

  .actions {
    display: flex;
    gap: var(--gn-space-sm);
  }

  button {
    border: var(--gn-border-width) solid var(--gn-border);
    border-radius: var(--gn-radius-sm);
    background: var(--gn-bg-panel);
    padding: var(--gn-space-xs) var(--gn-space-sm);
    color: var(--gn-text-muted);
    box-shadow: var(--gn-shadow-small);
    cursor: pointer;
    font-family: var(--gn-font-mono);
    font-size: var(--gn-text-xs);
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
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

  button.destructive:not(:disabled) {
    border-color: var(--gn-error-border);
    background: var(--gn-error-bg);
    color: var(--gn-error-fg);
  }

  .editor {
    width: min(78%, 760px);
    box-sizing: border-box;
    border: var(--gn-border-width) solid var(--gn-user-border);
    background: var(--gn-user-bg);
    padding: var(--gn-space-sm) var(--gn-space-md);
    box-shadow: var(--gn-shadow-hard);
  }

  label {
    display: block;
    margin-bottom: var(--gn-space-xs);
    color: var(--gn-text-muted);
    font-family: var(--gn-font-mono);
    font-size: var(--gn-text-xs);
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  textarea {
    width: 100%;
    min-height: 96px;
    box-sizing: border-box;
    margin-bottom: var(--gn-space-sm);
    border: var(--gn-border-width) solid var(--gn-border);
    border-radius: var(--gn-radius-sm);
    background: var(--gn-bg-panel);
    padding: var(--gn-space-sm);
    color: var(--gn-text-primary);
    font: inherit;
    resize: vertical;
  }

  textarea:focus-visible {
    outline: none;
    box-shadow: var(--gn-focus-inset);
  }

  @media (max-width: 720px) {
    .editor {
      width: 100%;
    }
  }
</style>
