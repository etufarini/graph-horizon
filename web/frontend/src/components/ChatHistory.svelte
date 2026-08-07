<!--
ChatHistory.svelte presents the saved-chat sidebar/drawer, row menus, inline
rename draft, confirmations, and keyboard choreography as typed intents.
Store access, persistence, sorting, and collection mutation are excluded.
-->
<script lang="ts">
  import { createEventDispatcher, tick } from 'svelte';
  import type { ChatRecord } from '../chat/types';

  export let chats: ChatRecord[] = [];
  export let activeId: string;
  export let open = false;
  export let streaming = false;

  const dispatch = createEventDispatcher<{
    new: void;
    select: string;
    rename: { id: string; title: string };
    delete: string;
    close: void;
  }>();
  let menuId: string | null = null;
  let renameId: string | null = null;
  let renameDraft = '';
  let renameInput: HTMLInputElement;
  let newButton: HTMLButtonElement;
  let wasOpen = false;
  $: validRename = renameDraft.trim().length > 0 && Array.from(renameDraft.trim()).length <= 80;
  $: if (open !== wasOpen) {
    wasOpen = open;
    if (open) tick().then(() => newButton?.focus());
  }

  function select(id: string): void {
    if (streaming) return;
    dispatch('select', id);
    dispatch('close');
  }

  async function rename(chat: ChatRecord): Promise<void> {
    menuId = null;
    renameId = chat.id;
    renameDraft = chat.title;
    await tick();
    renameInput?.focus();
    renameInput?.select();
  }

  function saveRename(): void {
    if (!renameId || !validRename || streaming) return;
    dispatch('rename', { id: renameId, title: renameDraft.trim() });
    renameId = null;
  }

  function renameKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter') {
      event.preventDefault();
      saveRename();
    } else if (event.key === 'Escape') {
      event.preventDefault();
      renameId = null;
    }
  }

  function remove(chat: ChatRecord): void {
    menuId = null;
    if (confirm(`Eliminare "${chat.title}"? La chat non potrà essere recuperata.`)) {
      dispatch('delete', chat.id);
    }
  }

  function windowClick(event: MouseEvent): void {
    const target = event.target;
    if (menuId && target instanceof Element && !target.closest('[data-chat-menu]')) {
      menuId = null;
    }
  }

  function windowKeydown(event: KeyboardEvent): void {
    if (event.key !== 'Escape') return;
    if (menuId) menuId = null;
    else if (renameId) renameId = null;
    else if (open) dispatch('close');
  }
</script>

<svelte:window on:click={windowClick} on:keydown={windowKeydown} />
{#if open}<button class="backdrop" type="button" aria-label="Chiudi cronologia" on:click={() => dispatch('close')}></button>{/if}
<aside id="chat-history" class:open aria-label="Chat salvate" aria-hidden={!open} inert={!open}>
  <button class="new-chat" type="button" disabled={streaming} bind:this={newButton} on:click={() => dispatch('new')}>
    Nuova chat
  </button>
  <div class="chat-list">
    {#each chats as chat (chat.id)}
      <div class:active={chat.id === activeId} class="chat-row">
        {#if renameId === chat.id}
          <div class="rename-row">
            <input bind:this={renameInput} bind:value={renameDraft} disabled={streaming} aria-label={`Rinomina ${chat.title}`} on:keydown|stopPropagation={renameKeydown} />
            <div class="rename-actions">
              <button type="button" disabled={streaming || !validRename} on:click={saveRename}>Salva</button>
              <button type="button" disabled={streaming} on:click={() => renameId = null}>Annulla</button>
            </div>
          </div>
        {:else}
          <button class="chat-title" type="button" disabled={streaming} aria-current={chat.id === activeId ? 'true' : undefined} title={chat.title} on:click={() => select(chat.id)}>
            {chat.title}
          </button>
          <div class="menu" data-chat-menu>
            <button class="menu-trigger" type="button" disabled={streaming} aria-label={`Azioni per ${chat.title}`} aria-expanded={menuId === chat.id} on:click|stopPropagation={() => menuId = menuId === chat.id ? null : chat.id}>…</button>
            {#if menuId === chat.id}
              <div class="menu-items">
                <button type="button" disabled={streaming} on:click={() => rename(chat)}>Rinomina</button>
                <button class="destructive" type="button" disabled={streaming} on:click={() => remove(chat)}>Elimina</button>
              </div>
            {/if}
          </div>
        {/if}
      </div>
    {/each}
  </div>
</aside>

<style lang="scss">
  aside {
    width: var(--gn-history-width);
    min-width: var(--gn-history-width);
    min-height: 0;
    box-sizing: border-box;
    display: none;
    flex-direction: column;
    gap: var(--gn-space-md);
    border: var(--gn-border-width) solid var(--gn-border);
    background: var(--gn-bg-panel);
    padding: var(--gn-space-md);
    box-shadow: var(--gn-shadow-hard);
  }
  aside.open { display: flex; }
  .new-chat, .chat-row button, .rename-row button {
    border: var(--gn-border-width) solid var(--gn-border);
    border-radius: var(--gn-radius-sm);
    background: var(--gn-bg-panel);
    color: var(--gn-text-muted);
    cursor: pointer;
    font-family: var(--gn-font-mono);
    font-size: var(--gn-text-xs);
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .new-chat { padding: var(--gn-space-sm); background: var(--gn-accent); color: var(--gn-text-primary); box-shadow: var(--gn-shadow-small); }
  button:focus-visible, input:focus-visible { outline: none; box-shadow: var(--gn-focus-ring); }
  button:disabled { background: var(--gn-bg-panel-raised); color: var(--gn-text-muted); box-shadow: none; cursor: default; }
  .chat-list { min-height: 0; overflow-y: auto; display: flex; flex-direction: column; gap: var(--gn-space-sm); padding: 2px; }
  .chat-row { min-width: 0; display: flex; position: relative; border-left: 4px solid transparent; background: var(--gn-bg-page); }
  .chat-row.active { border-left-color: var(--gn-accent); background: var(--gn-bg-panel-raised); box-shadow: var(--gn-shadow-small); }
  .chat-title { min-width: 0; flex: 1; overflow: hidden; border: 0 !important; padding: var(--gn-space-sm); text-align: left; text-overflow: ellipsis; white-space: nowrap; }
  .menu-trigger { width: 34px; height: 100%; border: 0 !important; font-size: var(--gn-text-md) !important; }
  .menu-items { position: absolute; z-index: 3; top: calc(100% + var(--gn-space-xs)); right: 0; display: grid; gap: var(--gn-space-xs); border: var(--gn-border-width) solid var(--gn-border); background: var(--gn-bg-panel); padding: var(--gn-space-xs); box-shadow: var(--gn-shadow-hard); }
  .menu-items button, .rename-actions button { padding: var(--gn-space-xs) var(--gn-space-sm); }
  button.destructive { border-color: var(--gn-error-border); background: var(--gn-error-bg); color: var(--gn-error-fg); }
  .rename-row { min-width: 0; flex: 1; padding: var(--gn-space-xs); }
  input { width: 100%; box-sizing: border-box; border: var(--gn-border-width) solid var(--gn-border); background: var(--gn-bg-panel); padding: var(--gn-space-xs); color: var(--gn-text-primary); font: inherit; }
  .rename-actions { display: flex; gap: var(--gn-space-xs); margin-top: var(--gn-space-xs); }
  .backdrop { display: none; }
  @media (max-width: 720px) {
    aside { display: flex; position: fixed; z-index: 11; inset: 0 auto 0 0; transform: translateX(-110%); }
    aside.open { transform: translateX(0); }
    .backdrop { display: block; position: fixed; z-index: 10; inset: 0; border: 0; background: var(--gn-history-backdrop); }
  }
</style>
