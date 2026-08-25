<!--
ChatHistory.svelte presents the saved-chat sidebar/drawer, row menus, inline
rename draft, confirmations, and keyboard choreography as typed intents.
Store access, persistence, sorting, and collection mutation are excluded.
-->
<script lang="ts">
  import { createEventDispatcher, tick } from 'svelte';
  import CollapseControl from './CollapseControl.svelte';
  import PanelTab from './PanelTab.svelte';
  import type { ChatRecord } from '../chat/types';

  export let chats: ChatRecord[] = [];
  export let activeId: string;
  export let open = false;
  export let overlay = false;
  export let blocked = false;
  export let streaming = false;

  const dispatch = createEventDispatcher<{
    new: void;
    select: string;
    rename: { id: string; title: string };
    delete: string;
    toggle: void;
    close: void;
  }>();
  let menuId: string | null = null;
  let renameId: string | null = null;
  let renameDraft = '';
  let renameInput: HTMLInputElement;
  let reopenButton: HTMLButtonElement;
  let closeButton: HTMLButtonElement;
  $: validRename = renameDraft.trim().length > 0 && Array.from(renameDraft.trim()).length <= 80;

  async function toggle(): Promise<void> {
    dispatch('toggle');
    await tick();
    (open ? closeButton : reopenButton)?.focus();
  }

  async function close(): Promise<void> {
    dispatch('close');
    await tick();
    reopenButton?.focus();
  }

  function select(id: string): void {
    if (streaming) return;
    dispatch('select', id);
    if (overlay) void close();
  }

  async function rename(chat: ChatRecord): Promise<void> {
    menuId = null;
    renameId = chat.id;
    renameDraft = chat.title;
    await tick();
    renameInput?.focus();
    renameInput?.select();
  }

  async function toggleMenu(id: string, event: MouseEvent): Promise<void> {
    menuId = menuId === id ? null : id;
    if (menuId) {
      await tick();
      (event.currentTarget as HTMLElement).closest('.chat-row')?.scrollIntoView({ block: 'nearest' });
    }
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
    if (confirm(`Delete "${chat.title}"? This chat cannot be recovered.`)) {
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
    if (event.key !== 'Escape' || blocked) return;
    if (menuId) menuId = null;
    else if (renameId) renameId = null;
    else if (open && overlay) void close();
  }
</script>

<svelte:window on:click={windowClick} on:keydown={windowKeydown} />
<div class="history-shell" class:overlay class:closed={!open}>
  {#if !open && !blocked}
    <PanelTab bind:element={reopenButton} side="left" label="chat history" controls="chat-history" count={chats.length} on:toggle={toggle} />
  {/if}
  {#if open && overlay}<button class="backdrop" type="button" aria-label="Close chat history" on:click={close}></button>{/if}
  <aside id="chat-history" class:open class:overlay aria-label="Saved chats"
    aria-hidden={!open || blocked} inert={!open || blocked}
    role={overlay ? 'dialog' : undefined} aria-modal={overlay ? 'true' : undefined}>
    <div class="history-header">
      <strong>Chats</strong>
      <span>{chats.length}</span>
      <button class="new-chat" type="button" disabled={streaming} aria-label="New chat" on:click={() => dispatch('new')}>+ New</button>
      <CollapseControl bind:element={closeButton} expanded={true} controls="chat-history"
        openLabel="Open chat history" closeLabel="Close chat history"
        expandDirection="right" collapseDirection="left" on:toggle={toggle} />
    </div>
    <div class="chat-list">
      {#each chats as chat (chat.id)}
        <div class:active={chat.id === activeId} class="chat-row">
          {#if renameId === chat.id}
            <div class="rename-row">
              <input bind:this={renameInput} bind:value={renameDraft} disabled={streaming} aria-label={`Rename ${chat.title}`} on:keydown|stopPropagation={renameKeydown} />
              <div class="rename-actions">
                <button type="button" disabled={streaming || !validRename} on:click={saveRename}>Save</button>
                <button type="button" disabled={streaming} on:click={() => renameId = null}>Cancel</button>
              </div>
            </div>
          {:else}
            <button class="chat-title" type="button" disabled={streaming} aria-current={chat.id === activeId ? 'true' : undefined} title={chat.title} on:click={() => select(chat.id)}>
              {chat.title}
            </button>
            <div class="menu" data-chat-menu>
              <button class="menu-trigger" type="button" disabled={streaming} aria-label={`Actions for ${chat.title}`} aria-expanded={menuId === chat.id} on:click|stopPropagation={event => toggleMenu(chat.id, event)}>…</button>
            </div>
            {#if menuId === chat.id}
              <div class="menu-items" data-chat-menu>
                <button type="button" disabled={streaming} on:click={() => rename(chat)}>Rename</button>
                <button class="destructive" type="button" disabled={streaming} on:click={() => remove(chat)}>Delete</button>
              </div>
            {/if}
          {/if}
        </div>
      {/each}
    </div>
  </aside>
</div>

<style lang="scss">
  .history-shell { width: var(--gn-history-width); min-width: var(--gn-history-width); height: 100%; min-height: 0; }
  .history-shell.closed { width: 0; min-width: 0; }
  .history-shell.overlay { width: 0; min-width: 0; }
  aside {
    width: 100%;
    min-width: 100%;
    height: 100%;
    min-height: 0;
    box-sizing: border-box;
    display: none;
    flex-direction: column;
    gap: var(--gn-space-sm);
    border: 0;
    border-right: var(--gn-rule-width) solid var(--gn-border-subtle);
    border-radius: 0;
    background: var(--gn-bg-panel);
    padding: var(--gn-space-md);
  }
  aside.open { display: flex; }
  .new-chat, .chat-row button, .rename-row button {
    min-height: var(--gn-control-height);
    border: var(--gn-rule-width) solid var(--gn-border);
    border-radius: var(--gn-radius-sm);
    background: var(--gn-bg-panel);
    color: var(--gn-text-muted);
    cursor: pointer;
    font-family: var(--gn-font-sans);
    font-size: var(--gn-text-xs);
    font-weight: 650;
  }
  .history-header { min-width: 0; display: flex; align-items: center; gap: var(--gn-space-sm); }
  .history-header strong { min-width: 0; flex: 1 1 auto; overflow: hidden; color: var(--gn-text-primary); font: 700 var(--gn-text-sm) var(--gn-font-sans); text-overflow: ellipsis; white-space: nowrap; }
  .history-header > span { color: var(--gn-text-muted); font: var(--gn-text-xs) var(--gn-font-mono); }
  .new-chat { padding: var(--gn-space-xs) var(--gn-space-sm); border-color: var(--gn-accent); background: var(--gn-accent); color: var(--gn-bg-panel); white-space: nowrap; }
  button:hover:not(:disabled) { border-color: var(--gn-accent); background: var(--gn-accent-soft); color: var(--gn-accent-ink); }
  .new-chat:hover:not(:disabled) { background: var(--gn-accent-ink); color: var(--gn-bg-panel); }
  button:focus-visible, input:focus-visible { outline: none; box-shadow: var(--gn-focus-ring); }
  button:disabled { background: var(--gn-bg-panel-raised); color: var(--gn-text-muted); box-shadow: none; cursor: default; }
  .chat-list { min-height: 0; overflow-y: auto; display: flex; flex-direction: column; gap: var(--gn-space-sm); padding: var(--gn-space-2xs); }
  .chat-row { min-width: 0; display: flex; flex-wrap: wrap; border-left: 3px solid transparent; border-radius: var(--gn-radius-sm); background: var(--gn-bg-page); }
  .chat-row.active { border-left-color: var(--gn-accent); background: var(--gn-accent-soft); }
  .chat-title { min-width: 0; flex: 1; overflow: hidden; border: 0 !important; padding: var(--gn-space-sm); text-align: left; text-overflow: ellipsis; white-space: nowrap; }
  .menu { flex: 0 0 34px; }
  .menu-trigger { width: 34px; height: 100%; border: 0 !important; font-size: var(--gn-text-md) !important; }
  .menu-items { flex: 0 0 100%; box-sizing: border-box; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--gn-space-xs); border-top: var(--gn-rule-width) solid var(--gn-border-subtle); background: var(--gn-bg-panel); padding: var(--gn-space-xs); }
  .menu-items button { min-width: 0; width: 100%; }
  .menu-items button, .rename-actions button { padding: var(--gn-space-xs) var(--gn-space-sm); }
  button.destructive { border-color: var(--gn-error-border); background: var(--gn-error-bg); color: var(--gn-error-fg); }
  .rename-row { min-width: 0; flex: 1; padding: var(--gn-space-xs); }
  input { width: 100%; box-sizing: border-box; border: var(--gn-border-width) solid var(--gn-border); background: var(--gn-bg-panel); padding: var(--gn-space-xs); color: var(--gn-text-primary); font: inherit; }
  .rename-actions { display: flex; gap: var(--gn-space-xs); margin-top: var(--gn-space-xs); }
  .backdrop { display: block; position: fixed; z-index: 10; inset: 0; border: 0; background: var(--gn-history-backdrop); }
  .backdrop:focus-visible { outline: none; box-shadow: var(--gn-focus-inset); }
  aside.overlay { display: flex; position: fixed; z-index: 11; inset: env(safe-area-inset-top) auto env(safe-area-inset-bottom) env(safe-area-inset-left); width: var(--gn-history-width); min-width: var(--gn-history-width); height: auto; border: var(--gn-border-width) solid var(--gn-border); border-left: 0; box-shadow: var(--gn-shadow-hard); transform: translateX(-110%); transition: transform var(--gn-motion-fast) ease; }
  aside.overlay.open { transform: translateX(0); }
  @media (max-width: 640px) {
    aside.overlay { width: min(var(--gn-history-width), 88vw); min-width: min(var(--gn-history-width), 88vw); }
    .new-chat, .chat-row button, .rename-row button { min-height: var(--gn-touch-height); }
  }
</style>
