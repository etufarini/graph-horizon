<!--
Markdown-file panel: present active-chat selection, picker/drop input, current
records, and the responsive right drawer as typed intents. Storage, validation,
prompt projection, downloads, and chat lifecycle remain outside.
-->
<script lang="ts">
  import { createEventDispatcher, tick } from 'svelte';
  import Preview from './Preview.svelte';
  import type { MarkdownFileRecord } from '../../chat/files/record.ts';

  export let files: MarkdownFileRecord[] = [];
  export let open = false;
  export let overlay = false;
  export let disabled = false;
  export let busy = false;
  export let ready = false;

  const dispatch = createEventDispatcher<{
    add: File[];
    download: MarkdownFileRecord;
    delete: string;
    close: void;
  }>();
  let picker: HTMLInputElement;
  let closeButton: HTMLButtonElement;
  let selectedId: string | null = null;
  let dragging = false;
  let wasOpen = false;
  $: if (open !== wasOpen) {
    wasOpen = open;
    if (open && overlay) tick().then(() => closeButton?.focus());
  }
  $: if (!selectedId || !files.some(file => file.id === selectedId)) {
    selectedId = files[0]?.id ?? null;
  }
  $: selected = files.find(file => file.id === selectedId) ?? null;

  function submit(list: FileList | File[]): void {
    if (disabled || busy || !ready) return;
    const selected = Array.from(list);
    if (selected.length === 0) return;
    const duplicates = selected.filter(file => files.some(current => current.name === file.name));
    if (duplicates.length > 0 && !confirm(
      `Replace ${duplicates.length === 1 ? `“${duplicates[0].name}”` : `${duplicates.length} files with matching names`}?`
    )) return;
    dispatch('add', selected);
  }

  function picked(): void {
    if (picker.files) submit(picker.files);
    picker.value = '';
  }

  function dropped(event: DragEvent): void {
    dragging = false;
    if (event.dataTransfer?.files) submit(event.dataTransfer.files);
  }

  function remove(file: MarkdownFileRecord): void {
    if (disabled || busy) return;
    if (confirm(`Delete “${file.name}”? The file cannot be recovered.`)) {
      dispatch('delete', file.id);
    }
  }

  function keydown(event: KeyboardEvent): void {
    if (event.key === 'Escape' && open && overlay) dispatch('close');
  }

  function size(bytes: number): string {
    return bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(bytes < 10 * 1024 ? 1 : 0)} KiB`;
  }
</script>

<svelte:window on:keydown={keydown} />
{#if open && overlay}<button class="backdrop" type="button" aria-label="Close Markdown files" on:click={() => dispatch('close')}></button>{/if}
<aside id="markdown-files" class:open class:overlay class:dragging aria-label="Markdown files" aria-hidden={!open} inert={!open}
  role={overlay ? 'dialog' : undefined} aria-modal={overlay ? 'true' : undefined}
  on:dragover|preventDefault={() => { if (!disabled && ready) dragging = true; }}
  on:dragleave={() => dragging = false}
  on:drop|preventDefault={dropped}>
  <header>
    <div>
      <strong>Markdown files</strong>
      <span>{files.length} / 10</span>
    </div>
    {#if overlay}<button bind:this={closeButton} class="close" type="button" aria-label="Close Markdown files" on:click={() => dispatch('close')}>×</button>{/if}
  </header>
  <button class="add" type="button" disabled={disabled || busy || !ready} on:click={() => picker.click()}>
    {busy ? 'Saving…' : ready ? '+ Add .md' : 'Loading…'}
  </button>
  <input type="file" accept=".md,text/markdown,text/plain" multiple bind:this={picker} on:change={picked} aria-hidden="true" tabindex="-1" />
  <p class="hint">Drop UTF-8 files here. They will provide context for future requests.</p>
  <div class="file-list" aria-label="Saved files">
    {#each files as file (file.id)}
      <button class:active={file.id === selectedId} type="button" title={file.name} on:click={() => selectedId = file.id}>
        <span>{file.name}</span><small>{size(file.utf8Bytes)}</small>
      </button>
    {:else}
      <p class="empty">No files in this chat</p>
    {/each}
  </div>
  {#if selected}
    <Preview file={selected} {disabled} on:download={event => dispatch('download', event.detail)} on:delete={event => remove(event.detail)} />
  {/if}
</aside>

<style lang="scss">
  aside { width: var(--gn-files-width); min-width: var(--gn-files-width); min-height: 0; box-sizing: border-box; display: none; grid-template-rows: auto auto auto minmax(72px, 0.35fr) minmax(0, 1fr); gap: var(--gn-space-sm); border: var(--gn-rule-width) solid var(--gn-border-subtle); border-radius: var(--gn-radius-md); background: var(--gn-bg-panel); padding: var(--gn-space-md); }
  aside.open { display: grid; }
  aside.dragging { box-shadow: var(--gn-focus-inset); background: var(--gn-bg-panel-raised); }
  header, header div { min-width: 0; display: flex; align-items: center; justify-content: space-between; gap: var(--gn-space-sm); }
  header strong { color: var(--gn-text-primary); font: 700 var(--gn-text-sm) var(--gn-font-sans); }
  header span { color: var(--gn-text-muted); font: var(--gn-text-xs) var(--gn-font-mono); }
  button { border-radius: var(--gn-radius-sm); font-family: var(--gn-font-mono); }
  .close { min-width: var(--gn-control-height); min-height: var(--gn-control-height); border: 0; border-radius: var(--gn-radius-sm); background: transparent; color: var(--gn-text-muted); cursor: pointer; font-size: var(--gn-text-lg); }
  .add { min-height: var(--gn-control-height); border: var(--gn-rule-width) solid var(--gn-accent); border-radius: var(--gn-radius-sm); background: var(--gn-accent); padding: var(--gn-space-sm); color: var(--gn-bg-panel); cursor: pointer; font-size: var(--gn-text-xs); font-weight: 700; }
  .close:hover, .add:hover:not(:disabled) { background: var(--gn-accent-ink); color: var(--gn-bg-panel); }
  .add:disabled { background: var(--gn-bg-panel-raised); color: var(--gn-text-muted); box-shadow: none; cursor: default; }
  button:focus-visible { outline: none; box-shadow: var(--gn-focus-ring); }
  input { display: none; }
  .hint, .empty { margin: 0; color: var(--gn-text-muted); font-size: var(--gn-text-xs); }
  .file-list { min-height: 0; overflow-y: auto; display: flex; flex-direction: column; gap: var(--gn-space-xs); padding: 2px; }
  .file-list button { min-width: 0; min-height: var(--gn-control-height); display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: var(--gn-space-sm); border: var(--gn-rule-width) solid transparent; border-radius: var(--gn-radius-sm); background: var(--gn-bg-page); padding: var(--gn-space-sm); color: var(--gn-text-primary); cursor: pointer; text-align: left; }
  .file-list button:hover { border-color: var(--gn-border-subtle); background: var(--gn-bg-panel-raised); }
  .file-list button.active { border-color: var(--gn-accent); background: var(--gn-accent-soft); }
  .file-list span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .file-list small { color: var(--gn-text-muted); font-size: var(--gn-text-xs); }
  .backdrop { display: block; position: fixed; z-index: 10; inset: 0; border: 0; background: var(--gn-history-backdrop); }
  aside.overlay { position: fixed; z-index: 11; inset: 0 0 0 auto; border-radius: var(--gn-radius-md) 0 0 var(--gn-radius-md); box-shadow: var(--gn-shadow-hard); transform: translateX(110%); }
  aside.overlay.open { transform: translateX(0); }
  @media (max-width: 640px) {
    aside { width: min(var(--gn-files-width), 92vw); min-width: min(var(--gn-files-width), 92vw); }
    .close, .add, .file-list button { min-height: var(--gn-touch-height); }
  }
</style>
