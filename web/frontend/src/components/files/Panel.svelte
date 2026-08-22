<!--
Markdown-file panel: present active-chat selection, picker/drop input, current
records, and the responsive right drawer as typed intents. Storage, validation,
prompt projection, downloads, and chat lifecycle remain outside.
-->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
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
  let selectedId: string | null = null;
  let dragging = false;
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
  on:dragover|preventDefault={() => { if (!disabled && ready) dragging = true; }}
  on:dragleave={() => dragging = false}
  on:drop|preventDefault={dropped}>
  <header>
    <div>
      <strong>Markdown files</strong>
      <span>{files.length} / 10</span>
    </div>
    {#if overlay}<button class="close" type="button" aria-label="Close Markdown files" on:click={() => dispatch('close')}>×</button>{/if}
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
  aside { width: var(--gn-files-width); min-width: var(--gn-files-width); min-height: 0; box-sizing: border-box; display: none; grid-template-rows: auto auto auto minmax(72px, 0.35fr) minmax(0, 1fr); gap: var(--gn-space-sm); border: var(--gn-border-width) solid var(--gn-border); background: var(--gn-bg-panel); padding: var(--gn-space-md); box-shadow: var(--gn-shadow-hard); }
  aside.open { display: grid; }
  aside.dragging { box-shadow: var(--gn-focus-inset), var(--gn-shadow-hard); background: var(--gn-bg-panel-raised); }
  header, header div { min-width: 0; display: flex; align-items: center; justify-content: space-between; gap: var(--gn-space-sm); }
  header strong { color: var(--gn-accent-ink); font: 700 var(--gn-text-sm) var(--gn-font-mono); letter-spacing: 0.08em; text-transform: uppercase; }
  header span { color: var(--gn-text-muted); font: var(--gn-text-xs) var(--gn-font-mono); }
  button { border-radius: var(--gn-radius-sm); font-family: var(--gn-font-mono); }
  .close { border: 0; background: transparent; color: var(--gn-text-muted); cursor: pointer; font-size: var(--gn-text-lg); }
  .add { border: var(--gn-border-width) solid var(--gn-border); background: var(--gn-accent); padding: var(--gn-space-sm); color: var(--gn-text-primary); box-shadow: var(--gn-shadow-small); cursor: pointer; font-size: var(--gn-text-xs); font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
  .add:disabled { background: var(--gn-bg-panel-raised); color: var(--gn-text-muted); box-shadow: none; cursor: default; }
  button:focus-visible { outline: none; box-shadow: var(--gn-focus-ring); }
  input { display: none; }
  .hint, .empty { margin: 0; color: var(--gn-text-muted); font-size: var(--gn-text-xs); }
  .file-list { min-height: 0; overflow-y: auto; display: flex; flex-direction: column; gap: var(--gn-space-xs); padding: 2px; }
  .file-list button { min-width: 0; display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: var(--gn-space-sm); border: var(--gn-border-width) solid transparent; background: var(--gn-bg-page); padding: var(--gn-space-sm); color: var(--gn-text-primary); cursor: pointer; text-align: left; }
  .file-list button.active { border-color: var(--gn-border); background: var(--gn-bg-panel-raised); box-shadow: var(--gn-shadow-small); }
  .file-list span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .file-list small { color: var(--gn-text-muted); font-size: var(--gn-text-xs); }
  .backdrop { display: block; position: fixed; z-index: 10; inset: 0; border: 0; background: var(--gn-history-backdrop); }
  aside.overlay { position: fixed; z-index: 11; inset: 0 0 0 auto; transform: translateX(110%); }
  aside.overlay.open { transform: translateX(0); }
</style>
