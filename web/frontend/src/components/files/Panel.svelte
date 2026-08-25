<!--
Markdown-file panel: present active-chat selection, picker/drop input, current
records, and the responsive right drawer as typed intents. Storage, validation,
prompt projection, downloads, and chat lifecycle remain outside.
-->
<script lang="ts">
  import { createEventDispatcher, tick } from 'svelte';
  import CollapseControl from '../CollapseControl.svelte';
  import PanelTab from '../PanelTab.svelte';
  import Preview from './Preview.svelte';
  import type { MarkdownFileRecord } from '../../chat/files/record.ts';

  export let files: MarkdownFileRecord[] = [];
  export let open = false;
  export let overlay = false;
  export let blocked = false;
  export let disabled = false;
  export let busy = false;
  export let ready = false;

  const dispatch = createEventDispatcher<{
    add: File[];
    download: MarkdownFileRecord;
    delete: string;
    toggle: void;
    close: void;
  }>();
  let picker: HTMLInputElement;
  let closeButton: HTMLButtonElement;
  let reopenButton: HTMLButtonElement;
  let selectedId: string | null = null;
  let dragging = false;
  $: if (!selectedId || !files.some(file => file.id === selectedId)) {
    selectedId = files[0]?.id ?? null;
  }
  $: selected = files.find(file => file.id === selectedId) ?? null;

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
    if (event.key === 'Escape' && open && overlay && !blocked) void close();
  }

  function size(bytes: number): string {
    return bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(bytes < 10 * 1024 ? 1 : 0)} KiB`;
  }
</script>

<svelte:window on:keydown={keydown} />
<div class="files-shell" class:overlay class:closed={!open}>
  {#if !open && !blocked}
    <PanelTab bind:element={reopenButton} side="right" label="Markdown files" controls="markdown-files" count={files.length} on:toggle={toggle} />
  {/if}
  {#if open && overlay}<button class="backdrop" type="button" aria-label="Close Markdown files" on:click={close}></button>{/if}
  <aside id="markdown-files" class:open class:overlay class:dragging aria-label="Markdown files" aria-hidden={!open || blocked} inert={!open || blocked}
    role={overlay ? 'dialog' : undefined} aria-modal={overlay ? 'true' : undefined}
    on:dragover|preventDefault={() => { if (!disabled && ready) dragging = true; }}
    on:dragleave={() => dragging = false}
    on:drop|preventDefault={dropped}>
    <header>
      <div class="panel-title">
        <strong>Markdown files</strong>
        <span>{files.length} / 10</span>
      </div>
      <button class="add" type="button" disabled={disabled || busy || !ready} aria-label="Add Markdown files" on:click={() => picker.click()}>
        {busy ? 'Saving…' : ready ? '+ Add' : 'Loading…'}
      </button>
      <CollapseControl
        bind:element={closeButton}
        expanded={true}
        controls="markdown-files"
        openLabel="Open Markdown files"
        closeLabel="Close Markdown files"
        expandDirection="left"
        collapseDirection="right"
        on:toggle={toggle}
      />
    </header>
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
</div>

<style lang="scss">
  .files-shell { width: var(--gn-files-width); min-width: var(--gn-files-width); height: 100%; min-height: 0; }
  .files-shell.closed { width: 0; min-width: 0; }
  .files-shell.overlay { width: 0; min-width: 0; }
  aside { width: 100%; min-width: 100%; height: 100%; min-height: 0; display: none; grid-template-rows: auto auto minmax(72px, 0.35fr) minmax(0, 1fr); gap: var(--gn-space-sm); border: 0; border-left: var(--gn-rule-width) solid var(--gn-border-subtle); border-radius: 0; background: var(--gn-bg-panel); padding: var(--gn-space-md); }
  aside.open { display: grid; }
  aside.dragging { box-shadow: var(--gn-focus-inset); background: var(--gn-bg-panel-raised); }
  header, .panel-title { min-width: 0; display: flex; align-items: center; gap: var(--gn-space-sm); }
  .panel-title { flex: 1 1 auto; }
  .panel-title span { margin-left: auto; }
  header strong { min-width: 0; flex: 1 1 auto; overflow: hidden; color: var(--gn-text-primary); font: 700 var(--gn-text-sm) var(--gn-font-sans); text-overflow: ellipsis; white-space: nowrap; }
  header span { color: var(--gn-text-muted); font: var(--gn-text-xs) var(--gn-font-mono); }
  button { border-radius: var(--gn-radius-sm); font-family: var(--gn-font-mono); }
  .add { min-height: var(--gn-control-height); flex: 0 0 auto; border: var(--gn-rule-width) solid var(--gn-accent); border-radius: var(--gn-radius-sm); background: var(--gn-accent); padding: var(--gn-space-xs) var(--gn-space-sm); color: var(--gn-bg-panel); cursor: pointer; font-size: var(--gn-text-xs); font-weight: 700; white-space: nowrap; }
  .add:hover:not(:disabled) { background: var(--gn-accent-ink); color: var(--gn-bg-panel); }
  .add:disabled { background: var(--gn-bg-panel-raised); color: var(--gn-text-muted); box-shadow: none; cursor: default; }
  button:focus-visible { outline: none; box-shadow: var(--gn-focus-ring); }
  input { display: none; }
  .hint, .empty { margin: 0; color: var(--gn-text-muted); font-size: var(--gn-text-xs); }
  .file-list { min-height: 0; overflow-y: auto; display: flex; flex-direction: column; gap: var(--gn-space-xs); padding: var(--gn-space-2xs); }
  .file-list button { min-width: 0; min-height: var(--gn-control-height); display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: var(--gn-space-sm); border: var(--gn-rule-width) solid transparent; border-radius: var(--gn-radius-sm); background: var(--gn-bg-page); padding: var(--gn-space-sm); color: var(--gn-text-primary); cursor: pointer; text-align: left; }
  .file-list button:hover { border-color: var(--gn-border-subtle); background: var(--gn-bg-panel-raised); }
  .file-list button.active { border-color: var(--gn-accent); background: var(--gn-accent-soft); }
  .file-list span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .file-list small { color: var(--gn-text-muted); font-size: var(--gn-text-xs); }
  .backdrop { display: block; position: fixed; z-index: 10; inset: 0; border: 0; background: var(--gn-history-backdrop); }
  aside.overlay { display: grid; position: fixed; z-index: 11; inset: env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) auto; width: var(--gn-files-width); min-width: var(--gn-files-width); height: auto; border: var(--gn-border-width) solid var(--gn-border); border-right: 0; box-shadow: var(--gn-shadow-hard); transform: translateX(110%); transition: transform var(--gn-motion-fast) ease; }
  aside.overlay.open { transform: translateX(0); }
  @media (max-width: 640px) {
    aside.overlay { width: min(var(--gn-files-width), 92vw); min-width: min(var(--gn-files-width), 92vw); }
    .add, .file-list button { min-height: var(--gn-touch-height); }
  }
</style>
