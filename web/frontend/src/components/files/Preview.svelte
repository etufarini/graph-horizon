<!--
Markdown-file preview: present one validated stored document through the shared
sanitized Markdown renderer and expose typed download/delete intents. File-list
selection, persistence, confirmation, and responsive panel state are excluded.
-->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Markdown from '../Markdown.svelte';
  import type { MarkdownFileRecord } from '../../chat/files/record.ts';

  export let file: MarkdownFileRecord;
  export let disabled = false;

  const dispatch = createEventDispatcher<{
    download: MarkdownFileRecord;
    delete: MarkdownFileRecord;
  }>();
</script>

<section class="preview" aria-label={`Preview ${file.name}`}>
  <header>
    <strong title={file.name}>{file.name}</strong>
    <div class="actions">
      <button type="button" on:click={() => dispatch('download', file)}>Download</button>
      <button class="destructive" type="button" {disabled} on:click={() => dispatch('delete', file)}>Delete</button>
    </div>
  </header>
  <div class="content"><Markdown content={file.content} documentPreview /></div>
</section>

<style lang="scss">
  .preview { min-height: 0; display: grid; grid-template-rows: auto minmax(0, 1fr); border-top: var(--gn-border-width) solid var(--gn-border); }
  header { min-width: 0; display: grid; gap: var(--gn-space-sm); padding: var(--gn-space-sm) 2px; }
  strong { min-width: 0; overflow: hidden; color: var(--gn-accent-ink); font: 700 var(--gn-text-sm) var(--gn-font-mono); text-overflow: ellipsis; white-space: nowrap; }
  .actions { display: flex; gap: var(--gn-space-xs); }
  button { border: var(--gn-border-width) solid var(--gn-border); background: var(--gn-bg-panel); padding: var(--gn-space-xs) var(--gn-space-sm); color: var(--gn-text-muted); cursor: pointer; font: 700 var(--gn-text-xs) var(--gn-font-mono); text-transform: uppercase; }
  button:hover:not(:disabled) { border-color: var(--gn-accent-ink); color: var(--gn-accent-ink); }
  button:focus-visible { outline: none; box-shadow: var(--gn-focus-ring); }
  button:disabled { background: var(--gn-bg-panel-raised); color: var(--gn-text-muted); cursor: default; }
  button.destructive { border-color: var(--gn-error-border); background: var(--gn-error-bg); color: var(--gn-error-fg); }
  .content { min-height: 0; overflow: auto; background: var(--gn-bg-page); padding: var(--gn-space-md); }
</style>
