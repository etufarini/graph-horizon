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
  .preview { min-height: 0; display: grid; grid-template-rows: auto minmax(0, 1fr); border-top: var(--gn-rule-width) solid var(--gn-border-subtle); }
  header { min-width: 0; display: flex; align-items: center; gap: var(--gn-space-sm); padding: var(--gn-space-xs) 2px; }
  strong { min-width: 0; flex: 1 1 auto; overflow: hidden; color: var(--gn-text-primary); font: 700 var(--gn-text-sm) var(--gn-font-sans); text-overflow: ellipsis; white-space: nowrap; }
  .actions { flex: 0 0 auto; display: flex; gap: var(--gn-space-xs); }
  button { min-height: var(--gn-control-height); border: var(--gn-rule-width) solid var(--gn-border); border-radius: var(--gn-radius-sm); background: var(--gn-bg-panel); padding: var(--gn-space-xs) var(--gn-space-sm); color: var(--gn-text-muted); cursor: pointer; font: 650 var(--gn-text-xs) var(--gn-font-sans); }
  button:hover:not(:disabled) { border-color: var(--gn-accent); background: var(--gn-accent-soft); color: var(--gn-accent-ink); }
  button:focus-visible { outline: none; box-shadow: var(--gn-focus-ring); }
  button:disabled { background: var(--gn-bg-panel-raised); color: var(--gn-text-muted); cursor: default; }
  button.destructive { border-color: var(--gn-error-border); background: var(--gn-error-bg); color: var(--gn-error-fg); }
  .content { min-height: 0; overflow: auto; border-radius: var(--gn-radius-sm); background: var(--gn-bg-page); padding: var(--gn-space-md); }
  @media (max-width: 640px) { button { min-height: var(--gn-touch-height); } }
</style>
