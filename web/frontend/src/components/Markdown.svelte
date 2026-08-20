<script lang="ts">
  /*
   * Markdown.svelte — the single {@html} injection point of the app. Its
   * input is always sanitized by chat/markdown.ts; no other component may
   * use {@html}. Recomputes reactively on every content change so streamed
   * Markdown renders progressively. All content styling lives in
   * theme/markdown.scss under the .markdown-body scope class.
   */
  import { renderMarkdown } from '../chat/markdown';

  export let content: string;
  export let documentPreview = false;

  $: html = renderMarkdown(content, documentPreview);
</script>

<div class="markdown-body">{@html html}</div>
