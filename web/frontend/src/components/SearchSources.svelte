<!--
SearchSources.svelte presents one collapsible persisted Web-search report apart
from assistant Markdown. Its compact header preserves query, source count, and
activity while validated provider links remain inside the expanded body.
-->
<script lang="ts">
  import CollapseControl from './CollapseControl.svelte';
  import type { SearchReport } from '../chat/types';
  import { splitReasoning } from '../chat/reasoning';

  export let report: SearchReport;
  export let answer: string;
  export let messageId: string;
  export let streaming = false;
  let open = true;

  $: visibleAnswer = splitReasoning(answer, false).answer;
  $: cited = new Set(Array.from(visibleAnswer.matchAll(/\[S([1-5])\]/g), match => `S${match[1]}`));
  $: sourcesId = `${messageId}-search-sources`;
</script>

<aside class="sources" aria-label="Web search provenance">
  <header>
    <strong>Web search</strong>
    <span>{report.provider} · “{report.query}” · {report.sources.length} {report.sources.length === 1 ? 'source' : 'sources'}</span>
    {#if streaming}<small class="activity" aria-live="polite">Answering</small>{/if}
    <CollapseControl
      expanded={open}
      controls={sourcesId}
      openLabel="Open Web search sources"
      closeLabel="Close Web search sources"
      on:toggle={() => (open = !open)}
    />
  </header>
  <ol id={sourcesId} hidden={!open}>
    {#each report.sources as source}
      <li>
        <span class:cited={cited.has(source.id)}>{cited.has(source.id) ? 'Cited' : 'Search result'} {source.id}</span>
        <a href={source.url} target="_blank" rel="noreferrer">{source.title}</a>
        {#if source.publisher}<small>{source.publisher}</small>{/if}
      </li>
    {/each}
  </ol>
</aside>

<style lang="scss">
  .sources {
    width: min(88%, 840px); box-sizing: border-box;
    border: var(--gn-rule-width) solid var(--gn-border-subtle); border-radius: var(--gn-radius-md);
    background: var(--gn-bg-panel-raised); padding: var(--gn-space-sm) var(--gn-space-md);
  }
  header { min-width: 0; display: flex; align-items: center; gap: var(--gn-space-xs) var(--gn-space-sm); }
  header strong, li > span {
    font-family: var(--gn-font-sans); font-size: var(--gn-text-xs); font-weight: 650;
  }
  header > span { min-width: 0; flex: 1; overflow: hidden; color: var(--gn-text-muted); font-size: var(--gn-text-xs); text-overflow: ellipsis; white-space: nowrap; }
  small { color: var(--gn-text-muted); font-size: var(--gn-text-xs); }
  .activity { color: var(--gn-accent-ink); font-weight: 700; }
  [hidden] { display: none; }
  ol { margin: var(--gn-space-sm) 0 0; padding-left: 1.25rem; }
  li { margin-top: var(--gn-space-xs); }
  li > span { margin-right: var(--gn-space-xs); color: var(--gn-text-muted); }
  li > span.cited { color: var(--gn-accent-ink); }
  a { color: var(--gn-text-primary); overflow-wrap: anywhere; }
  small { display: block; margin-left: 0; }
  @media (max-width: 640px) {
    .sources { width: 100%; }
    header { flex-wrap: wrap; }
    header > span { flex-basis: calc(100% - var(--gn-touch-height) - var(--gn-space-sm)); }
  }
</style>
