<!--
SearchSources.svelte presents persisted Web-search provenance separately from
assistant Markdown. It labels only identifiers actually cited by the answer as
cited and opens validated provider URLs without sharing an opener.
-->
<script lang="ts">
  import type { SearchReport } from '../chat/types';
  import { splitReasoning } from '../chat/reasoning';

  export let report: SearchReport;
  export let answer: string;

  $: visibleAnswer = splitReasoning(answer, false).answer;
  $: cited = new Set(Array.from(visibleAnswer.matchAll(/\[S([1-5])\]/g), match => `S${match[1]}`));
</script>

<aside class="sources" aria-label="Web search provenance">
  <header>
    <strong>Web search</strong>
    <span>{report.provider} · “{report.query}”</span>
  </header>
  <ol>
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
  header { display: flex; flex-wrap: wrap; gap: var(--gn-space-xs) var(--gn-space-sm); }
  header strong, li > span {
    font-family: var(--gn-font-sans); font-size: var(--gn-text-xs); font-weight: 650;
  }
  header span, small { color: var(--gn-text-muted); font-size: var(--gn-text-xs); }
  ol { margin: var(--gn-space-sm) 0 0; padding-left: 1.25rem; }
  li { margin-top: var(--gn-space-xs); }
  li > span { margin-right: var(--gn-space-xs); color: var(--gn-text-muted); }
  li > span.cited { color: var(--gn-accent-ink); }
  a { color: var(--gn-text-primary); overflow-wrap: anywhere; }
  small { display: block; margin-left: 0; }
  @media (max-width: 640px) { .sources { width: 100%; } }
</style>
