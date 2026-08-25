<!--
SearchOptions.svelte presents the compact provider-bound query and category
strip. Search enablement, interpretation, and transport stay outside.
-->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { SearchCategory, SearchSelection } from '../chat/types';

  export let value: SearchSelection;
  export let provider: string;
  export let maxQueryCharacters: number;
  const dispatch = createEventDispatcher<{ change: SearchSelection }>();

  function category(event: Event): void {
    change({ category: (event.currentTarget as HTMLSelectElement).value as SearchCategory });
  }

  function query(event: Event): void {
    change({ query: (event.currentTarget as HTMLInputElement).value });
  }

  function change(update: Partial<SearchSelection>): void {
    dispatch('change', { ...value, ...update });
  }
</script>

<section class="search-options" aria-label="Web search options">
  <fieldset>
    <legend>Web search options</legend>
    <label class="query">
      <span>Query</span>
      <input
        type="text"
        value={value.query}
        maxlength={maxQueryCharacters + 1}
        placeholder="Use the message text"
        on:input={query}
      />
    </label>
    <label>
      <span>Source</span>
      <select value={value.category} on:change={category}>
        <option value="web">Web</option>
        <option value="news">News</option>
      </select>
    </label>
    <p>{value.category === 'news' ? 'Use concise keywords for focused News results. ' : ''}Only this query is sent to {provider}; the conversation and files stay local.</p>
  </fieldset>
</section>

<style lang="scss">
  .search-options {
    min-width: 0;
    box-sizing: border-box;
    border-top: var(--gn-rule-width) solid var(--gn-border-subtle);
    background: var(--gn-bg-panel-raised);
  }
  fieldset { min-width: 0; margin: 0; display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: end; gap: var(--gn-space-sm) var(--gn-space-md); border: 0; padding: var(--gn-space-sm) var(--gn-space-md); }
  legend { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  label { min-width: 0; display: grid; gap: var(--gn-space-xs); }
  label.query input { min-width: 0; width: 100%; }
  span {
    color: var(--gn-text-muted);
    font-family: var(--gn-font-sans);
    font-size: var(--gn-text-xs);
  }
  select, input {
    min-height: var(--gn-control-height);
    border: var(--gn-rule-width) solid var(--gn-border);
    border-radius: var(--gn-radius-sm);
    background: var(--gn-bg-panel);
    padding: 0 var(--gn-space-sm);
    color: var(--gn-text-primary);
    font: inherit;
  }
  select:focus-visible, input:focus-visible {
    outline: none;
    box-shadow: var(--gn-focus-ring);
  }
  p {
    grid-column: 1 / -1; margin: 0; color: var(--gn-text-muted);
    font-size: var(--gn-text-xs);
  }
  @media (max-width: 640px) {
    select, input { min-height: var(--gn-touch-height); }
  }
  @media (max-width: 520px) {
    fieldset { grid-template-columns: 1fr; }
    p { grid-column: auto; }
  }
</style>
