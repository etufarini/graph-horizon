<!--
SearchOptions.svelte presents the collapsible provider-bound query and category.
Search enablement, interpretation, and transport stay outside this component.
-->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import CollapseControl from './CollapseControl.svelte';
  import type { SearchCategory, SearchSelection } from '../chat/types';

  export let value: SearchSelection;
  export let provider: string;
  export let maxQueryCharacters: number;
  const dispatch = createEventDispatcher<{ change: SearchSelection }>();
  let open = true;

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
  <header>
    <strong>Web search</strong>
    <span>{value.category === 'news' ? 'News' : 'Web'} · {value.query.trim() || 'Uses message text'}</span>
    <CollapseControl
      expanded={open}
      controls="web-search-options"
      openLabel="Open Web search options"
      closeLabel="Close Web search options"
      on:toggle={() => (open = !open)}
    />
  </header>
  <fieldset id="web-search-options" hidden={!open}>
    <legend>Web search options</legend>
    <label class="query">
      <span>Search query</span>
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
  header {
    display: flex;
    align-items: center;
    gap: var(--gn-space-sm);
    min-height: var(--gn-control-height);
    padding: var(--gn-space-xs) var(--gn-space-md);
  }
  header strong { color: var(--gn-text-primary); font: 650 var(--gn-text-xs) var(--gn-font-sans); }
  header > span { min-width: 0; flex: 1; overflow: hidden; color: var(--gn-text-muted); font-size: var(--gn-text-xs); text-overflow: ellipsis; white-space: nowrap; }
  fieldset { min-width: 0; margin: 0; display: flex; flex-wrap: wrap; gap: var(--gn-space-sm); border: 0; border-top: var(--gn-rule-width) solid var(--gn-border-subtle); padding: var(--gn-space-sm) var(--gn-space-md); }
  legend { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  [hidden] { display: none; }
  label { display: flex; align-items: center; gap: var(--gn-space-xs); }
  label.query { flex: 1 1 100%; }
  label.query input { flex: 1 1 auto; min-width: 160px; }
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
    color: var(--gn-text-primary);
    font: inherit;
  }
  select:focus-visible, input:focus-visible {
    outline: none;
    box-shadow: var(--gn-focus-ring);
  }
  p {
    flex: 1 1 100%; margin: 0; color: var(--gn-text-muted);
    font-size: var(--gn-text-xs);
  }
  @media (max-width: 640px) {
    header { min-height: var(--gn-touch-height); }
    fieldset { align-items: stretch; }
    label { flex: 1 1 140px; justify-content: space-between; }
    select, input { min-height: var(--gn-touch-height); }
  }
</style>
