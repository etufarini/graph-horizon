<script lang="ts">
  /* Explicit Web-search category and calendar scope; query interpretation and
     transport remain outside this presentation component. */
  import { createEventDispatcher } from 'svelte';
  import type { SearchCategory, SearchPeriod, SearchSelection } from '../chat/types';

  export let value: SearchSelection;
  export let provider: string;
  export let maxQueryCharacters: number;
  const dispatch = createEventDispatcher<{ change: SearchSelection }>();

  function category(event: Event): void {
    change({ category: (event.currentTarget as HTMLSelectElement).value as SearchCategory });
  }

  function period(event: Event): void {
    change({ period: (event.currentTarget as HTMLSelectElement).value as SearchPeriod });
  }

  function date(field: 'from' | 'to', event: Event): void {
    change({ [field]: (event.currentTarget as HTMLInputElement).value });
  }

  function query(event: Event): void {
    change({ query: (event.currentTarget as HTMLInputElement).value });
  }

  function change(update: Partial<SearchSelection>): void {
    dispatch('change', { ...value, ...update });
  }
</script>

<fieldset class="search-options">
  <legend>Web search</legend>
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
  <label>
    <span>Published</span>
    <select value={value.period} on:change={period}>
      <option value="any">Any time</option>
      <option value="day">Today</option>
      <option value="week">Last 7 days</option>
      <option value="month">Last 30 days</option>
      <option value="custom">Custom dates</option>
    </select>
  </label>
  {#if value.period === 'custom'}
    <label>
      <span>From</span>
      <input type="date" value={value.from} on:input={event => date('from', event)} />
    </label>
    <label>
      <span>To</span>
      <input type="date" value={value.to} min={value.from || undefined} on:input={event => date('to', event)} />
    </label>
  {/if}
  <p>Only this query is sent to {provider}; the conversation and files stay local.</p>
</fieldset>

<style lang="scss">
  .search-options {
    min-width: 0;
    box-sizing: border-box;
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    gap: var(--gn-space-sm);
    border: 0;
    border-top: var(--gn-rule-width) solid var(--gn-border-subtle);
    padding: var(--gn-space-sm) var(--gn-space-md);
    background: var(--gn-bg-panel-raised);
  }
  legend { float: left; margin: 0 var(--gn-space-sm) 0 0; padding: 0; color: var(--gn-text-primary); font: 650 var(--gn-text-xs) var(--gn-font-sans); }
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
    .search-options { align-items: stretch; }
    label { flex: 1 1 140px; justify-content: space-between; }
    select, input { min-height: var(--gn-touch-height); }
  }
</style>
