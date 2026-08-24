<script lang="ts">
  /* Explicit Web-search category and calendar scope; query interpretation and
     transport remain outside this presentation component. */
  import { createEventDispatcher } from 'svelte';
  import type { SearchCategory, SearchPeriod, SearchSelection } from '../chat/types';

  export let value: SearchSelection;
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

  function change(update: Partial<SearchSelection>): void {
    dispatch('change', { ...value, ...update });
  }
</script>

<div class="search-options" aria-label="Web search options">
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
</div>

<style lang="scss">
  .search-options {
    display: flex;
    flex-wrap: wrap;
    gap: var(--gn-space-sm);
    border-top: var(--gn-border-width) solid var(--gn-border);
    padding: var(--gn-space-sm) var(--gn-space-md);
    background: var(--gn-bg-panel-raised);
  }
  label { display: flex; align-items: center; gap: var(--gn-space-xs); }
  span {
    color: var(--gn-text-muted);
    font-family: var(--gn-font-mono);
    font-size: var(--gn-text-xs);
  }
  select, input {
    min-height: 30px;
    border: var(--gn-border-width) solid var(--gn-border);
    border-radius: var(--gn-radius-sm);
    background: var(--gn-bg-panel);
    color: var(--gn-text-primary);
    font: inherit;
  }
  select:focus-visible, input:focus-visible {
    outline: none;
    box-shadow: var(--gn-focus-ring);
  }
  @media (max-width: 720px) {
    .search-options { align-items: stretch; }
    label { flex: 1 1 140px; justify-content: space-between; }
  }
</style>
