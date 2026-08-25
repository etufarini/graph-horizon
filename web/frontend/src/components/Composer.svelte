<script lang="ts">
  /*
   * Composer.svelte
   * Single responsibility: keep the next draft and its explicit Web-search
   * choice editable while exposing send or stop for the active request.
   */
  import { createEventDispatcher } from 'svelte';
  import { defaultSearch, validSearch } from '../chat/search';
  import type { SearchCapability, SearchSelection } from '../chat/types';
  import SearchOptions from './SearchOptions.svelte';

  export let value = '';
  export let streaming = false;
  export let contextAvailable = false;
  export let search: SearchSelection | null = null;
  export let searchCapability: SearchCapability | null = null;

  const dispatch = createEventDispatcher<{
    send: void;
    stop: void;
  }>();
  $: searchTerms = search ? search.query.trim() || value.trim() : '';
  $: queryTooLong = search !== null && searchCapability !== null &&
    Array.from(searchTerms).length > searchCapability.maxQueryCharacters;
  $: searchAvailable = searchCapability !== null;
  $: searchState = !searchAvailable ? 'Search unavailable' : search ? 'Search on' : 'Search off';
  $: searchAction = !searchAvailable
    ? 'Web search is not configured'
    : search ? 'Disable Web search' : 'Enable Web search';
  $: canSend = value.trim().length > 0 && !streaming && contextAvailable &&
    (search === null || (searchAvailable && !queryTooLong && validSearch(search)));

  function submit(): void {
    // Whitespace-only drafts cannot be sent; streaming never submits here
    // (the Stop button is type="button" and dispatches its own event).
    if (canSend) {
      dispatch('send');
    }
  }

  function keydown(event: KeyboardEvent): void {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault();
      // Ctrl/Cmd+Enter while streaming does nothing.
      submit();
    }
  }
</script>

<form class="composer" on:submit|preventDefault={submit}>
  <textarea bind:value rows="2" on:keydown={keydown} aria-label="Message"
    placeholder="Message Graph Horizon…"></textarea>
  <div class="composer-bar">
    <button
      class:search-active={search !== null} class="action action-search" type="button"
      disabled={!searchAvailable} aria-pressed={search !== null}
      aria-label={searchAction} title={searchAction}
      on:click={() => { search = search ? null : defaultSearch(); }}
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
        <circle cx="12" cy="12" r="9" />
        <path d="M3 12h18M12 3a15 15 0 0 1 0 18M12 3a15 15 0 0 0 0 18" />
      </svg>
      <span>{searchState}</span>
    </button>
    <span class="composer-hint">
      {queryTooLong
        ? `Search query exceeds ${searchCapability?.maxQueryCharacters} characters`
        : streaming ? 'Generating… prepare your next message' : 'Ctrl/⌘ + Enter to send'}
    </span>
    {#if streaming}
      <button class="action action-stop" type="button"
        on:click={() => dispatch('stop')} aria-label="Stop">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <rect x="6" y="6" width="12" height="12" rx="1" />
        </svg>
      </button>
    {:else}
      <button class="action action-send" type="submit" disabled={!canSend} aria-label="Send">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M22 2 11 13" />
          <path d="M22 2 15 22 11 13 2 9 22 2z" />
        </svg>
      </button>
    {/if}
  </div>
  {#if search}
    <SearchOptions
      value={search}
      provider={searchCapability?.provider ?? 'the search provider'}
      maxQueryCharacters={searchCapability?.maxQueryCharacters ?? 512}
      on:change={event => { search = event.detail; }}
    />
  {/if}
</form>

<style lang="scss">
  /* Textarea and action bar form one focusable visual unit. */
  .composer {
    border: var(--gn-border-width) solid var(--gn-border);
    border-radius: var(--gn-radius-md);
    background: var(--gn-bg-panel); clip-path: var(--gn-panel-clip);
    overflow: hidden;
  }

  .composer:focus-within {
    border-color: var(--gn-accent);
    box-shadow: var(--gn-focus-inset);
  }

  textarea {
    display: block;
    width: 100%;
    min-height: 52px;
    max-height: 160px;
    resize: vertical;
    box-sizing: border-box;
    border: none;
    outline: none;
    background: transparent var(--gn-color-rail) left bottom / 100% var(--gn-color-rail-height) no-repeat;
    padding: var(--gn-space-sm) var(--gn-space-md);
    color: var(--gn-text-primary);
    font-family: var(--gn-font-sans);
    font-size: var(--gn-text-md);
    line-height: var(--gn-line-height);
  }

  .composer-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--gn-space-sm);
    min-height: var(--gn-control-height);
    padding: var(--gn-space-xs) var(--gn-space-sm) var(--gn-space-sm);
  }

  .composer-hint {
    min-width: 0;
    flex: 1 1 auto;
    overflow: hidden;
    color: var(--gn-text-muted);
    font-family: var(--gn-font-mono);
    font-size: var(--gn-text-xs);
    letter-spacing: 0.04em;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .action-search {
    border: var(--gn-border-width) solid var(--gn-border);
    background: var(--gn-bg-panel-raised);
    color: var(--gn-text-muted);
  }

  .action.action-search {
    width: auto;
    min-width: var(--gn-control-height);
    gap: var(--gn-space-xs);
    padding: 0 var(--gn-space-sm);
    font-family: var(--gn-font-sans);
    font-size: var(--gn-text-xs);
    font-weight: 700;
    white-space: nowrap;
  }

  .action-search:hover,
  .action-search.search-active {
    border-color: var(--gn-accent);
    color: var(--gn-accent-ink);
  }

  .action-search:disabled {
    border-color: var(--gn-border); color: var(--gn-text-muted);
    box-shadow: none; cursor: default; opacity: 0.55;
  }

  .action-search.search-active {
    background: var(--gn-accent);
    color: var(--gn-bg-panel);
  }

  .action {
    width: 36px;
    height: 36px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--gn-radius-sm);
    box-shadow: var(--gn-shadow-small);
    cursor: pointer;
    flex: 0 0 auto;
  }

  .action-send {
    border: var(--gn-border-width) solid var(--gn-accent);
    background: var(--gn-accent);
    color: var(--gn-bg-panel);
  }

  .action-send:hover:not(:disabled) { background: var(--gn-accent-ink); }

  .action:focus-visible { outline: none; box-shadow: var(--gn-focus-ring); }

  .action-send:disabled {
    border-color: var(--gn-border);
    background: var(--gn-bg-panel-raised);
    color: var(--gn-text-muted);
    box-shadow: none;
    cursor: default;
  }

  /* Stop remains distinct from Send through its outlined panel treatment. */
  .action-stop {
    border: var(--gn-border-width) solid var(--gn-error-border);
    background: var(--gn-error-bg);
    color: var(--gn-error-fg);
  }

  .action-stop:hover { background: var(--gn-bg-panel-raised); }

  @media (max-width: 640px) {
    .action { width: var(--gn-touch-height); height: var(--gn-touch-height); }
    .composer-bar { min-height: var(--gn-touch-height); }
  }
</style>
