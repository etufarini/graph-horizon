<script lang="ts">
  /*
   * Composer.svelte
   * Single responsibility: keep the next draft and its explicit Web-search
   * choice editable while exposing send or stop for the active request.
   */
  // @ts-expect-error Vite resolves this local asset and fails the build if it is missing.
  import logoUrl from '../../../../assets/graph-horizon-logo.svg';
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
  <textarea
    bind:value
    rows="3"
    on:keydown={keydown}
    aria-label="Message"
    placeholder="Message Graph Horizon…"
  ></textarea>
  <div class="composer-bar">
    <img class="composer-logo" src={logoUrl} alt="" aria-hidden="true" />
    <button
      class:search-active={search !== null}
      class="action action-search"
      type="button"
      disabled={!searchAvailable}
      aria-pressed={search !== null}
      aria-label={!searchAvailable ? 'Web search is not configured' : search ? 'Disable Web search' : 'Enable Web search'}
      on:click={() => { search = search ? null : defaultSearch(); }}
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
        <circle cx="12" cy="12" r="9" />
        <path d="M3 12h18M12 3a15 15 0 0 1 0 18M12 3a15 15 0 0 0 0 18" />
      </svg>
    </button>
    <span class="composer-hint">
      {queryTooLong
        ? `Search query exceeds ${searchCapability?.maxQueryCharacters} characters`
        : streaming ? 'Generating… prepare your next message' : 'Ctrl/⌘ + Enter to send'}
    </span>
    {#if streaming}
      <button
        class="action action-stop"
        type="button"
        on:click={() => dispatch('stop')}
        aria-label="Stop"
      >
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
  /* The panel is the single visual unit: textarea and action bar live
     inside it, and focus styling applies to the whole group. */
  .composer {
    border: var(--gn-border-width) solid var(--gn-border);
    border-radius: var(--gn-radius-sm);
    background: var(--gn-bg-panel);
    /* The grouped composer owns the theme's single stepped corner. */
    clip-path: var(--gn-panel-clip);
  }

  .composer:focus-within {
    box-shadow: var(--gn-focus-inset);
  }

  textarea {
    display: block;
    width: 100%;
    min-height: 56px;
    max-height: 160px;
    resize: vertical;
    box-sizing: border-box;
    border: none;
    outline: none;
    background-color: transparent;
    background-image: var(--gn-color-rail);
    background-position: left bottom;
    background-repeat: no-repeat;
    background-size: 100% var(--gn-color-rail-height);
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
    padding: 0 var(--gn-space-sm) var(--gn-space-sm) var(--gn-space-md);
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

  .composer-logo {
    width: var(--gn-composer-logo-size);
    height: var(--gn-composer-logo-size);
    flex: 0 0 auto;
    object-fit: contain;
  }

  .action-search {
    border: var(--gn-border-width) solid var(--gn-border);
    background: var(--gn-bg-panel-raised);
    color: var(--gn-text-muted);
  }

  .action-search:hover,
  .action-search.search-active {
    border-color: var(--gn-accent);
    color: var(--gn-text-primary);
  }

  .action-search:disabled {
    border-color: var(--gn-border); color: var(--gn-text-muted);
    box-shadow: none; cursor: default; opacity: 0.55;
  }

  .action-search.search-active {
    background: var(--gn-accent);
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
    border: var(--gn-border-width) solid var(--gn-border);
    background: var(--gn-accent);
    color: var(--gn-text-primary);
  }

  .action-send:hover:not(:disabled) {
    background: var(--gn-accent-bright);
  }

  .action:focus-visible {
    outline: none;
    box-shadow: var(--gn-focus-ring), var(--gn-shadow-small);
  }

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

  .action-stop:hover {
    background: var(--gn-bg-panel-raised);
  }

  @media (max-width: 720px) {
    .composer-logo {
      width: var(--gn-composer-logo-size-mobile);
      height: var(--gn-composer-logo-size-mobile);
    }
  }
</style>
