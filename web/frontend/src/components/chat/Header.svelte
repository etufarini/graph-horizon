<!--
Chat header: present brand, runtime identity, and accessible history/file panel
toggles as typed intents. Responsive state, focus restoration, chat lifecycle,
and panel contents remain owned by the parent composition boundary.
-->
<script lang="ts">
  // @ts-expect-error Vite resolves this local asset and fails the build if it is missing.
  import logoUrl from '../../../../../assets/graph-horizon-logo.svg';
  import { createEventDispatcher } from 'svelte';
  import Runtime from '../Runtime.svelte';
  import type { RuntimeInfo } from '../../chat/types';

  export let historyOpen = false;
  export let filesOpen = false;
  export let fileCount = 0;
  export let runtimeInfo: RuntimeInfo | null = null;
  export let historyToggle: HTMLButtonElement;
  export let filesToggle: HTMLButtonElement;

  const dispatch = createEventDispatcher<{ history: void; files: void }>();
</script>

<header>
  <div class="main">
    <div class="identity">
      <img src={logoUrl} alt="" aria-hidden="true" />
      <h1><span>Graph Horizon</span><small>Local inference</small></h1>
    </div>
    <nav aria-label="Workspace panels">
      <button type="button" bind:this={historyToggle} aria-label={historyOpen ? 'Hide chat history' : 'Show chat history'} aria-expanded={historyOpen} aria-controls="chat-history" on:click={() => dispatch('history')}>Chats</button>
      <button type="button" bind:this={filesToggle} aria-label={filesOpen ? 'Hide Markdown files' : 'Show Markdown files'} aria-expanded={filesOpen} aria-controls="markdown-files" on:click={() => dispatch('files')}>Files <span aria-hidden="true">{fileCount}</span></button>
    </nav>
  </div>
  {#if runtimeInfo}<Runtime info={runtimeInfo} />{/if}
</header>

<style lang="scss">
  header { min-width: 0; display: grid; gap: var(--gn-space-xs); border-bottom: var(--gn-border-width) solid var(--gn-border); padding-bottom: var(--gn-space-sm); }
  .main { min-width: 0; display: flex; align-items: center; justify-content: space-between; gap: var(--gn-space-sm); }
  .identity { min-width: 0; display: flex; align-items: center; gap: var(--gn-space-sm); }
  img { width: var(--gn-brand-logo-size); height: var(--gn-brand-logo-size); flex: 0 0 auto; object-fit: contain; }
  h1 { min-width: 0; margin: 0; display: flex; align-items: baseline; gap: var(--gn-space-sm); color: var(--gn-text-primary); font-size: var(--gn-text-md); line-height: 1.2; }
  h1 span { font-family: var(--gn-font-mono); font-weight: 750; letter-spacing: 0.08em; text-transform: uppercase; }
  h1 small { color: var(--gn-text-muted); font-family: var(--gn-font-mono); font-size: var(--gn-text-xs); font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; }
  nav { display: flex; gap: var(--gn-space-xs); }
  button { min-height: var(--gn-control-height); border: var(--gn-border-width) solid var(--gn-border); border-radius: var(--gn-radius-sm); background: var(--gn-bg-panel); box-shadow: var(--gn-shadow-small); padding: var(--gn-space-xs) var(--gn-space-sm); color: var(--gn-text-primary); cursor: pointer; font: 700 var(--gn-text-xs) var(--gn-font-mono); letter-spacing: 0.06em; text-transform: uppercase; }
  button span { min-width: 1.4em; display: inline-block; margin-left: 2px; border-radius: 0; background: var(--gn-bg-panel-raised); color: var(--gn-text-muted); text-align: center; }
  button:hover, button[aria-expanded='true'] { border-color: var(--gn-accent); background: var(--gn-accent-soft); }
  button:focus-visible { outline: none; box-shadow: var(--gn-focus-ring); }
  @media (max-width: 640px) {
    h1 small { display: none; }
    button { min-height: var(--gn-touch-height); }
  }
</style>
