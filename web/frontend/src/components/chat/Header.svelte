<!--
Chat header: present brand, runtime identity, and accessible history/file panel
toggles as typed intents. Responsive state, focus restoration, chat lifecycle,
and panel contents remain owned by the parent composition boundary.
-->
<script lang="ts">
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
    <button type="button" bind:this={historyToggle} aria-label="Show chat history" aria-expanded={historyOpen} aria-controls="chat-history" on:click={() => dispatch('history')}>Chat</button>
    <button type="button" bind:this={filesToggle} aria-label="Show Markdown files" aria-expanded={filesOpen} aria-controls="markdown-files" on:click={() => dispatch('files')}>Files · {fileCount}</button>
    <h1>
      <span class="brand">Graph Horizon</span>
      <span class="divider" aria-hidden="true">//</span>
      <span class="sub">local inference console</span>
    </h1>
  </div>
  {#if runtimeInfo}<Runtime info={runtimeInfo} />{/if}
</header>

<style lang="scss">
  header { display: grid; gap: var(--gn-space-xs); border-bottom: var(--gn-rule-width) solid var(--gn-border); padding-bottom: var(--gn-space-sm); }
  .main { display: flex; align-items: center; gap: var(--gn-space-sm); }
  button { border: var(--gn-border-width) solid var(--gn-border); border-radius: var(--gn-radius-sm); background: var(--gn-bg-panel); padding: var(--gn-space-xs) var(--gn-space-sm); color: var(--gn-text-muted); box-shadow: var(--gn-shadow-small); cursor: pointer; font: 700 var(--gn-text-xs) var(--gn-font-mono); letter-spacing: 0.08em; text-transform: uppercase; }
  button:focus-visible { outline: none; box-shadow: var(--gn-focus-ring), var(--gn-shadow-small); }
  h1 { min-width: 0; margin: 0; display: flex; flex-wrap: wrap; align-items: center; gap: var(--gn-space-sm); font-size: var(--gn-text-md); line-height: 1.2; }
  .brand { color: var(--gn-accent-ink); font-family: var(--gn-font-mono); font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; }
  .divider { color: var(--gn-border); font-family: var(--gn-font-mono); font-weight: 700; }
  .sub { color: var(--gn-text-muted); font-weight: 500; letter-spacing: 0.02em; }
</style>
