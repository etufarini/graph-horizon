<script lang="ts">
  /*
   * Bubble.svelte
   * Single responsibility: render one user or assistant text message,
   * including its display-only Reasoning view. Tools, confirmations, workspace
   * data, raw chat ownership, and HTML sanitization remain outside this file.
   */
  import Markdown from './Markdown.svelte';
  import { splitReasoning } from '../chat/reasoning';
  import type { ChatMessage } from '../chat/types';

  export let message: ChatMessage;
  export let streaming = false;

  $: isUser = message.role === 'user';
  $: label = isUser ? 'Tu' : 'Graph Horizon';
  // Only assistant presentation derives sections; the raw message stays intact.
  $: reasoning = isUser ? null : splitReasoning(message.content, streaming);
</script>

<article class={`bubble ${isUser ? 'bubble-user' : 'bubble-assistant'}`}>
  <div class="bubble-label">{label}</div>
  {#if isUser}
    <p class="user-text">{message.content}</p>
  {:else if reasoning}
    {#if reasoning.thinking !== undefined}
      <details class="reasoning">
        <summary>THINK</summary>
        <Markdown content={reasoning.thinking} />
      </details>
    {/if}
    {#if !reasoning.pending}
      {#if reasoning.incomplete}
        <p class="placeholder">Risposta incompleta</p>
      {:else if !streaming && reasoning.answer.trim() === ''}
        <p class="placeholder">Nessuna risposta</p>
      {:else if reasoning.answer !== ''}
        <Markdown content={reasoning.answer} />
      {/if}
    {/if}
    {#if streaming}
      <!-- Indicator only on the trailing assistant bubble, never on user bubbles. -->
      <span class="cursor" aria-hidden="true"></span>
    {/if}
  {/if}
</article>

<style lang="scss">
  .bubble {
    width: fit-content;
    max-width: min(78%, 760px);
    min-width: 120px;
    box-sizing: border-box;
    border: var(--gn-border-width) solid var(--gn-border);
    border-radius: var(--gn-radius-sm);
    padding: var(--gn-space-sm) var(--gn-space-md);
    background: var(--gn-bg-panel);
  }

  .bubble-assistant {
    align-self: flex-start;
    border-left-color: var(--gn-accent-ink);
    /* The assistant surface owns the theme's single stepped corner. */
    clip-path: var(--gn-panel-clip);
  }

  .bubble-user {
    align-self: flex-end;
    background: var(--gn-user-bg);
    border-color: var(--gn-user-border);
    box-shadow: var(--gn-shadow-hard);
  }

  .bubble-label {
    margin-bottom: var(--gn-space-xs);
    font-family: var(--gn-font-mono);
    font-size: var(--gn-text-xs);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--gn-text-muted);
  }

  .bubble-assistant .bubble-label {
    color: var(--gn-accent-ink);
  }

  .user-text {
    margin: 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }

  .placeholder {
    margin: 0;
    color: var(--gn-text-muted);
    font-style: italic;
  }

  .reasoning {
    margin-bottom: var(--gn-space-sm);
    border-left: var(--gn-border-width) solid var(--gn-text-muted);
    background: var(--gn-bg-panel-raised);
    padding: var(--gn-space-xs) var(--gn-space-sm);
    color: var(--gn-text-muted);
  }

  .reasoning summary {
    cursor: pointer;
    font-family: var(--gn-font-mono);
    font-size: var(--gn-text-xs);
    font-weight: 700;
    letter-spacing: 0.08em;
  }

  .cursor {
    display: inline-block;
    width: 0.6em;
    height: 0.9em;
    margin-top: var(--gn-space-xs);
    vertical-align: text-bottom;
    background: var(--gn-accent);
    animation: pulse var(--gn-motion-pulse) ease-in-out infinite;
  }

  @keyframes pulse {
    0%,
    100% {
      opacity: 1;
    }
    50% {
      opacity: var(--gn-motion-low-opacity);
    }
  }

  @media (max-width: 720px) {
    .bubble {
      max-width: 100%;
    }
  }
</style>
