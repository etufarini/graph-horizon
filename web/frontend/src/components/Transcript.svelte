<script lang="ts">
  /*
   * Transcript.svelte
   * Single responsibility: render the ordered user/assistant transcript and
   * keep scroll position ergonomic. It does not render tools, workspace state,
   * confirmations, or a separate reasoning channel.
   */
  import { afterUpdate } from 'svelte';
  import Bubble from './Bubble.svelte';
  import type { ChatMessage } from '../chat/types';

  export let messages: ChatMessage[] = [];
  export let streaming = false;

  let transcript: HTMLDivElement;
  let pinned = true;
  let seen = messages.length;

  function onScroll(): void {
    // Pinned is recomputed only on scroll events (user or programmatic),
    // never on content growth, so textarea/bubble resize cannot flip it
    // mid-stream.
    pinned = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight <= 40;
  }

  afterUpdate(() => {
    if (!transcript) {
      return;
    }
    const grew = messages.length > seen;
    seen = messages.length;
    // New message sent → scroll unconditionally; stream delta → only when
    // pinned. Initial empty render scrolls nothing (grew=false, no overflow).
    if (grew || pinned) {
      transcript.scrollTop = transcript.scrollHeight;
    }
  });
</script>

<div class="transcript" bind:this={transcript} on:scroll={onScroll}>
  {#if messages.length === 0}
    <div class="empty-state">
      <span class="empty-mark" aria-hidden="true"></span>
      <h2>Motore di inferenza pronto</h2>
      <p>Scrivi un messaggio per avviare la sessione: tutto gira in locale.</p>
    </div>
  {:else}
    {#each messages as message, index (message.id)}
      <Bubble {message} streaming={streaming && index === messages.length - 1} />
    {/each}
  {/if}
</div>

<style lang="scss">
  .transcript {
    min-height: 0;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: var(--gn-space-md);
    /* Horizontal breathing room keeps hard shadows and corners unclipped. */
    padding: var(--gn-space-sm) var(--gn-space-xs);
  }

  .empty-state {
    margin: auto;
    width: min(420px, 100%);
    text-align: center;
    color: var(--gn-text-muted);
  }

  /* Square idle mark, animated only when reduced motion is not requested. */
  .empty-mark {
    display: inline-block;
    width: 16px;
    height: 16px;
    margin-bottom: var(--gn-space-md);
    background: var(--gn-ready);
    animation: idle-pulse var(--gn-motion-idle) ease-in-out infinite;
  }

  @keyframes idle-pulse {
    0%,
    100% {
      opacity: 1;
    }
    50% {
      opacity: var(--gn-motion-low-opacity);
    }
  }

  .empty-state h2 {
    margin: 0 0 var(--gn-space-sm);
    color: var(--gn-ready-ink);
    font-family: var(--gn-font-mono);
    font-size: var(--gn-text-lg);
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }

  .empty-state p {
    margin: 0;
    font-size: var(--gn-text-sm);
  }
</style>
