<!--
Transcript.svelte renders validated complete pairs through Turn, delegates turn
intents, and owns pinned/autoscroll behavior. State mutation, chat
history, transcript repair, and persistence remain outside this component.
-->
<script lang="ts">
  import { afterUpdate, createEventDispatcher } from 'svelte';
  import Turn from './Turn.svelte';
  import type { ChatMessage } from '../chat/types';

  export let messages: ChatMessage[] = [];
  export let streaming = false;
  export let searchEnabled = false;

  const dispatch = createEventDispatcher<{
    regenerate: void;
    edit: { userId: string; text: string };
    delete: void;
  }>();
  let transcript: HTMLElement;
  let pinned = true;
  let firstId: string | undefined;
  let lastId: string | undefined;
  $: turns = Array.from({ length: messages.length / 2 }, (_, index) =>
    [messages[index * 2], messages[index * 2 + 1]] as [ChatMessage, ChatMessage]
  );

  function onScroll(): void {
    // User/programmatic scroll events own pin state; content growth cannot
    // silently unpin the viewport during a stream.
    pinned = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight <= 40;
  }

  afterUpdate(() => {
    if (!transcript) return;
    const nextFirst = messages[0]?.id;
    const nextLast = messages.at(-1)?.id;
    const transcriptChanged = nextFirst !== firstId || nextLast !== lastId;
    firstId = nextFirst;
    lastId = nextLast;
    if (transcriptChanged || pinned) {
      transcript.scrollTop = transcript.scrollHeight;
    }
  });
</script>

<section class="transcript" aria-label="Conversation" bind:this={transcript} on:scroll={onScroll}>
  {#if turns.length === 0}
    <div class="empty-state">
      <span class="empty-mark" aria-hidden="true"></span>
      <h2>Inference engine ready</h2>
      <p>{searchEnabled
        ? 'Inference stays local. Search sends only the displayed query to the selected provider.'
        : 'Send a message to start the local session.'}</p>
    </div>
  {:else}
    {#each turns as turn, index (turn[0].id)}
      <Turn
        user={turn[0]}
        assistant={turn[1]}
        final={index === turns.length - 1}
        {streaming}
        on:regenerate={() => dispatch('regenerate')}
        on:edit={event => dispatch('edit', event.detail)}
        on:delete={() => dispatch('delete')}
      />
    {/each}
  {/if}
</section>

<style lang="scss">
  .transcript {
    min-height: 0;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: var(--gn-space-md);
    padding: var(--gn-space-sm) var(--gn-space-xs) var(--gn-space-md);
  }

  .empty-state {
    margin: auto;
    width: min(420px, 100%);
    text-align: center;
    color: var(--gn-text-muted);
  }

  .empty-mark {
    display: inline-block;
    width: 12px;
    height: 12px;
    margin-bottom: var(--gn-space-md);
    border-radius: 0;
    background: var(--gn-ready);
    animation: idle-pulse var(--gn-motion-idle) ease-in-out infinite;
  }

  @keyframes idle-pulse {
    0%,
    100% { opacity: 1; }
    50% { opacity: var(--gn-motion-low-opacity); }
  }

  .empty-state h2 {
    margin: 0 0 var(--gn-space-sm);
    color: var(--gn-ready-ink);
    font-family: var(--gn-font-mono);
    font-size: var(--gn-text-lg);
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  .empty-state p {
    margin: 0;
    font-size: var(--gn-text-sm);
  }
</style>
