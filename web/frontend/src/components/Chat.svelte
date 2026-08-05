<script lang="ts">
  /*
   * Chat.svelte
   * Single responsibility: compose the text-only chat page from transcript,
   * status, system prompt, session actions, and composer. It does not wire
   * tools, workspace controls, confirmations, or reasoning views. The first
   * row remains the textual brand lockup.
   */
  import Composer from './Composer.svelte';
  import SessionActions from './SessionActions.svelte';
  import Status from './Status.svelte';
  import SystemPrompt from './SystemPrompt.svelte';
  import Transcript from './Transcript.svelte';
  import { downloadChatFile } from '../chat/download';
  import { chat } from '../chat/state';
  import { serializeChat } from '../chat/transfer';

  let draft = '';

  $: streaming = $chat.status === 'streaming';

  $: lastMessage = $chat.messages[$chat.messages.length - 1];
  $: streamChars =
    streaming && lastMessage?.role === 'assistant' ? lastMessage.content.length : 0;

  async function send(): Promise<void> {
    const submitted = draft;
    draft = '';
    await chat.send(submitted);
    if ($chat.status === 'error' && submitted.trim()) {
      draft = submitted;
    }
  }

  function stop(): void {
    chat.stop();
  }
</script>

<section class="chat-layout">
  <header class="chat-header">
    <h1 class="chat-title">
      <span class="chat-brand">Graph Orizon</span>
      <span class="chat-divider" aria-hidden="true">//</span>
      <span class="chat-sub">console d'inferenza locale</span>
    </h1>
  </header>

  <SystemPrompt value={$chat.systemPrompt} on:change={event => chat.setSystemPrompt(event.detail)} />

  <SessionActions
    importDisabled={streaming}
    confirmBeforeImport={$chat.messages.length > 0}
    on:export={() => downloadChatFile(serializeChat($chat.messages, $chat.systemPrompt))}
    on:import={event => chat.importChat(event.detail)}
  />

  <Transcript messages={$chat.messages} {streaming} />

  <Status status={$chat.status} error={$chat.error} stats={$chat.stats} {streamChars} />

  <Composer
    bind:value={draft}
    {streaming}
    on:send={send}
    on:stop={stop}
  />
</section>

<style lang="scss">
  .chat-layout {
    width: min(1040px, 100%);
    /* Fills the shell exactly; minmax(0,1fr) lets the transcript shrink and
       scroll instead of pushing the composer out of the viewport. */
    height: 100%;
    min-height: 0;
    display: grid;
    grid-template-rows: auto auto auto minmax(0, 1fr) auto auto;
    gap: var(--gn-space-md);
  }

  .chat-header {
    border-bottom: var(--gn-rule-width) solid var(--gn-border);
    padding-bottom: var(--gn-space-sm);
  }

  /* Single textual lockup: brand + divider + role, one h1. */
  .chat-title {
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--gn-space-sm);
    font-size: var(--gn-text-md);
    line-height: 1.2;
  }

  .chat-brand {
    color: var(--gn-accent-ink);
    font-family: var(--gn-font-mono);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }

  .chat-divider {
    color: var(--gn-border);
    font-family: var(--gn-font-mono);
    font-weight: 700;
  }

  .chat-sub {
    color: var(--gn-text-muted);
    font-weight: 500;
    letter-spacing: 0.02em;
  }
</style>
