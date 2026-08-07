<script lang="ts">
  /*
   * Chat.svelte
   * Single responsibility: initialize context and compose transcript, session
   * actions, persistence warning, status, and capacity-gated submission.
   */
  import Composer from './Composer.svelte';
  import SessionActions from './SessionActions.svelte';
  import Status from './Status.svelte';
  import SystemPrompt from './SystemPrompt.svelte';
  import Transcript from './Transcript.svelte';
  import { onDestroy, onMount } from 'svelte';
  import { loadRuntimeContext } from '../chat/client';
  import { contextUsage } from '../chat/context';
  import { downloadChatFile } from '../chat/download';
  import { chat, wireMessages } from '../chat/state';
  import { serializeChat } from '../chat/transfer';
  import type { RuntimeContext } from '../chat/types';

  let draft = '';
  let runtimeContext: RuntimeContext | null = null;
  let configurationError: string | null = null;
  const contextController = new AbortController();

  onMount(async () => {
    const result = await loadRuntimeContext(contextController.signal);
    if (contextController.signal.aborted) {
      return;
    }
    if (result.ok) {
      runtimeContext = result.context;
    } else {
      configurationError =
        result.error === 'no-prompt-space'
          ? 'max_tokens non lascia spazio al prompt'
          : 'Configurazione del contesto non disponibile';
    }
  });

  onDestroy(() => contextController.abort());

  $: streaming = $chat.status === 'streaming';
  $: usage = runtimeContext
    ? contextUsage(wireMessages($chat.messages, $chat.systemPrompt, streaming ? '' : draft), runtimeContext)
    : null;

  async function send(): Promise<void> {
    const submitted = draft;
    draft = '';
    if (!runtimeContext) {
      return;
    }
    await chat.send(submitted, runtimeContext);
    // A failed request restores its prompt only if the user has not already
    // prepared the next draft while the request was running.
    if ($chat.status === 'error' && submitted.trim() && !draft) {
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
      <span class="chat-brand">Graph Horizon</span>
      <span class="chat-divider" aria-hidden="true">//</span>
      <span class="chat-sub">console d'inferenza locale</span>
    </h1>
  </header>

  <SystemPrompt value={$chat.systemPrompt} on:change={event => chat.setSystemPrompt(event.detail)} />

  <SessionActions
    importDisabled={streaming}
    confirmBeforeImport={$chat.messages.length > 0}
    hasMessages={$chat.messages.length > 0}
    on:reset={() => chat.newChat()}
    on:export={() => downloadChatFile(serializeChat($chat.messages, $chat.systemPrompt))}
    on:import={event => chat.importChat(event.detail)}
  />

  <Transcript messages={$chat.messages} {streaming} />

  <Status
    warning={$chat.persistenceWarning}
    error={configurationError ?? $chat.error}
    {usage}
    generationStartedAt={$chat.generationStartedAt}
    generationMs={$chat.generationMs}
  />

  <Composer
    bind:value={draft}
    {streaming}
    contextAvailable={runtimeContext !== null}
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
