<!--
Chat.svelte is the sole Web UI composition boundary for active-chat prompt and
transcript derivation, responsive history, context, transfer, status, and gated
submission. Collection rules, transport, and storage schemas remain outside.
-->
<script lang="ts">
  import { onMount, tick } from 'svelte';
  import ChatHistory from './ChatHistory.svelte';
  import Composer from './Composer.svelte';
  import SessionActions from './SessionActions.svelte';
  import Status from './Status.svelte';
  import SystemPrompt from './SystemPrompt.svelte';
  import Transcript from './Transcript.svelte';
  import { loadRuntimeContext } from '../chat/client';
  import { contextUsage } from '../chat/context';
  import { downloadChatFile } from '../chat/download';
  import { activeChat, orderedChats } from '../chat/sessions';
  import { chat, wireMessages } from '../chat/state';
  import { serializeChat } from '../chat/transfer';
  import type { RuntimeContext } from '../chat/types';

  let draft = '';
  let runtimeContext: RuntimeContext | null = null;
  let configurationError: string | null = null;
  let historyOpen = false;
  let mobile = false;
  let historyToggle: HTMLButtonElement;
  const contextController = new AbortController();

  onMount(() => {
    const media = matchMedia('(max-width: 720px)');
    const applyBreakpoint = () => {
      mobile = media.matches;
      historyOpen = !mobile;
    };
    applyBreakpoint();
    media.addEventListener('change', applyBreakpoint);
    void loadRuntimeContext(contextController.signal).then(result => {
      if (contextController.signal.aborted) return;
      if (result.ok) runtimeContext = result.context;
      else configurationError = 'Configurazione del contesto non disponibile';
    });
    return () => {
      media.removeEventListener('change', applyBreakpoint);
      contextController.abort();
    };
  });

  $: streaming = $chat.status === 'streaming';
  $: currentChat = activeChat($chat.collection);
  $: messages = currentChat.messages;
  $: chats = orderedChats($chat.collection);
  $: usage = runtimeContext
    ? contextUsage(wireMessages(messages, currentChat.systemPrompt, streaming ? '' : draft), runtimeContext)
    : null;
  $: persistenceWarning = $chat.persistenceWarning === 'invalid-record'
    ? 'Archivio chat non valido: avvio con una nuova chat'
    : $chat.persistenceWarning === 'unavailable'
      ? 'Persistenza non disponibile: le chat resteranno solo in memoria'
      : null;

  async function send(): Promise<void> {
    const submitted = draft;
    draft = '';
    if (!runtimeContext) return;
    await chat.send(submitted, runtimeContext);
    // Restore only when no newer draft was prepared during the failed request.
    if ($chat.status === 'error' && submitted.trim() && !draft) draft = submitted;
  }

  function closeHistory(): void {
    historyOpen = false;
    void tick().then(() => historyToggle?.focus());
  }

  function selectChat(id: string): void {
    chat.selectChat(id);
    if (mobile) closeHistory();
  }

  function regenerate(): void {
    if (runtimeContext) void chat.regenerate(runtimeContext);
  }

  function editLastPrompt(text: string): void {
    if (runtimeContext) void chat.editLastPrompt(text, runtimeContext);
  }
</script>

<section class="application">
  <ChatHistory
    {chats}
    activeId={$chat.collection.activeChatId}
    open={historyOpen}
    {streaming}
    on:new={() => chat.newChat()}
    on:select={event => selectChat(event.detail)}
    on:rename={event => chat.renameChat(event.detail.id, event.detail.title)}
    on:delete={event => chat.deleteChat(event.detail)}
    on:close={closeHistory}
  />

  <section class="chat-layout">
    <header class="chat-header">
      <button class="history-toggle" type="button" bind:this={historyToggle} aria-label="Mostra cronologia chat" aria-expanded={historyOpen} aria-controls="chat-history" on:click={() => historyOpen = !historyOpen}>Chat</button>
      <h1 class="chat-title">
        <span class="chat-brand">Graph Horizon</span>
        <span class="chat-divider" aria-hidden="true">//</span>
        <span class="chat-sub">console d'inferenza locale</span>
      </h1>
    </header>

    <SystemPrompt value={currentChat.systemPrompt} disabled={streaming} on:change={event => chat.setSystemPrompt(event.detail)} />
    <SessionActions importDisabled={streaming} on:export={() => downloadChatFile(serializeChat(messages, currentChat.systemPrompt))} on:import={event => chat.importChat(event.detail)} />
    <Transcript {messages} {streaming} on:regenerate={regenerate} on:edit={event => editLastPrompt(event.detail)} on:delete={() => chat.deleteLastTurn()} />
    {#if persistenceWarning}<div class="persistence-warning" role="status">{persistenceWarning}</div>{/if}
    <Status warning={null} error={configurationError ?? $chat.error} {usage} generationStartedAt={$chat.generationStartedAt} generationMs={$chat.generationMs} />
    <Composer bind:value={draft} {streaming} contextAvailable={runtimeContext !== null} on:send={send} on:stop={() => chat.stop()} />
  </section>
</section>

<style lang="scss">
  .application {
    width: min(var(--gn-application-width), 100%);
    height: 100%; min-height: 0; min-width: 0;
    display: flex; gap: var(--gn-space-md);
  }
  .chat-layout {
    flex: 1; min-width: 0; height: 100%; min-height: 0;
    display: grid;
    grid-template-rows: auto auto auto minmax(0, 1fr) auto auto auto;
    gap: var(--gn-space-md);
  }
  .chat-header {
    display: flex; align-items: center; gap: var(--gn-space-sm);
    border-bottom: var(--gn-rule-width) solid var(--gn-border);
    padding-bottom: var(--gn-space-sm);
  }
  .history-toggle {
    border: var(--gn-border-width) solid var(--gn-border); border-radius: var(--gn-radius-sm);
    background: var(--gn-bg-panel); padding: var(--gn-space-xs) var(--gn-space-sm);
    color: var(--gn-text-muted); box-shadow: var(--gn-shadow-small); cursor: pointer;
    font: 700 var(--gn-text-xs) var(--gn-font-mono); letter-spacing: 0.08em; text-transform: uppercase;
  }
  .history-toggle:focus-visible { outline: none; box-shadow: var(--gn-focus-ring), var(--gn-shadow-small); }
  .chat-title {
    min-width: 0; margin: 0; display: flex; flex-wrap: wrap; align-items: center;
    gap: var(--gn-space-sm); font-size: var(--gn-text-md); line-height: 1.2;
  }
  .chat-brand { color: var(--gn-accent-ink); font-family: var(--gn-font-mono); font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; }
  .chat-divider { color: var(--gn-border); font-family: var(--gn-font-mono); font-weight: 700; }
  .chat-sub { color: var(--gn-text-muted); font-weight: 500; letter-spacing: 0.02em; }
  .persistence-warning {
    border: var(--gn-border-width) solid var(--gn-streaming); background: var(--gn-bg-panel);
    padding: var(--gn-space-sm) var(--gn-space-md); color: var(--gn-text-primary);
    font-size: var(--gn-text-sm); font-weight: 600;
  }
</style>
