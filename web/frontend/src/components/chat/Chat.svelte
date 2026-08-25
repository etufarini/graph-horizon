<!-- Chat.svelte composes the active chat, optional Web search, files, panels,
context, transfer, status, and gated submission; domain rules remain outside. -->
<script lang="ts">
  import { onMount } from 'svelte';
  import ChatHistory from '../ChatHistory.svelte';
  import Composer from '../Composer.svelte';
  import FilesPanel from '../files/Panel.svelte';
  import Header from './Header.svelte';
  import Metrics from '../Metrics.svelte';
  import SessionActions from '../SessionActions.svelte';
  import Status from '../Status.svelte';
  import SystemPrompt from '../SystemPrompt.svelte';
  import Transcript from '../Transcript.svelte';
  import { loadRuntimeContext, loadRuntimeInfo } from '../../chat/client';
  import { contextUsage } from '../../chat/context';
  import { downloadChatFile, downloadMarkdownFile } from '../../chat/download';
  import { markdownFileOverhead } from '../../chat/files/context';
  import { markdownFiles } from '../../chat/files/state';
  import { activeChat, orderedChats } from '../../chat/sessions';
  import { chat, wireMessages } from '../../chat/state';
  import { serializeChat } from '../../chat/transfer';
  import type { RuntimeContext, RuntimeInfo, SearchSelection } from '../../chat/types';

  let draft = '';
  let runtimeContext: RuntimeContext | null = null;
  let runtimeInfo: RuntimeInfo | null = null;
  let configurationError: string | null = null;
  let historyOpen = false;
  let filesOpen = false;
  let panelsOverlay = false;
  let systemPromptOpen = false;
  let search: SearchSelection | null = null;
  let selectedFileChat = '';
  const contextController = new AbortController();

  onMount(() => {
    const panelMedia = matchMedia('(max-width: 1199px)');
    const applyPanelBreakpoint = () => { panelsOverlay = panelMedia.matches; };
    applyPanelBreakpoint();
    panelMedia.addEventListener('change', applyPanelBreakpoint);
    void markdownFiles.reconcile($chat.collection.chats.map(chat => chat.id));
    void loadRuntimeContext(contextController.signal).then(result => {
      if (contextController.signal.aborted) return;
      if (result.ok) runtimeContext = result.context;
      else configurationError = 'Context configuration unavailable';
    });
    void loadRuntimeInfo(contextController.signal).then(result => {
      if (!contextController.signal.aborted && result.ok) runtimeInfo = result.info;
    });
    return () => {
      panelMedia.removeEventListener('change', applyPanelBreakpoint);
      contextController.abort();
    };
  });

  $: streaming = $chat.status === 'streaming';
  $: currentChat = activeChat($chat.collection);
  $: messages = currentChat.messages;
  $: chats = orderedChats($chat.collection);
  $: if (currentChat.id !== selectedFileChat) {
    selectedFileChat = currentChat.id;
    void markdownFiles.select(currentChat.id);
  }
  $: filesLoaded = $markdownFiles.ready && $markdownFiles.chatId === currentChat.id;
  $: filesReady = filesLoaded && !$markdownFiles.busy;
  $: chatLocked = streaming || $markdownFiles.busy;
  $: fileOverhead = filesLoaded ? markdownFileOverhead($markdownFiles.files) : '';
  $: occupancyMessages = [
    ...wireMessages(messages, currentChat.systemPrompt, streaming ? '' : draft),
    ...(fileOverhead ? [{ role: 'user' as const, content: fileOverhead }] : [])
  ];
  $: usage = runtimeContext
    ? contextUsage(occupancyMessages, runtimeContext, search ? runtimeContext.search.maxContextCharacters : 0)
    : null;
  $: persistenceWarning = $chat.persistenceWarning === 'invalid-record'
    ? 'Invalid chat archive: starting with a new chat'
    : $chat.persistenceWarning === 'unavailable'
      ? 'Persistence unavailable: chats will remain in memory only'
      : $markdownFiles.warning === 'invalid-record'
        ? 'Invalid file archive: damaged records were removed'
        : $markdownFiles.warning === 'unavailable'
          ? 'File persistence unavailable: added files will remain in memory only'
          : null;

  async function send(): Promise<void> {
    const submitted = draft;
    draft = '';
    if (!runtimeContext || !filesReady) return;
    await chat.send(submitted, runtimeContext, $markdownFiles.files, search);
    // Restore only when no newer draft was prepared during the failed request.
    if ($chat.status === 'error' && submitted.trim() && !draft) draft = submitted;
  }

  function toggleHistory(): void {
    // Workspace invariant: opening either side panel always closes the other.
    const opening = !historyOpen;
    historyOpen = opening;
    if (opening) filesOpen = false;
  }

  function toggleFiles(): void {
    const opening = !filesOpen;
    filesOpen = opening;
    if (opening) historyOpen = false;
  }

  function regenerate(): void { if (runtimeContext && filesReady) void chat.regenerate(runtimeContext, $markdownFiles.files, search); }

  function editPrompt(userId: string, text: string): void {
    if (runtimeContext && filesReady) void chat.editPrompt(userId, text, runtimeContext, $markdownFiles.files, search);
  }

  function addFiles(selected: File[]): void {
    if (!runtimeContext || !filesReady || streaming) return;
    const base = wireMessages(messages, currentChat.systemPrompt, draft);
    void markdownFiles.add(selected, currentChat.id, base, runtimeContext);
  }
</script>

<section class:history-closed={!historyOpen} class:files-closed={!filesOpen} class="application">
  <ChatHistory {chats} activeId={$chat.collection.activeChatId} open={historyOpen} overlay={panelsOverlay}
    blocked={filesOpen} streaming={chatLocked} on:new={() => chat.newChat()}
    on:select={event => chat.selectChat(event.detail)} on:rename={event => chat.renameChat(event.detail.id, event.detail.title)}
    on:delete={event => chat.deleteChat(event.detail)} on:toggle={toggleHistory} on:close={() => historyOpen = false} />

  <section class="chat-layout" inert={panelsOverlay && (historyOpen || filesOpen)}>
    <Header {runtimeInfo} />

    <div class:prompt-open={systemPromptOpen} class="chat-tools">
      <SystemPrompt bind:open={systemPromptOpen} value={currentChat.systemPrompt} disabled={chatLocked} on:change={event => chat.setSystemPrompt(event.detail)} />
      <SessionActions importDisabled={chatLocked} on:export={() => downloadChatFile(serializeChat(messages, currentChat.systemPrompt))} on:import={event => chat.importChat(event.detail)} />
    </div>
    <Transcript {messages} {streaming} searchEnabled={runtimeContext !== null} on:regenerate={regenerate} on:edit={event => editPrompt(event.detail.userId, event.detail.text)} on:delete={() => { if (filesReady) chat.deleteLastTurn(); }} />
    <div class="feedback">
      {#if persistenceWarning}<div class="persistence-warning" role="status">{persistenceWarning}</div>{/if}
      <Status warning={null} error={configurationError ?? $chat.error ?? $markdownFiles.error} {usage} />
      <Metrics telemetry={$chat.telemetry} />
    </div>
    <Composer bind:value={draft} bind:search {streaming} searchCapability={runtimeContext?.search ?? null} contextAvailable={runtimeContext !== null && filesReady} on:send={send} on:stop={() => chat.stop()} />
  </section>

  <FilesPanel files={$markdownFiles.files} open={filesOpen} overlay={panelsOverlay} blocked={historyOpen}
    disabled={streaming || runtimeContext === null} busy={$markdownFiles.busy} ready={filesLoaded}
    on:add={event => addFiles(event.detail)} on:download={event => downloadMarkdownFile(event.detail.name, event.detail.content)}
    on:delete={event => markdownFiles.remove(event.detail)} on:toggle={toggleFiles} on:close={() => filesOpen = false} />
</section>

<style lang="scss">
  .application {
    width: 100%;
    height: 100%; min-height: 0; min-width: 0;
    display: flex; gap: 0;
    overflow: hidden;
    --gn-header-start-reserve: 0px;
    --gn-header-end-reserve: 0px;
  }
  .application.history-closed { --gn-header-start-reserve: calc(var(--gn-panel-rail-width) + var(--gn-space-sm)); }
  .application.files-closed { --gn-header-end-reserve: calc(var(--gn-panel-rail-width) + var(--gn-space-sm)); }
  .chat-layout {
    flex: 1; min-width: 0; height: 100%; min-height: 0;
    display: grid;
    grid-template-rows: auto auto minmax(0, 1fr) auto auto;
    background: var(--gn-bg-page);
    padding: var(--gn-space-sm) var(--gn-space-md);
    row-gap: var(--gn-space-sm);
  }
  .chat-tools { min-width: 0; display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: start; gap: var(--gn-space-sm); }
  .chat-tools.prompt-open { grid-template-columns: 1fr; }
  .prompt-open :global(.session-actions) { display: none; }
  .feedback { min-width: 0; display: flex; flex-wrap: wrap; align-items: stretch; gap: var(--gn-space-xs) var(--gn-space-lg); border-top: var(--gn-rule-width) solid var(--gn-border-subtle); background: var(--gn-bg-panel); }
  .feedback :global(.status-stack) { min-width: min(280px, 100%); flex: 1 1 320px; }
  .feedback :global(.metrics) { min-width: min(320px, 100%); flex: 1 1 420px; }
  .persistence-warning {
    flex: 1 0 100%;
    border: var(--gn-rule-width) solid var(--gn-warning); border-radius: var(--gn-radius-sm); background: var(--gn-warning-bg);
    padding: var(--gn-space-sm); color: var(--gn-warning);
    font-size: var(--gn-text-sm); font-weight: 600;
  }
  @media (max-width: 640px) {
    .chat-layout { padding: var(--gn-space-sm); }
  }
</style>
