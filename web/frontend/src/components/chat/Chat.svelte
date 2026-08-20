<!--
Chat.svelte is the sole Web UI composition boundary for active-chat prompt,
transcript, Markdown files, responsive panels, context, transfer, status, and
gated submission. Collection rules, transport, and storage schemas remain outside.
-->
<script lang="ts">
  import { onMount, tick } from 'svelte';
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
  import type { RuntimeContext, RuntimeInfo } from '../../chat/types';

  let draft = '';
  let runtimeContext: RuntimeContext | null = null;
  let runtimeInfo: RuntimeInfo | null = null;
  let configurationError: string | null = null;
  let historyOpen = false;
  let mobile = false;
  let filesOpen = false;
  let filesOverlay = false;
  let selectedFileChat = '';
  let historyToggle: HTMLButtonElement;
  let filesToggle: HTMLButtonElement;
  const contextController = new AbortController();

  onMount(() => {
    const historyMedia = matchMedia('(max-width: 720px)');
    const filesMedia = matchMedia('(max-width: 1180px)');
    const applyHistoryBreakpoint = () => {
      mobile = historyMedia.matches;
      historyOpen = !mobile;
    };
    const applyFilesBreakpoint = () => {
      filesOverlay = filesMedia.matches;
      filesOpen = !filesOverlay;
    };
    applyHistoryBreakpoint();
    applyFilesBreakpoint();
    historyMedia.addEventListener('change', applyHistoryBreakpoint);
    filesMedia.addEventListener('change', applyFilesBreakpoint);
    void markdownFiles.reconcile($chat.collection.chats.map(chat => chat.id));
    void loadRuntimeContext(contextController.signal).then(result => {
      if (contextController.signal.aborted) return;
      if (result.ok) runtimeContext = result.context;
      else configurationError = 'Configurazione del contesto non disponibile';
    });
    void loadRuntimeInfo(contextController.signal).then(result => {
      if (!contextController.signal.aborted && result.ok) runtimeInfo = result.info;
    });
    return () => {
      historyMedia.removeEventListener('change', applyHistoryBreakpoint);
      filesMedia.removeEventListener('change', applyFilesBreakpoint);
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
    ? contextUsage(occupancyMessages, runtimeContext)
    : null;
  $: persistenceWarning = $chat.persistenceWarning === 'invalid-record'
    ? 'Archivio chat non valido: avvio con una nuova chat'
    : $chat.persistenceWarning === 'unavailable'
      ? 'Persistenza non disponibile: le chat resteranno solo in memoria'
      : $markdownFiles.warning === 'invalid-record'
        ? 'Archivio file non valido: i record danneggiati sono stati rimossi'
        : $markdownFiles.warning === 'unavailable'
          ? 'Persistenza file non disponibile: i file aggiunti resteranno solo in memoria'
          : null;

  async function send(): Promise<void> {
    const submitted = draft;
    draft = '';
    if (!runtimeContext || !filesReady) return;
    await chat.send(submitted, runtimeContext, $markdownFiles.files);
    // Restore only when no newer draft was prepared during the failed request.
    if ($chat.status === 'error' && submitted.trim() && !draft) draft = submitted;
  }

  function closeHistory(): void {
    historyOpen = false;
    void tick().then(() => historyToggle?.focus());
  }

  function closeFiles(): void {
    filesOpen = false;
    void tick().then(() => filesToggle?.focus());
  }

  function toggleHistory(): void {
    const opening = !historyOpen;
    historyOpen = opening;
    if (opening && filesOverlay) filesOpen = false;
  }

  function toggleFiles(): void {
    const opening = !filesOpen;
    filesOpen = opening;
    if (opening && mobile) historyOpen = false;
  }

  function selectChat(id: string): void {
    chat.selectChat(id);
    if (mobile) closeHistory();
  }

  function regenerate(): void {
    if (runtimeContext && filesReady) void chat.regenerate(runtimeContext, $markdownFiles.files);
  }

  function editPrompt(userId: string, text: string): void {
    if (runtimeContext && filesReady) {
      void chat.editPrompt(userId, text, runtimeContext, $markdownFiles.files);
    }
  }

  function addFiles(selected: File[]): void {
    if (!runtimeContext || !filesReady || streaming) return;
    const base = wireMessages(messages, currentChat.systemPrompt, draft);
    void markdownFiles.add(selected, currentChat.id, base, runtimeContext);
  }
</script>

<section class="application">
  <ChatHistory
    {chats}
    activeId={$chat.collection.activeChatId}
    open={historyOpen}
    streaming={chatLocked}
    on:new={() => chat.newChat()}
    on:select={event => selectChat(event.detail)}
    on:rename={event => chat.renameChat(event.detail.id, event.detail.title)}
    on:delete={event => chat.deleteChat(event.detail)}
    on:close={closeHistory}
  />

  <section class="chat-layout">
    <Header
      bind:historyToggle
      bind:filesToggle
      {historyOpen}
      {filesOpen}
      fileCount={$markdownFiles.files.length}
      {runtimeInfo}
      on:history={toggleHistory}
      on:files={toggleFiles}
    />

    <SystemPrompt value={currentChat.systemPrompt} disabled={chatLocked} on:change={event => chat.setSystemPrompt(event.detail)} />
    <SessionActions importDisabled={chatLocked} on:export={() => downloadChatFile(serializeChat(messages, currentChat.systemPrompt))} on:import={event => chat.importChat(event.detail)} />
    <Transcript {messages} {streaming} on:regenerate={regenerate} on:edit={event => editPrompt(event.detail.userId, event.detail.text)} on:delete={() => { if (filesReady) chat.deleteLastTurn(); }} />
    {#if persistenceWarning}<div class="persistence-warning" role="status">{persistenceWarning}</div>{/if}
    <Status warning={null} error={configurationError ?? $chat.error ?? $markdownFiles.error} {usage} />
    <Metrics telemetry={$chat.telemetry} />
    <Composer bind:value={draft} {streaming} contextAvailable={runtimeContext !== null && filesReady} on:send={send} on:stop={() => chat.stop()} />
  </section>

  <FilesPanel
    files={$markdownFiles.files}
    open={filesOpen}
    overlay={filesOverlay}
    disabled={streaming || runtimeContext === null}
    busy={$markdownFiles.busy}
    ready={filesLoaded}
    on:add={event => addFiles(event.detail)}
    on:download={event => downloadMarkdownFile(event.detail.name, event.detail.content)}
    on:delete={event => markdownFiles.remove(event.detail)}
    on:close={closeFiles}
  />
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
    grid-template-rows: auto auto auto minmax(0, 1fr) auto auto auto auto;
    gap: var(--gn-space-md);
  }
  .persistence-warning {
    border: var(--gn-border-width) solid var(--gn-streaming); background: var(--gn-bg-panel);
    padding: var(--gn-space-sm) var(--gn-space-md); color: var(--gn-text-primary);
    font-size: var(--gn-text-sm); font-weight: 600;
  }
</style>
