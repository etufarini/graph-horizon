/*
 * Application chat lifecycle coordinator: owns the canonical per-chat collection
 * store and public actions while generation, sessions, persistence, and transfer
 * modules retain their separate boundaries.
 */
import { get, writable } from 'svelte/store';
import { createGeneration } from './generation.ts';
import { markdownFiles } from './files/state.ts';
import { loadChats, saveChats } from './persistence.ts';
import {
  activeChat,
  appendChat,
  deleteChat as deleteSession,
  newChat as createSession,
  renameChat as renameSession,
  replaceActiveSystemPrompt,
  replaceActiveTranscript,
  selectChat as selectSession
} from './sessions.ts';
import { finalPair, hydrateTranscript, removeTrailingTurn } from './transcript.ts';
import { parseChatFile } from './transfer.ts';
import type { ChatCollection, ChatSnapshot, RuntimeContext } from './types.ts';
import type { MarkdownFileRecord } from './files/record.ts';

export { wireMessages } from './transcript.ts';

function initialSnapshot(): ChatSnapshot {
  const restored = loadChats();
  return {
    collection: restored.collection,
    status: 'idle',
    error: null,
    persistenceWarning: restored.warning,
    telemetry: null
  };
}

function createChatState() {
  const store = writable<ChatSnapshot>(initialSnapshot());
  const generation = createGeneration(store, checkpoint);

  async function send(
    text: string,
    context: RuntimeContext,
    files: MarkdownFileRecord[] = []
  ): Promise<void> {
    await generation.send(text, context, files);
  }

  function stop(): void {
    generation.stop();
  }

  async function regenerate(
    context: RuntimeContext,
    files: MarkdownFileRecord[] = []
  ): Promise<void> {
    await generation.regenerate(context, files);
  }

  async function editPrompt(
    userId: string,
    text: string,
    context: RuntimeContext,
    files: MarkdownFileRecord[] = []
  ): Promise<void> {
    await generation.editPrompt(userId, text, context, files);
  }

  function deleteLastTurn(): void {
    const current = get(store);
    if (current.status === 'streaming') return;
    const chat = activeChat(current.collection);
    const pair = finalPair(chat.messages);
    if (!pair) return;
    const messages = removeTrailingTurn(chat.messages, pair[0].id, pair[1].id);
    applyStable(replaceActiveTranscript(current.collection, messages), false);
  }

  function newChat(): void {
    mutate(collection => createSession(collection));
  }

  function selectChat(id: string): void {
    mutate(collection => selectSession(collection, id));
  }

  function renameChat(id: string, requestedTitle: string): boolean {
    const current = get(store);
    const title = requestedTitle.trim();
    if (current.status === 'streaming' || !title || Array.from(title).length > 80 ||
        !current.collection.chats.some(chat => chat.id === id)) {
      return false;
    }
    const collection = renameSession(current.collection, id, title);
    store.set({ ...current, collection });
    persist(collection);
    return true;
  }

  function deleteChat(id: string): void {
    const current = get(store);
    if (current.status === 'streaming' || !current.collection.chats.some(chat => chat.id === id)) {
      return;
    }
    mutate(collection => deleteSession(collection, id));
    void markdownFiles.removeChat(id);
  }

  function importChat(text: string): void {
    const current = get(store);
    if (current.status === 'streaming') return;
    const result = parseChatFile(text);
    if (!result.ok) {
      const error = result.error === 'invalid-json'
        ? 'File non valido: JSON non riconosciuto'
        : 'File non valido: formato chat non riconosciuto';
      store.set({ ...current, status: 'error', error });
      return;
    }
    const collection = appendChat(
      current.collection,
      hydrateTranscript(result.payload.messages),
      result.payload.systemPrompt
    );
    store.set({
      ...current,
      collection,
      status: 'idle',
      error: null,
      telemetry: null
    });
    persist(collection);
  }

  function setSystemPrompt(text: string): void {
    const current = get(store);
    // Persisting during streaming would checkpoint a partial assistant response.
    if (current.status === 'streaming') return;
    const collection = replaceActiveSystemPrompt(current.collection, text);
    if (collection === current.collection) return;
    store.set({ ...current, collection });
    persist(collection);
  }

  function mutate(change: (collection: ChatCollection) => ChatCollection): void {
    const current = get(store);
    if (current.status === 'streaming') return;
    const collection = change(current.collection);
    if (collection === current.collection) return;
    store.set({
      ...current,
      collection,
      status: 'idle',
      error: null,
      telemetry: null
    });
    persist(collection);
  }

  function checkpoint(chatId: string): void {
    const current = get(store);
    if (current.collection.activeChatId !== chatId) return;
    const messages = activeChat(current.collection).messages;
    applyStable(replaceActiveTranscript(current.collection, messages), true);
  }

  function applyStable(collection: ChatCollection, keepTelemetry: boolean): void {
    const current = get(store);
    store.set({
      ...current,
      collection,
      status: 'idle',
      error: null,
      telemetry: keepTelemetry ? current.telemetry : null
    });
    persist(collection);
  }

  function persist(collection: ChatCollection): void {
    const persistenceWarning = saveChats(collection);
    store.update(snapshot => ({ ...snapshot, persistenceWarning }));
  }

  return {
    subscribe: store.subscribe,
    send, stop, regenerate, editPrompt, deleteLastTurn,
    newChat, selectChat, renameChat, deleteChat, importChat, setSystemPrompt
  };
}

export const chat = createChatState();
