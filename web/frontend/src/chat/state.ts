/*
 * Application chat lifecycle coordinator: owns the canonical collection store
 * and public actions while generation.ts, sessions.ts, persistence.ts, transfer,
 * and system-prompt modules retain their separate boundaries.
 */
import { get, writable } from 'svelte/store';
import { createGeneration } from './generation.ts';
import { loadChats, saveChats } from './persistence.ts';
import {
  activeChat,
  appendChat,
  deleteChat as deleteSession,
  newChat as createSession,
  renameChat as renameSession,
  replaceActiveTranscript,
  selectChat as selectSession
} from './sessions.ts';
import { loadSystemPrompt, saveSystemPrompt } from './systemPrompt.ts';
import { finalPair, hydrateTranscript, removeTrailingTurn } from './transcript.ts';
import { parseChatFile } from './transfer.ts';
import type { ChatCollection, ChatSnapshot, RuntimeContext } from './types.ts';

export { wireMessages } from './transcript.ts';

function initialSnapshot(): ChatSnapshot {
  const restored = loadChats();
  return {
    collection: restored.collection,
    status: 'idle',
    error: null,
    persistenceWarning: restored.warning,
    systemPrompt: loadSystemPrompt(),
    generationStartedAt: null,
    generationMs: null
  };
}

function createChatState() {
  const store = writable<ChatSnapshot>(initialSnapshot());
  const generation = createGeneration(store, checkpoint);

  async function send(text: string, context: RuntimeContext): Promise<void> {
    await generation.send(text, context);
  }

  function stop(): void {
    generation.stop();
  }

  async function regenerate(context: RuntimeContext): Promise<void> {
    await generation.regenerate(context);
  }

  async function editLastPrompt(text: string, context: RuntimeContext): Promise<void> {
    await generation.editLastPrompt(text, context);
  }

  function deleteLastTurn(): void {
    const current = get(store);
    if (current.status === 'streaming') return;
    const chat = activeChat(current.collection);
    const pair = finalPair(chat.messages);
    if (!pair) return;
    const messages = removeTrailingTurn(chat.messages, pair[0].id, pair[1].id);
    applyStable(replaceActiveTranscript(current.collection, messages));
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
    mutate(collection => deleteSession(collection, id));
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
    const collection = appendChat(current.collection, hydrateTranscript(result.payload.messages));
    store.set({
      ...current,
      collection,
      status: 'idle',
      error: null,
      systemPrompt: result.payload.systemPrompt,
      generationStartedAt: null,
      generationMs: null
    });
    saveSystemPrompt(result.payload.systemPrompt);
    persist(collection);
  }

  function setSystemPrompt(text: string): void {
    store.update(snapshot => ({ ...snapshot, systemPrompt: text }));
    saveSystemPrompt(text);
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
      generationStartedAt: null,
      generationMs: null
    });
    persist(collection);
  }

  function checkpoint(chatId: string): void {
    const current = get(store);
    if (current.collection.activeChatId !== chatId) return;
    const messages = activeChat(current.collection).messages;
    applyStable(replaceActiveTranscript(current.collection, messages));
  }

  function applyStable(collection: ChatCollection): void {
    const current = get(store);
    store.set({ ...current, collection, status: 'idle', error: null });
    persist(collection);
  }

  function persist(collection: ChatCollection): void {
    const persistenceWarning = saveChats(collection);
    store.update(snapshot => ({ ...snapshot, persistenceWarning }));
  }

  return {
    subscribe: store.subscribe,
    send, stop, regenerate, editLastPrompt, deleteLastTurn,
    newChat, selectChat, renameChat, deleteChat, importChat, setSystemPrompt
  };
}

export const chat = createChatState();
