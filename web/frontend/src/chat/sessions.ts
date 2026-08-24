/*
 * Pure chat-collection transformations: owns per-chat prompt and transcript
 * creation, active selection, deterministic ordering, rename, deletion, and
 * stable replacement. Persistence, stores, transport, and UI are excluded.
 */
import { validateTranscript } from './transcript.ts';
import type { ChatCollection, ChatMessage, ChatRecord } from './types.ts';

export const NEW_CHAT_TITLE = 'New chat';
export const MAX_TITLE_CODE_POINTS = 80;

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export function createCollection(
  updatedAt = Date.now(),
  idSource: () => string = () => globalThis.crypto.randomUUID(),
  systemPrompt = ''
): ChatCollection {
  const id = nextId([], idSource);
  return {
    activeChatId: id,
    chats: [{ id, title: NEW_CHAT_TITLE, systemPrompt, messages: [], updatedAt }]
  };
}

export function activeChat(collection: ChatCollection): ChatRecord {
  // A validated collection always contains its active ID.
  return collection.chats.find(chat => chat.id === collection.activeChatId)!;
}

export function orderedChats(collection: ChatCollection): ChatRecord[] {
  return [...collection.chats].sort(
    (left, right) => right.updatedAt - left.updatedAt || (left.id < right.id ? -1 : left.id > right.id ? 1 : 0)
  );
}

export function appendChat(
  collection: ChatCollection,
  messages: ChatMessage[],
  systemPrompt = '',
  updatedAt = Date.now(),
  idSource: () => string = () => globalThis.crypto.randomUUID()
): ChatCollection {
  if (validateTranscript(messages) === null) {
    return collection;
  }
  const id = nextId(collection.chats, idSource);
  const chat: ChatRecord = {
    id,
    title: derivedTitle(messages),
    systemPrompt,
    messages,
    updatedAt
  };
  return { activeChatId: id, chats: [...collection.chats, chat] };
}

export function newChat(
  collection: ChatCollection,
  updatedAt = Date.now(),
  idSource: () => string = () => globalThis.crypto.randomUUID()
): ChatCollection {
  const current = activeChat(collection);
  // A prompt-only chat owns durable user input and must not be reused as blank.
  return current.messages.length === 0 && current.systemPrompt === ''
    ? collection
    : appendChat(collection, [], '', updatedAt, idSource);
}

export function selectChat(collection: ChatCollection, id: string): ChatCollection {
  if (id === collection.activeChatId || !collection.chats.some(chat => chat.id === id)) {
    return collection;
  }
  return { ...collection, activeChatId: id };
}

export function renameChat(
  collection: ChatCollection,
  id: string,
  requestedTitle: string
): ChatCollection {
  const title = requestedTitle.trim();
  if (!title || Array.from(title).length > MAX_TITLE_CODE_POINTS) {
    return collection;
  }
  const index = collection.chats.findIndex(chat => chat.id === id);
  if (index < 0 || collection.chats[index].title === title) {
    return collection;
  }
  const chats = [...collection.chats];
  chats[index] = { ...chats[index], title };
  return { ...collection, chats };
}

export function deleteChat(
  collection: ChatCollection,
  id: string,
  updatedAt = Date.now(),
  idSource: () => string = () => globalThis.crypto.randomUUID()
): ChatCollection {
  if (!collection.chats.some(chat => chat.id === id)) {
    return collection;
  }
  if (collection.chats.length === 1) {
    return createCollection(updatedAt, idSource);
  }
  const chats = collection.chats.filter(chat => chat.id !== id);
  const activeChatId = id === collection.activeChatId
    ? orderedChats({ activeChatId: chats[0].id, chats })[0].id
    : collection.activeChatId;
  return { activeChatId, chats };
}

export function replaceActiveTranscript(
  collection: ChatCollection,
  messages: ChatMessage[],
  updatedAt = Date.now()
): ChatCollection {
  if (validateTranscript(messages) === null) {
    return collection;
  }
  const index = collection.chats.findIndex(chat => chat.id === collection.activeChatId);
  const current = collection.chats[index];
  const title = current.title === NEW_CHAT_TITLE ? derivedTitle(messages) : current.title;
  const chats = [...collection.chats];
  chats[index] = { ...current, title, messages, updatedAt };
  return { ...collection, chats };
}

export function replaceChatMessages(
  collection: ChatCollection,
  chatId: string,
  messages: ChatMessage[]
): ChatCollection {
  return {
    ...collection,
    chats: collection.chats.map(chat => chat.id === chatId ? { ...chat, messages } : chat)
  };
}

export function replaceActiveSystemPrompt(
  collection: ChatCollection,
  systemPrompt: string
): ChatCollection {
  const index = collection.chats.findIndex(chat => chat.id === collection.activeChatId);
  const current = collection.chats[index];
  if (current.systemPrompt === systemPrompt) {
    return collection;
  }
  const chats = [...collection.chats];
  // Prompt edits are metadata edits and preserve transcript recency.
  chats[index] = { ...current, systemPrompt };
  return { ...collection, chats };
}

function nextId(chats: ChatRecord[], idSource: () => string): string {
  const used = new Set(chats.map(chat => chat.id));
  let id = idSource();
  while (!UUID.test(id) || used.has(id)) {
    id = idSource();
  }
  return id;
}

function derivedTitle(messages: ChatMessage[]): string {
  const normalized = messages[0]?.content.trim().replace(/\s+/gu, ' ') ?? '';
  return normalized ? Array.from(normalized).slice(0, 48).join('') : NEW_CHAT_TITLE;
}
