/*
 * Browser chat-storage I/O boundary: owns exception-safe load, save, invalid
 * cleanup, and atomic migration choreography. Archive schemas belong to
 * archive.ts, collection rules to sessions.ts, and checkpoints to state.ts.
 */
import { parseArchive, serializeArchive, STORAGE_KEY } from './archive.ts';
import { createCollection } from './sessions.ts';
import type { ChatCollection, ChatLoadResult, ChatSaveResult } from './types.ts';

export function loadChats(
  updatedAt = Date.now(),
  idSource: () => string = () => globalThis.crypto.randomUUID()
): ChatLoadResult {
  let storage: Storage;
  let raw: string | null;
  try {
    storage = window.localStorage;
    raw = storage.getItem(STORAGE_KEY);
  } catch {
    return { collection: createCollection(updatedAt, idSource), warning: 'unavailable' };
  }

  if (raw === null) {
    const collection = createCollection(updatedAt, idSource);
    return { collection, warning: write(storage, collection) };
  }

  const parsed = parseArchive(raw, updatedAt, idSource);
  if (parsed.kind === 'current') {
    return { collection: parsed.collection, warning: null };
  }
  if (parsed.kind === 'legacy') {
    // One successful setItem atomically replaces the exact legacy value.
    return { collection: parsed.collection, warning: write(storage, parsed.collection) };
  }

  const collection = createCollection(updatedAt, idSource);
  try {
    storage.removeItem(STORAGE_KEY);
    return { collection, warning: 'invalid-record' };
  } catch {
    return { collection, warning: 'unavailable' };
  }
}

export function saveChats(collection: ChatCollection): ChatSaveResult {
  const serialized = serializeArchive(collection);
  if (!serialized.ok) {
    return 'unavailable';
  }
  try {
    window.localStorage.setItem(STORAGE_KEY, serialized.raw);
    return null;
  } catch {
    return 'unavailable';
  }
}

function write(storage: Storage, collection: ChatCollection): ChatSaveResult {
  const serialized = serializeArchive(collection);
  if (!serialized.ok) {
    return 'unavailable';
  }
  try {
    storage.setItem(STORAGE_KEY, serialized.raw);
    return null;
  } catch {
    return 'unavailable';
  }
}
