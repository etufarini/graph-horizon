/*
 * Browser chat-storage I/O boundary: owns exception-safe load, save, invalid
 * cleanup, and migration from the legacy global prompt. Archive schemas belong
 * to archive.ts, collection rules to sessions.ts, and checkpoints to state.ts.
 */
import { parseArchive, serializeArchive, STORAGE_KEY } from './archive.ts';
import { createCollection } from './sessions.ts';
import type { ChatCollection, ChatLoadResult, ChatSaveResult } from './types.ts';

export const LEGACY_SYSTEM_PROMPT_KEY = 'graph-horizon.system-prompt';

export function loadChats(
  updatedAt = Date.now(),
  idSource: () => string = () => globalThis.crypto.randomUUID()
): ChatLoadResult {
  let storage: Storage;
  let raw: string | null;
  let legacySystemPrompt: string | null;
  try {
    storage = window.localStorage;
    raw = storage.getItem(STORAGE_KEY);
    legacySystemPrompt = storage.getItem(LEGACY_SYSTEM_PROMPT_KEY);
  } catch {
    return { collection: createCollection(updatedAt, idSource), warning: 'unavailable' };
  }

  if (raw === null) {
    const collection = createCollection(updatedAt, idSource, legacySystemPrompt ?? '');
    return { collection, warning: write(storage, collection, legacySystemPrompt !== null) };
  }

  const parsed = parseArchive(raw, updatedAt, idSource, legacySystemPrompt ?? '');
  if (parsed.kind === 'current') {
    if (legacySystemPrompt !== null) removeLegacyPrompt(storage);
    return { collection: parsed.collection, warning: null };
  }
  if (parsed.kind === 'legacy') {
    // One successful setItem atomically replaces the exact legacy value.
    return {
      collection: parsed.collection,
      warning: write(storage, parsed.collection, legacySystemPrompt !== null)
    };
  }

  const collection = createCollection(updatedAt, idSource, legacySystemPrompt ?? '');
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
    const storage = window.localStorage;
    storage.setItem(STORAGE_KEY, serialized.raw);
    removeLegacyPrompt(storage);
    return null;
  } catch {
    return 'unavailable';
  }
}

function write(
  storage: Storage,
  collection: ChatCollection,
  cleanupLegacy = false
): ChatSaveResult {
  const serialized = serializeArchive(collection);
  if (!serialized.ok) {
    return 'unavailable';
  }
  try {
    storage.setItem(STORAGE_KEY, serialized.raw);
    if (cleanupLegacy) removeLegacyPrompt(storage);
    return null;
  } catch {
    return 'unavailable';
  }
}

function removeLegacyPrompt(storage: Storage): void {
  try {
    // Once version 3 is durable, failure to remove the ignored old key is benign.
    storage.removeItem(LEGACY_SYSTEM_PROMPT_KEY);
  } catch {
    // A version-3 archive remains authoritative.
  }
}
