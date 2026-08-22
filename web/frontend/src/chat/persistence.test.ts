/*
 * Deterministic in-memory storage coverage for version-3 collection startup,
 * stable saves, invalid cleanup, and global-prompt migration. Real browser,
 * filesystem, and network I/O are excluded.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { serializeArchive, STORAGE_KEY } from './archive.ts';
import { LEGACY_SYSTEM_PROMPT_KEY, loadChats, saveChats } from './persistence.ts';
import { createCollection, replaceActiveTranscript } from './sessions.ts';
import { hydrateTranscript } from './transcript.ts';

class MemoryStorage {
  values = new Map<string, string>();
  getCalls: string[] = [];
  setCalls: [string, string][] = [];
  removeCalls: string[] = [];
  throwGet = false;
  throwSet = false;
  throwRemove = false;

  getItem(key: string): string | null {
    this.getCalls.push(key);
    if (this.throwGet) throw new Error('denied get');
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.setCalls.push([key, value]);
    if (this.throwSet) throw new Error('quota');
    this.values.set(key, value);
  }

  removeItem(key: string): void {
    this.removeCalls.push(key);
    if (this.throwRemove) throw new Error('denied remove');
    this.values.delete(key);
  }
}

const firstId = '00000000-0000-4000-8000-000000000001';
const secondId = '00000000-0000-4000-8000-000000000002';
const source = (id: string) => () => id;
const plain = [
  { role: 'user' as const, content: 'hello 🧠' },
  { role: 'assistant' as const, content: '' }
];

function useStorage(storage: MemoryStorage): void {
  Object.defineProperty(globalThis, 'window', {
    value: { localStorage: storage },
    configurable: true
  });
}

function collection() {
  return replaceActiveTranscript(
    createCollection(10, source(firstId), 'specific'),
    hydrateTranscript(plain),
    11
  );
}

function rawCollection(): string {
  const result = serializeArchive(collection());
  assert.equal(result.ok, true);
  return result.ok ? result.raw : '';
}

test('missing storage creates and immediately persists one empty collection', () => {
  const storage = new MemoryStorage();
  useStorage(storage);
  const result = loadChats(42, source(firstId));
  assert.equal(result.warning, null);
  assert.equal(result.collection.activeChatId, firstId);
  assert.equal(result.collection.chats[0].systemPrompt, '');
  assert.deepEqual(result.collection.chats[0].messages, []);
  assert.equal(storage.setCalls.length, 1);
  assert.equal(JSON.parse(storage.setCalls[0][1]).version, 3);
});

test('a valid archive loads without storage mutation and hydrates IDs', () => {
  const storage = new MemoryStorage();
  storage.values.set(STORAGE_KEY, rawCollection());
  useStorage(storage);
  const result = loadChats();
  assert.equal(result.warning, null);
  assert.deepEqual(
    result.collection.chats[0].messages.map(({ role, content }) => ({ role, content })),
    plain
  );
  assert.equal(result.collection.chats[0].messages[0].id.length > 0, true);
  assert.deepEqual(storage.setCalls, []);
  assert.deepEqual(storage.removeCalls, []);
});

test('invalid archive cleanup reports the exact warning and touches only its key', () => {
  const storage = new MemoryStorage();
  storage.values.set(STORAGE_KEY, '{');
  storage.values.set(LEGACY_SYSTEM_PROMPT_KEY, 'keep');
  useStorage(storage);
  const result = loadChats(1, source(firstId));
  assert.equal(result.warning, 'invalid-record');
  assert.equal(result.collection.chats[0].systemPrompt, 'keep');
  assert.deepEqual(storage.removeCalls, [STORAGE_KEY]);
  assert.equal(storage.values.get(LEGACY_SYSTEM_PROMPT_KEY), 'keep');
  assert.deepEqual(storage.setCalls, []);
});

test('failed invalid cleanup keeps a usable collection and reports unavailable', () => {
  const storage = new MemoryStorage();
  storage.values.set(STORAGE_KEY, '{');
  storage.throwRemove = true;
  useStorage(storage);
  const result = loadChats(1, source(firstId));
  assert.equal(result.warning, 'unavailable');
  assert.equal(result.collection.chats.length, 1);
  assert.equal(storage.values.get(STORAGE_KEY), '{');
});

test('legacy migration replaces the exact value only after a successful write', () => {
  const storage = new MemoryStorage();
  const legacy = JSON.stringify({ version: 1, messages: plain });
  storage.values.set(STORAGE_KEY, legacy);
  storage.values.set(LEGACY_SYSTEM_PROMPT_KEY, 'legacy prompt');
  useStorage(storage);
  const result = loadChats(77, source(secondId));
  assert.equal(result.warning, null);
  assert.equal(result.collection.activeChatId, secondId);
  assert.equal(result.collection.chats[0].systemPrompt, 'legacy prompt');
  assert.equal(storage.setCalls.length, 1);
  assert.equal(JSON.parse(storage.values.get(STORAGE_KEY)!).version, 3);
  assert.equal(storage.values.has(LEGACY_SYSTEM_PROMPT_KEY), false);

  const failed = new MemoryStorage();
  failed.values.set(STORAGE_KEY, legacy);
  failed.values.set(LEGACY_SYSTEM_PROMPT_KEY, 'legacy prompt');
  failed.throwSet = true;
  useStorage(failed);
  const fallback = loadChats(77, source(secondId));
  assert.equal(fallback.warning, 'unavailable');
  assert.deepEqual(
    fallback.collection.chats[0].messages.map(({ role, content }) => ({ role, content })),
    plain
  );
  assert.equal(failed.values.get(STORAGE_KEY), legacy);
  assert.equal(failed.values.get(LEGACY_SYSTEM_PROMPT_KEY), 'legacy prompt');
});

test('missing archive adopts the legacy prompt before removing its old key', () => {
  const storage = new MemoryStorage();
  storage.values.set(LEGACY_SYSTEM_PROMPT_KEY, 'prompt only');
  useStorage(storage);
  const result = loadChats(42, source(firstId));
  assert.equal(result.warning, null);
  assert.equal(result.collection.chats[0].systemPrompt, 'prompt only');
  assert.equal(JSON.parse(storage.values.get(STORAGE_KEY)!).chats[0].systemPrompt, 'prompt only');
  assert.equal(storage.values.has(LEGACY_SYSTEM_PROMPT_KEY), false);
});

test('stable save uses one setItem and a failed save preserves the prior value', () => {
  const storage = new MemoryStorage();
  storage.values.set(STORAGE_KEY, 'old');
  storage.values.set(LEGACY_SYSTEM_PROMPT_KEY, 'obsolete');
  useStorage(storage);
  assert.equal(saveChats(collection()), null);
  assert.equal(storage.setCalls.length, 1);
  const stable = storage.values.get(STORAGE_KEY);
  assert.equal(storage.values.has(LEGACY_SYSTEM_PROMPT_KEY), false);

  storage.throwSet = true;
  assert.equal(saveChats(createCollection(20, source(secondId))), 'unavailable');
  assert.equal(storage.values.get(STORAGE_KEY), stable);
});

test('storage acquisition, read, initial write, and cleanup are exception-safe', () => {
  const denied: Record<string, unknown> = {};
  Object.defineProperty(denied, 'localStorage', {
    get() { throw new Error('denied storage'); }
  });
  Object.defineProperty(globalThis, 'window', { value: denied, configurable: true });
  assert.equal(loadChats(1, source(firstId)).warning, 'unavailable');
  assert.equal(saveChats(collection()), 'unavailable');

  const read = new MemoryStorage();
  read.throwGet = true;
  useStorage(read);
  assert.equal(loadChats(1, source(firstId)).warning, 'unavailable');

  const initial = new MemoryStorage();
  initial.throwSet = true;
  useStorage(initial);
  const result = loadChats(1, source(firstId));
  assert.equal(result.warning, 'unavailable');
  assert.equal(result.collection.chats.length, 1);
});
