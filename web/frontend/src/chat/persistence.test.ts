/*
 * Deterministic acceptance tests for the exact versioned storage adapter.
 * They replace browser storage in memory and perform no filesystem or network I/O.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  clearConversation,
  FORMAT_VERSION,
  loadConversation,
  MAX_RECORD_BYTES,
  saveConversation,
  STORAGE_KEY
} from './persistence.ts';

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

function useStorage(storage: MemoryStorage): void {
  Object.defineProperty(globalThis, 'window', {
    value: { localStorage: storage },
    configurable: true
  });
}

function record(content = ''): string {
  return JSON.stringify({
    version: FORMAT_VERSION,
    messages: [
      { role: 'user', content },
      { role: 'assistant', content: '' }
    ]
  });
}

function exactLimitRecord(): string {
  const overhead = new TextEncoder().encode(record()).byteLength;
  return record('x'.repeat(MAX_RECORD_BYTES - overhead));
}

test('missing and valid records load without warning or runtime IDs', () => {
  const storage = new MemoryStorage();
  useStorage(storage);
  assert.deepEqual(loadConversation(), { messages: [], warning: null });
  storage.values.set(STORAGE_KEY, record('ciao 🧠'));
  assert.deepEqual(loadConversation(), {
    messages: [
      { role: 'user', content: 'ciao 🧠' },
      { role: 'assistant', content: '' }
    ],
    warning: null
  });
  assert.equal('id' in loadConversation().messages[0], false);
});

test('corrupt, unknown, inexact, and invalid transcripts are removed', () => {
  const invalid = [
    '{',
    JSON.stringify({ version: 2, messages: [] }),
    JSON.stringify({ version: 1, messages: [], extra: true }),
    JSON.stringify({
      version: 1,
      messages: [
        { role: 'user', content: 'x', extra: true },
        { role: 'assistant', content: '' }
      ]
    }),
    JSON.stringify({ version: 1, messages: [{ role: 'user', content: 'odd' }] }),
    `${exactLimitRecord()} `
  ];
  for (const raw of invalid) {
    const storage = new MemoryStorage();
    storage.values.set(STORAGE_KEY, raw);
    useStorage(storage);
    assert.deepEqual(loadConversation(), { messages: [], warning: 'invalid-record' });
    assert.deepEqual(storage.removeCalls, [STORAGE_KEY]);
    assert.equal(storage.values.has(STORAGE_KEY), false);
  }
});

test('failed corrupt-record cleanup reports unavailable', () => {
  const storage = new MemoryStorage();
  storage.values.set(STORAGE_KEY, '{');
  storage.throwRemove = true;
  useStorage(storage);
  assert.deepEqual(loadConversation(), { messages: [], warning: 'unavailable' });
  assert.equal(storage.values.get(STORAGE_KEY), '{');
});

test('exact 4 MiB save replaces once while one byte over preserves the old record', () => {
  const storage = new MemoryStorage();
  storage.values.set(STORAGE_KEY, 'old');
  useStorage(storage);
  const exact = exactLimitRecord();
  const exactMessages = JSON.parse(exact).messages;
  assert.equal(new TextEncoder().encode(exact).byteLength, MAX_RECORD_BYTES);
  assert.equal(saveConversation(exactMessages), null);
  assert.deepEqual(storage.setCalls, [[STORAGE_KEY, exact]]);

  storage.setCalls = [];
  assert.equal(saveConversation([{ ...exactMessages[0], content: `${exactMessages[0].content}x` }, exactMessages[1]]), 'unavailable');
  assert.deepEqual(storage.setCalls, []);
  assert.equal(storage.values.get(STORAGE_KEY), exact);
});

test('quota failure preserves the previous value and remains bounded', () => {
  const storage = new MemoryStorage();
  storage.values.set(STORAGE_KEY, 'old');
  storage.throwSet = true;
  useStorage(storage);
  assert.equal(saveConversation(JSON.parse(record('new')).messages), 'unavailable');
  assert.equal(storage.values.get(STORAGE_KEY), 'old');
  assert.equal(storage.setCalls.length, 1);
});

test('storage getter and each storage operation are exception-safe', () => {
  const denied: Record<string, unknown> = {};
  Object.defineProperty(denied, 'localStorage', {
    get() { throw new Error('denied storage'); }
  });
  Object.defineProperty(globalThis, 'window', {
    value: denied,
    configurable: true
  });
  assert.deepEqual(loadConversation(), { messages: [], warning: 'unavailable' });
  assert.equal(saveConversation([]), 'unavailable');
  assert.equal(clearConversation(), 'unavailable');

  const get = new MemoryStorage();
  get.throwGet = true;
  useStorage(get);
  assert.deepEqual(loadConversation(), { messages: [], warning: 'unavailable' });

  const remove = new MemoryStorage();
  remove.throwRemove = true;
  useStorage(remove);
  assert.equal(clearConversation(), 'unavailable');
});

test('clear removes only the conversation key', () => {
  const storage = new MemoryStorage();
  storage.values.set(STORAGE_KEY, record());
  storage.values.set('graph-horizon.system-prompt', 'keep');
  useStorage(storage);
  assert.equal(clearConversation(), null);
  assert.deepEqual(storage.removeCalls, [STORAGE_KEY]);
  assert.equal(storage.values.get('graph-horizon.system-prompt'), 'keep');
});
