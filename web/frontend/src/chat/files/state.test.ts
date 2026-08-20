/*
 * Markdown-file lifecycle tests: use an in-memory storage boundary to verify
 * durable additions, same-name replacement, capacity rejection, and fallback.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { get } from 'svelte/store';
import { createMarkdownFileState } from './state.ts';
import type { MarkdownFileStorage } from './persistence.ts';
import type { MarkdownFileRecord } from './record.ts';

const CHAT_ID = '00000000-0000-4000-8000-000000000001';
const OTHER_ID = '00000000-0000-4000-8000-000000000002';
const CONTEXT = { contextLimit: 32768, safePromptBudget: 29491 };

class MemoryStorage implements MarkdownFileStorage {
  records: MarkdownFileRecord[] = [];
  failWrites = false;

  async list(chatId: string) {
    return { files: this.records.filter(file => file.chatId === chatId), invalid: false };
  }
  async write(files: MarkdownFileRecord[], deletedIds: string[] = []) {
    if (this.failWrites) throw new Error('unavailable');
    this.records = this.records.filter(file => !deletedIds.includes(file.id));
    this.records.push(...files);
  }
  async delete(id: string) {
    if (this.failWrites) throw new Error('unavailable');
    this.records = this.records.filter(file => file.id !== id);
  }
  async deleteChat(chatId: string) {
    this.records = this.records.filter(file => file.chatId !== chatId);
  }
  async prune(validChatIds: string[]) {
    this.records = this.records.filter(file => validChatIds.includes(file.chatId));
  }
  async persist() { return true; }
}

function selected(name: string, content: string): File {
  const bytes = new TextEncoder().encode(content);
  return {
    name,
    size: bytes.byteLength,
    arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
  } as File;
}

test('adds durably and replaces the same exact filename', async () => {
  const storage = new MemoryStorage();
  const state = createMarkdownFileState(storage);
  await state.select(CHAT_ID);
  await state.add([selected('note.md', 'prima')], CHAT_ID, [], CONTEXT);
  assert.equal(get(state).files[0].content, 'prima');
  await state.add([selected('note.md', 'seconda')], CHAT_ID, [], CONTEXT);
  assert.equal(get(state).files.length, 1);
  assert.equal(get(state).files[0].content, 'seconda');
  assert.equal(storage.records.length, 1);
});

test('rejects a candidate whose full text exceeds the active prompt budget', async () => {
  const storage = new MemoryStorage();
  const state = createMarkdownFileState(storage);
  await state.select(CHAT_ID);
  const tiny = { contextLimit: 4, safePromptBudget: 3 };
  await state.add([selected('note.md', 'contenuto')], CHAT_ID, [], tiny);
  assert.equal(get(state).files.length, 0);
  assert.match(get(state).error ?? '', /Contesto insufficiente/);
});

test('rejects duplicate names inside one multi-file selection', async () => {
  const storage = new MemoryStorage();
  const state = createMarkdownFileState(storage);
  await state.select(CHAT_ID);
  await state.add(
    [selected('same.md', 'uno'), selected('same.md', 'due')],
    CHAT_ID,
    [],
    CONTEXT
  );
  assert.equal(get(state).files.length, 0);
  assert.match(get(state).error ?? '', /stesso nome/);
});

test('failed durable writes retain usable in-memory files with a warning', async () => {
  const storage = new MemoryStorage();
  storage.failWrites = true;
  const state = createMarkdownFileState(storage);
  await state.select(CHAT_ID);
  await state.add([selected('note.md', 'locale')], CHAT_ID, [], CONTEXT);
  assert.equal(get(state).files[0].content, 'locale');
  assert.equal(get(state).warning, 'unavailable');
});

test('reconcile removes records owned by deleted chats', async () => {
  const storage = new MemoryStorage();
  storage.records = [{
    id: '00000000-0000-4000-8000-000000000003',
    chatId: OTHER_ID,
    name: 'orphan.md',
    content: 'x',
    utf8Bytes: 1,
    addedAt: 0
  }];
  const state = createMarkdownFileState(storage);
  await state.reconcile([CHAT_ID]);
  assert.deepEqual(storage.records, []);
});
