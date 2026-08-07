/*
 * Deterministic state-coordination acceptance tests cover public multi-chat
 * actions, import, global prompt ownership, warnings, and streaming guards.
 * Generation transport details, Svelte rendering, and real storage are excluded.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import type { ChatSnapshot, RuntimeContext } from './types.ts';

const CONVERSATION_KEY = 'graph-horizon.conversation';
const SYSTEM_KEY = 'graph-horizon.system-prompt';

class TestStorage {
  values = new Map<string, string>();
  setCalls: [string, string][] = [];
  removeCalls: string[] = [];
  throwSet = false;

  getItem(key: string): string | null { return this.values.get(key) ?? null; }
  setItem(key: string, value: string): void {
    this.setCalls.push([key, value]);
    if (this.throwSet) throw new Error('quota');
    this.values.set(key, value);
  }
  removeItem(key: string): void {
    this.removeCalls.push(key);
    this.values.delete(key);
  }
  resetCalls(): void { this.setCalls = []; this.removeCalls = []; }
}

const restored = [
  { role: 'user', content: 'restored user' },
  { role: 'assistant', content: 'restored assistant' }
];
const storage = new TestStorage();
storage.values.set(CONVERSATION_KEY, JSON.stringify({ version: 1, messages: restored }));
storage.values.set(SYSTEM_KEY, 'restored system');
Object.defineProperty(globalThis, 'window', {
  value: { localStorage: storage },
  configurable: true
});

let fetchHandler: typeof fetch = async () => { throw new Error('unexpected fetch'); };
globalThis.fetch = (...args) => fetchHandler(...args);
const { chat } = await import('./state.ts');
let snapshot: ChatSnapshot;
chat.subscribe(value => { snapshot = value; });

const context: RuntimeContext = { contextLimit: 4096, maxTokens: 128, safeTotalBudget: 3686 };
const encoder = new TextEncoder();
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
const active = () => snapshot.collection.chats.find(
  candidate => candidate.id === snapshot.collection.activeChatId
)!;
const plain = () => active().messages.map(({ role, content }) => ({ role, content }));
const conversationSets = () => storage.setCalls.filter(([key]) => key === CONVERSATION_KEY);

function controlledFetch() {
  let controller: ReadableStreamDefaultController<Uint8Array>;
  fetchHandler = async (_input, init) => {
    const body = new ReadableStream<Uint8Array>({
      start(value) {
        controller = value;
        init?.signal?.addEventListener('abort', () =>
          controller.error(new DOMException('Aborted', 'AbortError')),
        { once: true });
      }
    });
    return new Response(body, { status: 200 });
  };
  return {
    delta(content: string) {
      const data = JSON.stringify({ choices: [{ delta: { content } }] });
      controller.enqueue(encoder.encode(`data: ${data}\n\n`));
    },
    done() {
      controller.enqueue(encoder.encode('data: [DONE]\n\n'));
      controller.close();
    }
  };
}

function imported(systemPrompt: string, messages = [
  { role: 'user', content: 'imported user' },
  { role: 'assistant', content: 'imported assistant' }
]): string {
  return JSON.stringify({ version: 1, systemPrompt, messages });
}

test('startup migrates into one canonical active collection', () => {
  assert.equal(snapshot.collection.chats.length, 1);
  assert.deepEqual(plain(), restored);
  assert.equal(snapshot.systemPrompt, 'restored system');
  assert.equal(snapshot.persistenceWarning, null);
  assert.equal(JSON.parse(storage.values.get(CONVERSATION_KEY)!).version, 2);
});

test('new chat no-ops when empty and persists creation otherwise', () => {
  storage.resetCalls();
  chat.newChat();
  assert.equal(snapshot.collection.chats.length, 2);
  assert.deepEqual(active().messages, []);
  assert.equal(conversationSets().length, 1);
  const collection = snapshot.collection;
  storage.resetCalls();
  chat.newChat();
  assert.equal(snapshot.collection, collection);
  assert.equal(conversationSets().length, 0);
});

test('selection and valid rename persist without changing recency', () => {
  const target = snapshot.collection.chats[0];
  const times = snapshot.collection.chats.map(item => item.updatedAt);
  storage.resetCalls();
  chat.selectChat(target.id);
  assert.equal(snapshot.collection.activeChatId, target.id);
  assert.deepEqual(snapshot.collection.chats.map(item => item.updatedAt), times);
  assert.equal(conversationSets().length, 1);

  storage.resetCalls();
  assert.equal(chat.renameChat(target.id, '  titolo   nuovo  '), true);
  assert.equal(active().title, 'titolo   nuovo');
  assert.equal(active().updatedAt, target.updatedAt);
  assert.equal(conversationSets().length, 1);
  const collection = snapshot.collection;
  assert.equal(chat.renameChat(target.id, '   '), false);
  assert.equal(chat.renameChat(target.id, 'x'.repeat(81)), false);
  assert.equal(snapshot.collection, collection);
});

test('delete last turn and chat each persist one stable non-empty collection', () => {
  storage.resetCalls();
  chat.deleteLastTurn();
  assert.deepEqual(active().messages, []);
  assert.equal(conversationSets().length, 1);

  const deletedId = snapshot.collection.activeChatId;
  storage.resetCalls();
  chat.deleteChat(deletedId);
  assert.equal(snapshot.collection.chats.some(item => item.id === deletedId), false);
  assert.equal(snapshot.collection.chats.length >= 1, true);
  assert.equal(conversationSets().length, 1);
});

test('valid import creates a new active chat and invalid import changes no durable data', () => {
  const count = snapshot.collection.chats.length;
  storage.resetCalls();
  chat.importChat(imported('imported system'));
  assert.equal(snapshot.collection.chats.length, count + 1);
  assert.deepEqual(plain(), [
    { role: 'user', content: 'imported user' },
    { role: 'assistant', content: 'imported assistant' }
  ]);
  assert.equal(active().title, 'imported user');
  assert.equal(snapshot.systemPrompt, 'imported system');
  assert.equal(storage.values.get(SYSTEM_KEY), 'imported system');
  assert.equal(conversationSets().length, 1);

  const collection = snapshot.collection;
  const archive = storage.values.get(CONVERSATION_KEY);
  storage.resetCalls();
  chat.importChat(JSON.stringify({ version: 1, systemPrompt: 'must not apply', messages: [{}] }));
  assert.equal(snapshot.collection, collection);
  assert.equal(snapshot.systemPrompt, 'imported system');
  assert.equal(storage.values.get(CONVERSATION_KEY), archive);
  assert.equal(conversationSets().length, 0);

  chat.importChat(imported('empty system', []));
  assert.equal(snapshot.collection.chats.length, count + 2);
  assert.deepEqual(active().messages, []);
});

test('streaming guards every collection and last-turn mutation', async () => {
  chat.selectChat(snapshot.collection.chats.find(item => item.messages.length > 0)!.id);
  storage.resetCalls();
  const stream = controlledFetch();
  const pending = chat.send('streaming guard', context);
  await tick();
  const collection = snapshot.collection;
  const target = snapshot.collection.chats.find(item => item.id !== snapshot.collection.activeChatId)!.id;
  assert.equal(snapshot.status, 'streaming');
  chat.newChat();
  chat.selectChat(target);
  assert.equal(chat.renameChat(snapshot.collection.activeChatId, 'blocked'), false);
  chat.deleteChat(target);
  chat.deleteLastTurn();
  chat.importChat(imported('blocked'));
  await chat.regenerate(context);
  await chat.editLastPrompt('blocked', context);
  assert.equal(snapshot.collection, collection);
  assert.equal(conversationSets().length, 0);
  stream.delta('kept');
  chat.stop();
  await pending;
  assert.equal(snapshot.status, 'idle');
  assert.equal(conversationSets().length, 1);
});

test('generation checkpoints stable recency and persistence after idle', async () => {
  storage.resetCalls();
  const before = active().updatedAt;
  const stream = controlledFetch();
  const pending = chat.send('stable', context);
  await tick();
  stream.delta('response');
  stream.done();
  await pending;
  assert.equal(snapshot.status, 'idle');
  assert.equal(active().updatedAt >= before, true);
  assert.equal(conversationSets().length, 1);
});

test('failed save keeps memory warning and a later stable action clears it', () => {
  const target = snapshot.collection.chats.find(item => item.id !== snapshot.collection.activeChatId)!;
  storage.throwSet = true;
  chat.selectChat(target.id);
  assert.equal(snapshot.collection.activeChatId, target.id);
  assert.equal(snapshot.persistenceWarning, 'unavailable');
  storage.throwSet = false;
  chat.renameChat(target.id, 'warning cleared');
  assert.equal(snapshot.persistenceWarning, null);
});

test('system prompt remains in its separate storage boundary', () => {
  storage.resetCalls();
  chat.setSystemPrompt('global prompt');
  assert.equal(snapshot.systemPrompt, 'global prompt');
  assert.equal(storage.values.get(SYSTEM_KEY), 'global prompt');
  assert.equal(conversationSets().length, 0);
});
