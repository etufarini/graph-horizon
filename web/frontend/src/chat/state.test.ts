/*
 * Deterministic state-coordination acceptance tests cover per-chat prompt
 * ownership, import, warnings, streaming guards, and the prompt-only runtime
 * fixture. Generation transport, Svelte rendering, and real storage are excluded.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { serializeChat } from './transfer.ts';
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

const context: RuntimeContext = {
  contextLimit: 4096,
  safePromptBudget: 3686,
  search: { enabled: true, provider: 'search.example', maxQueryCharacters: 512, maxContextCharacters: 2800, dateFilters: true }
};
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
        controller.enqueue(encoder.encode(
          'data: {"phase":"prefill"}\n\n' +
          'data: {"phase":"decode"}\n\n'
        ));
        init?.signal?.addEventListener('abort', () =>
          controller.error(new DOMException('Aborted', 'AbortError')),
        { once: true });
      }
    });
    return new Response(body, { status: 200 });
  };
  return {
    delta(content: string) {
      const data = JSON.stringify({ content });
      controller.enqueue(encoder.encode(`data: ${data}\n\n`));
    },
    done() {
      controller.enqueue(encoder.encode(
        'data: {"stats":{"prompt_tokens":12,"prefill_tokens":8,"completion_tokens":3,"prefill_ms":40,"decode_ms":60}}\n\n' +
        'data: {"done":true}\n\n'
      ));
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
  assert.equal(active().systemPrompt, 'restored system');
  assert.equal(snapshot.persistenceWarning, null);
  assert.equal(JSON.parse(storage.values.get(CONVERSATION_KEY)!).version, 4);
  assert.equal(storage.values.has(SYSTEM_KEY), false);
});

test('new chat no-ops when empty and persists creation otherwise', () => {
  storage.resetCalls();
  chat.newChat();
  assert.equal(snapshot.collection.chats.length, 2);
  assert.deepEqual(active().messages, []);
  assert.equal(active().systemPrompt, '');
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
  assert.equal(chat.renameChat(target.id, '  new   title  '), true);
  assert.equal(active().title, 'new   title');
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
  assert.equal(active().systemPrompt, 'imported system');
  assert.equal(storage.values.has(SYSTEM_KEY), false);
  assert.equal(conversationSets().length, 1);

  const collection = snapshot.collection;
  const archive = storage.values.get(CONVERSATION_KEY);
  storage.resetCalls();
  chat.importChat(JSON.stringify({ version: 1, systemPrompt: 'must not apply', messages: [{}] }));
  assert.equal(snapshot.collection, collection);
  assert.equal(active().systemPrompt, 'imported system');
  assert.equal(storage.values.get(CONVERSATION_KEY), archive);
  assert.equal(conversationSets().length, 0);

  chat.importChat(imported('empty system', []));
  assert.equal(snapshot.collection.chats.length, count + 2);
  assert.deepEqual(active().messages, []);
});

test('invalid JSON preserves prompt and archive while export stays version 2', () => {
  const collection = snapshot.collection;
  const systemPrompt = active().systemPrompt;
  const archive = storage.values.get(CONVERSATION_KEY);
  storage.resetCalls();
  chat.importChat('{');
  assert.equal(snapshot.collection, collection);
  assert.equal(active().systemPrompt, systemPrompt);
  assert.equal(storage.values.get(CONVERSATION_KEY), archive);
  assert.equal(storage.values.has(SYSTEM_KEY), false);
  assert.equal(conversationSets().length, 0);

  const exported = JSON.parse(serializeChat(active().messages, active().systemPrompt));
  assert.deepEqual(exported, {
    version: 2,
    systemPrompt,
    messages: plain()
  });
  assert.equal('activeChatId' in exported, false);
  assert.equal('title' in exported, false);
});

test('streaming guards every collection and turn mutation', async () => {
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
  chat.setSystemPrompt('blocked');
  await chat.regenerate(context);
  await chat.editPrompt(active().messages[0].id, 'blocked', context);
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

test('system prompts remain isolated in the canonical collection', () => {
  const first = snapshot.collection.chats[0];
  const second = snapshot.collection.chats.find(item => item.id !== first.id)!;
  storage.resetCalls();
  chat.selectChat(first.id);
  chat.setSystemPrompt('first prompt');
  assert.equal(active().systemPrompt, 'first prompt');
  chat.selectChat(second.id);
  chat.setSystemPrompt('second prompt');
  assert.equal(active().systemPrompt, 'second prompt');
  chat.selectChat(first.id);
  assert.equal(active().systemPrompt, 'first prompt');
  assert.equal(snapshot.collection.chats.find(item => item.id === second.id)!.systemPrompt,
    'second prompt');
  assert.equal(storage.values.has(SYSTEM_KEY), false);
  assert.equal(conversationSets().length, 4);
});
