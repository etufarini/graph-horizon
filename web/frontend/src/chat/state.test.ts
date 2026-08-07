/*
 * Deterministic chat-lifecycle acceptance suite: exercises startup hydration,
 * transport, rollback, stop, import, clear, and stable storage checkpoints.
 * Svelte rendering, browser automation, and real network/storage are excluded.
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
  throwRemove = false;
  onConversationSet: (() => void) | null = null;

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.setCalls.push([key, value]);
    if (this.throwSet) throw new Error('quota');
    this.values.set(key, value);
    if (key === CONVERSATION_KEY) this.onConversationSet?.();
  }

  removeItem(key: string): void {
    this.removeCalls.push(key);
    if (this.throwRemove) throw new Error('denied remove');
    this.values.delete(key);
  }

  resetCalls(): void {
    this.setCalls = [];
    this.removeCalls = [];
  }
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

// Browser globals are configured before this single state-module initialization.
const { chat } = await import('./state.ts');
let snapshot: ChatSnapshot;
const statuses: ChatSnapshot['status'][] = [];
chat.subscribe(value => {
  snapshot = value;
  statuses.push(value.status);
});

const context: RuntimeContext = {
  contextLimit: 4096,
  maxTokens: 128,
  safeTotalBudget: 3686
};
const encoder = new TextEncoder();
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
const conversationSets = () => storage.setCalls.filter(([key]) => key === CONVERSATION_KEY);
const plainMessages = () => snapshot.messages.map(({ role, content }) => ({ role, content }));

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
    },
    close() {
      controller.close();
    }
  };
}

function imported(systemPrompt: string, user = 'imported user'): string {
  return JSON.stringify({
    version: 1,
    systemPrompt,
    messages: [
      { role: 'user', content: user },
      { role: 'assistant', content: 'imported assistant' }
    ]
  });
}

test('startup synchronously restores plain messages with fresh IDs', () => {
  assert.deepEqual(plainMessages(), restored);
  assert.equal(snapshot.systemPrompt, 'restored system');
  assert.equal(snapshot.status, 'idle');
  assert.equal(snapshot.persistenceWarning, null);
  assert.equal(snapshot.messages.every(message => typeof message.id === 'string'), true);
  assert.equal(storage.setCalls.length, 0);
});

test('successful completion writes once after idle and never on deltas', async () => {
  storage.resetCalls();
  const setStatuses: string[] = [];
  storage.onConversationSet = () => setStatuses.push(snapshot.status);
  const stream = controlledFetch();
  const pending = chat.send('next turn', context);
  await tick();
  assert.equal(snapshot.status, 'streaming');
  assert.equal(conversationSets().length, 0);
  stream.delta('partial');
  await tick();
  assert.equal(snapshot.messages.at(-1)?.content, 'partial');
  assert.equal(conversationSets().length, 0);
  stream.done();
  await pending;
  assert.equal(snapshot.status, 'idle');
  assert.equal(snapshot.messages.at(-1)?.content, 'partial');
  assert.equal(conversationSets().length, 1);
  assert.deepEqual(setStatuses, ['idle']);
  assert.equal(snapshot.generationStartedAt, null);
  assert.equal(typeof snapshot.generationMs, 'number');
});

test('empty and partial stops persist only after abort settles', async () => {
  for (const partial of ['', 'stopped partial']) {
    storage.resetCalls();
    const stream = controlledFetch();
    const pending = chat.send(`stop ${partial || 'empty'}`, context);
    await tick();
    if (partial) {
      stream.delta(partial);
      await tick();
    }
    assert.equal(conversationSets().length, 0);
    chat.stop();
    await pending;
    assert.equal(snapshot.status, 'idle');
    assert.equal(snapshot.messages.at(-1)?.content, partial);
    assert.equal(conversationSets().length, 1);
  }
});

test('failed stream rolls back its pair and leaves the stable record unchanged', async () => {
  storage.resetCalls();
  const beforeMessages = plainMessages();
  const beforeRecord = storage.values.get(CONVERSATION_KEY);
  const stream = controlledFetch();
  const pending = chat.send('will fail', context);
  await tick();
  stream.delta('uncommitted');
  stream.close();
  await pending;
  assert.equal(snapshot.status, 'error');
  assert.deepEqual(plainMessages(), beforeMessages);
  assert.equal(storage.values.get(CONVERSATION_KEY), beforeRecord);
  assert.equal(conversationSets().length, 0);
});

test('capacity rejection performs neither fetch nor persistence', async () => {
  storage.resetCalls();
  let fetches = 0;
  fetchHandler = async () => { fetches += 1; throw new Error('unexpected'); };
  await chat.send('xxxxxxxx', { contextLimit: 10, maxTokens: 9, safeTotalBudget: 9 });
  assert.equal(snapshot.status, 'error');
  assert.equal(fetches, 0);
  assert.equal(conversationSets().length, 0);
});

test('system-prompt edits remain outside conversation checkpoints', () => {
  storage.resetCalls();
  chat.setSystemPrompt('edited only in prompt storage');
  assert.equal(snapshot.systemPrompt, 'edited only in prompt storage');
  assert.equal(conversationSets().length, 0);
  assert.equal(storage.values.get(SYSTEM_KEY), 'edited only in prompt storage');
});

test('valid import checkpoints messages while invalid import changes no transcript or record', () => {
  storage.resetCalls();
  chat.importChat(imported('imported system'));
  assert.equal(snapshot.systemPrompt, 'imported system');
  assert.equal(snapshot.persistenceWarning, null);
  assert.equal(conversationSets().length, 1);
  assert.equal(storage.values.get(SYSTEM_KEY), 'imported system');
  const beforeMessages = plainMessages();
  const beforeRecord = storage.values.get(CONVERSATION_KEY);
  storage.resetCalls();
  chat.importChat('{');
  assert.deepEqual(plainMessages(), beforeMessages);
  assert.equal(storage.values.get(CONVERSATION_KEY), beforeRecord);
  assert.equal(conversationSets().length, 0);

  storage.resetCalls();
  chat.importChat(JSON.stringify({ version: 1, systemPrompt: 'empty import', messages: [] }));
  assert.deepEqual(snapshot.messages, []);
  assert.equal(conversationSets().length, 1);
});

test('new chat clears memory and only the conversation key while retaining the prompt', () => {
  chat.importChat(imported('imported system'));
  storage.resetCalls();
  chat.newChat();
  assert.deepEqual(snapshot.messages, []);
  assert.equal(snapshot.status, 'idle');
  assert.equal(snapshot.error, null);
  assert.equal(snapshot.generationStartedAt, null);
  assert.equal(snapshot.generationMs, null);
  assert.equal(snapshot.systemPrompt, 'imported system');
  assert.deepEqual(storage.removeCalls, [CONVERSATION_KEY]);
  assert.equal(storage.values.get(SYSTEM_KEY), 'imported system');
  storage.resetCalls();
  chat.newChat();
  assert.deepEqual(storage.removeCalls, [CONVERSATION_KEY]);
});

test('storage failures keep memory and a later checkpoint clears the warning', async () => {
  chat.importChat(imported('kept system', 'before failed clear'));
  const oldRecord = storage.values.get(CONVERSATION_KEY);
  storage.throwRemove = true;
  chat.newChat();
  assert.equal(snapshot.messages.length, 0);
  assert.equal(snapshot.persistenceWarning, 'unavailable');
  assert.equal(storage.values.get(CONVERSATION_KEY), oldRecord);
  storage.throwRemove = false;

  storage.throwSet = true;
  const stream = controlledFetch();
  const pending = chat.send('memory survives quota', context);
  await tick();
  stream.delta('completed in memory');
  stream.done();
  await pending;
  assert.equal(snapshot.messages.at(-2)?.content, 'memory survives quota');
  assert.equal(snapshot.messages.at(-1)?.content, 'completed in memory');
  assert.equal(snapshot.persistenceWarning, 'unavailable');
  assert.equal(storage.values.get(CONVERSATION_KEY), oldRecord);
  storage.throwSet = false;
  chat.importChat(imported('kept system', 'warning recovers'));
  assert.equal(snapshot.persistenceWarning, null);
  assert.equal(snapshot.messages[0].content, 'warning recovers');
});
