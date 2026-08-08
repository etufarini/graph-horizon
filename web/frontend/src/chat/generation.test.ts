/*
 * Deterministic generation acceptance tests use controlled in-memory streams
 * for append, replacement, capacity, rollback, stop, timing, and checkpoints.
 * Svelte rendering, real network, and real browser storage are excluded.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { get, writable } from 'svelte/store';

import { createGeneration } from './generation.ts';
import { createCollection, replaceActiveTranscript } from './sessions.ts';
import { hydrateTranscript } from './transcript.ts';
import type { ChatSnapshot, RuntimeContext } from './types.ts';

const id = '00000000-0000-4000-8000-000000000001';
const context: RuntimeContext = { contextLimit: 4096, safePromptBudget: 3686 };
const encoder = new TextEncoder();
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
let fetchHandler: typeof fetch = async () => { throw new Error('unexpected fetch'); };
globalThis.fetch = (...args) => fetchHandler(...args);

function snapshot(messages = [
  { role: 'user' as const, content: 'prima domanda' },
  { role: 'assistant' as const, content: 'prima risposta' }
]): ChatSnapshot {
  let collection = createCollection(1, () => id);
  collection = replaceActiveTranscript(collection, hydrateTranscript(messages), 2);
  return {
    collection,
    status: 'idle',
    error: null,
    persistenceWarning: null,
    systemPrompt: ' sistema ',
    generationStartedAt: null,
    generationMs: 99
  };
}

function plain(value: ChatSnapshot) {
  return value.collection.chats[0].messages.map(({ role, content }) => ({ role, content }));
}

function controlledFetch() {
  let streamController: ReadableStreamDefaultController<Uint8Array>;
  let requestBody: any;
  fetchHandler = async (_input, init) => {
    requestBody = JSON.parse(String(init?.body));
    const body = new ReadableStream<Uint8Array>({
      start(value) {
        streamController = value;
        init?.signal?.addEventListener('abort', () =>
          streamController.error(new DOMException('Aborted', 'AbortError')),
        { once: true });
      }
    });
    return new Response(body, { status: 200 });
  };
  return {
    body: () => requestBody,
    delta(content: string) {
      const data = JSON.stringify({ choices: [{ delta: { content } }] });
      streamController.enqueue(encoder.encode(`data: ${data}\n\n`));
    },
    done() {
      streamController.enqueue(encoder.encode('data: [DONE]\n\n'));
      streamController.close();
    },
    close() { streamController.close(); }
  };
}

function harness(initial = snapshot()) {
  const store = writable(initial);
  const checkpoints: string[] = [];
  return {
    store,
    checkpoints,
    generation: createGeneration(store, chatId => checkpoints.push(chatId))
  };
}

test('append streams without checkpoints and commits once after idle', async () => {
  const stream = controlledFetch();
  const { store, checkpoints, generation } = harness();
  const pending = generation.send('  nuova domanda  ', context);
  await tick();
  assert.equal(get(store).status, 'streaming');
  assert.deepEqual(checkpoints, []);
  assert.deepEqual(stream.body().messages, [
    { role: 'system', content: 'sistema' },
    { role: 'user', content: 'prima domanda' },
    { role: 'assistant', content: 'prima risposta' },
    { role: 'user', content: 'nuova domanda' }
  ]);
  stream.delta('[THINK]π[/THINK]');
  await tick();
  assert.equal(get(store).collection.chats[0].messages.at(-1)?.content, '[THINK]π[/THINK]');
  assert.deepEqual(checkpoints, []);
  stream.done();
  await pending;
  assert.equal(get(store).status, 'idle');
  assert.equal(typeof get(store).generationMs, 'number');
  assert.deepEqual(checkpoints, [id]);
});

test('append stop keeps empty or partial assistant and commits once', async () => {
  for (const partial of ['', 'parziale']) {
    const stream = controlledFetch();
    const { store, checkpoints, generation } = harness();
    const pending = generation.send('stop', context);
    await tick();
    if (partial) {
      stream.delta(partial);
      await tick();
    }
    generation.stop();
    await pending;
    assert.equal(plain(get(store)).at(-1)?.content, partial);
    assert.equal(get(store).generationMs, null);
    assert.deepEqual(checkpoints, [id]);
  }
});

test('append transport failures roll back and never checkpoint', async () => {
  const original = snapshot();
  for (const known of [true, false]) {
    const { store, checkpoints, generation } = harness(original);
    if (known) {
      fetchHandler = async () => new Response(null, { status: 500 });
    } else {
      const stream = controlledFetch();
      const pending = generation.send('fallisce', context);
      await tick();
      stream.delta('non stabile');
      stream.close();
      await pending;
      assert.deepEqual(plain(get(store)), plain(original));
      assert.equal(get(store).error, 'Connessione interrotta');
      assert.deepEqual(checkpoints, []);
      continue;
    }
    await generation.send('fallisce', context);
    assert.deepEqual(plain(get(store)), plain(original));
    assert.equal(get(store).error, 'Richiesta non riuscita');
    assert.deepEqual(checkpoints, []);
  }
});

test('regenerate excludes the old response and preserves the exact user prompt', async () => {
  const original = snapshot([
    { role: 'user', content: 'contesto' },
    { role: 'assistant', content: 'risposta contesto' },
    { role: 'user', content: '  domanda esatta  ' },
    { role: 'assistant', content: 'vecchia risposta' }
  ]);
  const stream = controlledFetch();
  const { store, checkpoints, generation } = harness(original);
  const pending = generation.regenerate(context);
  await tick();
  assert.deepEqual(stream.body().messages, [
    { role: 'system', content: 'sistema' },
    { role: 'user', content: 'contesto' },
    { role: 'assistant', content: 'risposta contesto' },
    { role: 'user', content: '  domanda esatta  ' }
  ]);
  stream.delta('nuova risposta');
  stream.done();
  await pending;
  assert.equal(plain(get(store)).at(-1)?.content, 'nuova risposta');
  assert.deepEqual(checkpoints, [id]);
});

test('edit trims the prompt, and a failed replacement restores the exact pair', async () => {
  const original = snapshot();
  const stream = controlledFetch();
  const { store, checkpoints, generation } = harness(original);
  const pending = generation.editLastPrompt('  modificata 🧠  ', context);
  await tick();
  assert.equal(plain(get(store)).at(-2)?.content, 'modificata 🧠');
  stream.delta('provvisoria');
  stream.close();
  await pending;
  assert.deepEqual(plain(get(store)), plain(original));
  assert.equal(get(store).error, 'Connessione interrotta');
  assert.deepEqual(checkpoints, []);
});

test('stopping replacement commits candidate prompt with empty response', async () => {
  controlledFetch();
  const { store, checkpoints, generation } = harness();
  const pending = generation.editLastPrompt('  candidata  ', context);
  await tick();
  generation.stop();
  await pending;
  assert.deepEqual(plain(get(store)).slice(-2), [
    { role: 'user', content: 'candidata' },
    { role: 'assistant', content: '' }
  ]);
  assert.deepEqual(checkpoints, [id]);
});

test('empty edits and capacity rejection perform no fetch, mutation, or checkpoint', async () => {
  let fetches = 0;
  fetchHandler = async () => { fetches += 1; throw new Error('unexpected'); };
  const initial = snapshot();
  const { store, checkpoints, generation } = harness(initial);
  await generation.editLastPrompt('   ', context);
  await generation.send('troppo lungo', {
    contextLimit: 10,
    safePromptBudget: 9
  });
  assert.equal(fetches, 0);
  assert.deepEqual(plain(get(store)), plain(initial));
  assert.match(get(store).error ?? '', /^Contesto insufficiente:/);
  assert.deepEqual(checkpoints, []);
});
