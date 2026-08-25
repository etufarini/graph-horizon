/*
 * Controlled generation tests cover prompt-only capacity, request allowance,
 * timeout/interruption rollback, voluntary Stop, first terminal outcome,
 * timing, raw Reasoning retention, and checkpoints. Svelte rendering, real
 * networking, and browser storage are excluded.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { get, writable } from 'svelte/store';

import { createGeneration } from './generation.ts';
import { defaultSearch } from './search.ts';
import { createCollection, replaceActiveTranscript } from './sessions.ts';
import { hydrateTranscript } from './transcript.ts';
import type { ChatSnapshot, RuntimeContext, TranscriptMessage } from './types.ts';
import type { MarkdownFileRecord } from './files/record.ts';

const id = '00000000-0000-4000-8000-000000000001';
const context: RuntimeContext = {
  contextLimit: 4096,
  safePromptBudget: 3686,
  search: { provider: 'search.example', maxQueryCharacters: 512, maxContextCharacters: 2800 }
};
const encoder = new TextEncoder();
const providerReport = {
  query: 'public query', category: 'web', reference_date: '2026-08-24',
  published: null, provider: 'search.example',
  sources: [{ id: 'S1', title: 'Result', url: 'https://example.com/result', publisher: null, published_at_ms: null }]
};
const storedReport = {
  query: 'old query', category: 'web' as const, referenceDate: '2026-08-23',
  published: null, provider: 'search.example',
  sources: [{ id: 'S1', title: 'Old result', url: 'https://example.com/old', publisher: null, publishedAtMs: null }]
};
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
let fetchHandler: typeof fetch = async () => { throw new Error('unexpected fetch'); };
globalThis.fetch = (...args) => fetchHandler(...args);

function snapshot(messages: TranscriptMessage[] = [
  { role: 'user' as const, content: 'first question' },
  { role: 'assistant' as const, content: 'first response' }
]): ChatSnapshot {
  let collection = createCollection(1, () => id, ' system ');
  collection = replaceActiveTranscript(collection, hydrateTranscript(messages), 2);
  return {
    collection,
    status: 'idle',
    error: null,
    persistenceWarning: null,
    telemetry: null
  };
}

function plain(value: ChatSnapshot) {
  return value.collection.chats[0].messages.map(({ role, content }) => ({ role, content }));
}

function controlledFetch(report: object | null = null) {
  let streamController: ReadableStreamDefaultController<Uint8Array>;
  let requestBody: any;
  fetchHandler = async (_input, init) => {
    requestBody = JSON.parse(String(init?.body));
    const body = new ReadableStream<Uint8Array>({
      start(value) {
        streamController = value;
        streamController.enqueue(encoder.encode(
          (report ? `data: ${JSON.stringify({ search: report })}\n\n` : '') +
          'data: {"phase":"prefill"}\n\n' +
          'data: {"phase":"decode"}\n\n'
        ));
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
      const data = JSON.stringify({ content });
      streamController.enqueue(encoder.encode(`data: ${data}\n\n`));
    },
    frame(value: unknown) {
      streamController.enqueue(encoder.encode(`data: ${JSON.stringify(value)}\n\n`));
    },
    done() {
      streamController.enqueue(encoder.encode(
        'data: {"stats":{"prompt_tokens":12,"prefill_tokens":8,"completion_tokens":3,"prefill_ms":40,"decode_ms":60}}\n\n' +
        'data: {"done":true}\n\n'
      ));
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
  const pending = generation.send('  new question  ', context);
  await tick();
  assert.equal(get(store).status, 'streaming');
  assert.deepEqual(checkpoints, []);
  assert.deepEqual(stream.body().messages, [
    { role: 'system', content: 'system' },
    { role: 'user', content: 'first question' },
    { role: 'assistant', content: 'first response' },
    { role: 'user', content: 'new question' }
  ]);
  assert.deepEqual(Object.keys(stream.body()), ['messages']);
  stream.delta('[THINK]π[/THINK]');
  await tick();
  assert.equal(get(store).collection.chats[0].messages.at(-1)?.content, '[THINK]π[/THINK]');
  assert.deepEqual(checkpoints, []);
  stream.done();
  await pending;
  assert.equal(get(store).status, 'idle');
  assert.equal(get(store).telemetry?.stats?.completionTokens, 3);
  assert.deepEqual(checkpoints, [id]);
});

test('Markdown files expand only the outgoing user copy', async () => {
  const stream = controlledFetch();
  const { store, generation } = harness();
  const files: MarkdownFileRecord[] = [{
    id: '00000000-0000-4000-8000-000000000002',
    chatId: id,
    name: 'note.md',
    content: '# Source',
    utf8Bytes: 7,
    addedAt: 1
  }];
  const pending = generation.send('visible question', context, files);
  await tick();
  const sent = stream.body().messages.at(-1).content as string;
  assert.match(sent, /### File: note\.md/);
  assert.ok(sent.endsWith('### User request\nvisible question'));
  assert.equal(plain(get(store)).at(-2)?.content, 'visible question');
  stream.done();
  await pending;
  assert.equal(plain(get(store)).at(-2)?.content, 'visible question');
});

test('Web search sends the visible query apart from expanded Markdown context', async () => {
  const stream = controlledFetch();
  const { generation } = harness();
  const files: MarkdownFileRecord[] = [{
    id: '00000000-0000-4000-8000-000000000002',
    chatId: id,
    name: 'note.md',
    content: '# Source',
    utf8Bytes: 7,
    addedAt: 1
  }];
  const pending = generation.send(' visible question ', context, files, defaultSearch());
  await tick();
  assert.equal(stream.body().search.terms, 'visible question');
  assert.match(stream.body().search.reference_date, /^\d{4}-\d{2}-\d{2}$/);
  assert.match(stream.body().messages.at(-1).content, /### File: note\.md/);
  assert.deepEqual(Object.keys(stream.body()), ['messages', 'search']);
  stream.done();
  await pending;
});

test('explicit query and provenance never contaminate later model history', async () => {
  const stream = controlledFetch(providerReport);
  const { store, generation } = harness();
  const selection = { ...defaultSearch(), query: ' public query ' };
  const pending = generation.send('private visible prompt', context, [], selection);
  await tick();
  assert.equal(stream.body().search.terms, 'public query');
  stream.done();
  await pending;
  assert.equal(get(store).collection.chats[0].messages.at(-1)?.search?.query, 'public query');

  const next = controlledFetch();
  const followUp = generation.send('follow up', context);
  await tick();
  const priorAssistant = next.body().messages.at(-2);
  assert.deepEqual(Object.keys(priorAssistant), ['role', 'content']);
  next.done();
  await followUp;
});

test('Web search reserves its maximum context before visible mutation or fetch', async () => {
  let fetches = 0;
  fetchHandler = async () => { fetches += 1; throw new Error('unexpected fetch'); };
  const { store, generation } = harness(snapshot([]));
  await generation.send('x', {
    contextLimit: 10,
    safePromptBudget: 1,
    search: { ...context.search, maxContextCharacters: 8 }
  }, [], defaultSearch());
  assert.equal(fetches, 0);
  assert.deepEqual(plain(get(store)), []);
  assert.equal(
    get(store).error,
    'Insufficient context: ~3 estimated tokens exceed the safe budget of 1 tokens'
  );
});

test('valid zero-delta completion retains the empty response and checkpoints', async () => {
  const stream = controlledFetch();
  const { store, checkpoints, generation } = harness();
  const pending = generation.send('no tokens', context);
  await tick();
  stream.done();
  await pending;
  assert.equal(plain(get(store)).at(-1)?.content, '');
  assert.equal(get(store).telemetry?.stats?.completionTokens, 3);
  assert.deepEqual(checkpoints, [id]);
});

test('append stop keeps empty or partial assistant and commits once', async () => {
  for (const partial of ['', 'partial', '[THINK]step']) {
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
    assert.equal(get(store).telemetry, null);
    assert.equal(get(store).error, null);
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
      const pending = generation.send('fails', context);
      await tick();
      stream.delta('unstable');
      stream.close();
      await pending;
      assert.deepEqual(plain(get(store)), plain(original));
      assert.equal(get(store).error, 'Response interrupted');
      assert.deepEqual(checkpoints, []);
      continue;
    }
    await generation.send('fails', context);
    assert.deepEqual(plain(get(store)), plain(original));
    assert.equal(get(store).error, 'Request failed');
    assert.deepEqual(checkpoints, []);
  }
});

test('Web search failure rolls back with its specific generic error', async () => {
  fetchHandler = async () => new Response(null, { status: 502 });
  const original = snapshot();
  const { store, checkpoints, generation } = harness(original);
  await generation.send('needs current facts', context, [], defaultSearch());
  assert.deepEqual(plain(get(store)), plain(original));
  assert.equal(get(store).error, 'Web search is unavailable; no answer was generated');
  assert.deepEqual(checkpoints, []);
});

test('regenerate excludes the old response and preserves the exact user prompt', async () => {
  const original = snapshot([
    { role: 'user', content: 'context' },
    { role: 'assistant', content: 'context response' },
    { role: 'user', content: '  exact question  ' },
    { role: 'assistant', content: 'old response' }
  ]);
  const stream = controlledFetch();
  const { store, checkpoints, generation } = harness(original);
  const pending = generation.regenerate(context);
  await tick();
  assert.deepEqual(stream.body().messages, [
    { role: 'system', content: 'system' },
    { role: 'user', content: 'context' },
    { role: 'assistant', content: 'context response' },
    { role: 'user', content: '  exact question  ' }
  ]);
  stream.delta('new response');
  stream.done();
  await pending;
  assert.equal(plain(get(store)).at(-1)?.content, 'new response');
  assert.deepEqual(checkpoints, [id]);
});

test('regenerate recreates an assistant and never carries old search provenance', async () => {
  const original = snapshot([
    { role: 'user', content: 'question' },
    { role: 'assistant', content: 'old response', search: storedReport }
  ]);
  const oldAssistant = original.collection.chats[0].messages[1];
  const stream = controlledFetch();
  const { store, generation } = harness(original);
  const pending = generation.regenerate(context);
  await tick();
  const replacement = get(store).collection.chats[0].messages[1];
  assert.notEqual(replacement.id, oldAssistant.id);
  assert.equal(replacement.search, undefined);
  stream.delta('new response');
  stream.done();
  await pending;
  assert.equal(get(store).collection.chats[0].messages[1].search, undefined);
});

test('regenerate reruns an unchanged search and stores only its new report', async () => {
  const original = snapshot([
    { role: 'user', content: 'question' },
    { role: 'assistant', content: 'old response', search: storedReport }
  ]);
  const report = { ...providerReport, query: storedReport.query };
  const stream = controlledFetch(report);
  const { store, generation } = harness(original);
  const pending = generation.regenerate(
    context,
    [],
    { ...defaultSearch(), query: storedReport.query }
  );
  await tick();
  assert.equal(stream.body().search.terms, storedReport.query);
  stream.done();
  await pending;
  const search = get(store).collection.chats[0].messages[1].search;
  assert.equal(search?.query, storedReport.query);
  assert.equal(search?.referenceDate, providerReport.reference_date);
});

test('edit replaces old search provenance with only the new report', async () => {
  const original = snapshot([
    { role: 'user', content: 'old question' },
    { role: 'assistant', content: 'old response', search: storedReport }
  ]);
  const userId = original.collection.chats[0].messages[0].id;
  const stream = controlledFetch(providerReport);
  const { store, generation } = harness(original);
  const pending = generation.editPrompt(userId, 'new question', context, [], defaultSearch());
  await tick();
  assert.equal(
    get(store).collection.chats[0].messages[1].search?.query,
    'public query'
  );
  stream.done();
  await pending;
});

test('editing an earlier prompt truncates successors and sends only its causal prefix', async () => {
  const original = snapshot([
    { role: 'user', content: 'first question' },
    { role: 'assistant', content: 'first response' },
    { role: 'user', content: 'next question' },
    { role: 'assistant', content: 'next response' }
  ]);
  const userId = original.collection.chats[0].messages[0].id;
  const stream = controlledFetch();
  const { store, checkpoints, generation } = harness(original);
  const pending = generation.editPrompt(userId, '  edited 🧠  ', context);
  await tick();
  assert.deepEqual(stream.body().messages, [
    { role: 'system', content: 'system' },
    { role: 'user', content: 'edited 🧠' }
  ]);
  assert.equal(plain(get(store)).length, 2);
  assert.equal(plain(get(store)).at(-2)?.content, 'edited 🧠');
  stream.delta('new path');
  stream.done();
  await pending;
  assert.deepEqual(plain(get(store)), [
    { role: 'user', content: 'edited 🧠' },
    { role: 'assistant', content: 'new path' }
  ]);
  assert.deepEqual(checkpoints, [id]);
});

test('a failed earlier edit restores the exact complete transcript', async () => {
  const original = snapshot([
    { role: 'user', content: 'first question' },
    { role: 'assistant', content: 'first response', search: storedReport },
    { role: 'user', content: 'next question' },
    { role: 'assistant', content: 'next response' }
  ]);
  const userId = original.collection.chats[0].messages[0].id;
  const stream = controlledFetch();
  const { store, checkpoints, generation } = harness(original);
  const pending = generation.editPrompt(userId, 'edited', context);
  await tick();
  stream.delta('temporary');
  stream.close();
  await pending;
  assert.deepEqual(
    get(store).collection.chats[0].messages,
    original.collection.chats[0].messages
  );
  assert.equal(get(store).error, 'Response interrupted');
  assert.deepEqual(checkpoints, []);
});

test('stopping replacement commits candidate prompt without stale search provenance', async () => {
  controlledFetch();
  const original = snapshot([
    { role: 'user', content: 'first question' },
    { role: 'assistant', content: 'first response', search: storedReport }
  ]);
  const { store, checkpoints, generation } = harness(original);
  const userId = get(store).collection.chats[0].messages[0].id;
  const pending = generation.editPrompt(userId, '  candidate  ', context);
  await tick();
  generation.stop();
  await pending;
  assert.deepEqual(plain(get(store)).slice(-2), [
    { role: 'user', content: 'candidate' },
    { role: 'assistant', content: '' }
  ]);
  assert.equal(get(store).collection.chats[0].messages.at(-1)?.search, undefined);
  assert.deepEqual(checkpoints, [id]);
});

test('empty edits and capacity rejection perform no fetch, mutation, or checkpoint', async () => {
  let fetches = 0;
  fetchHandler = async () => { fetches += 1; throw new Error('unexpected'); };
  const initial = snapshot();
  const { store, checkpoints, generation } = harness(initial);
  const userId = initial.collection.chats[0].messages[0].id;
  await generation.editPrompt(userId, '   ', context);
  await generation.editPrompt('missing', 'valid', context);
  await generation.editPrompt(userId, 'far too long', {
    contextLimit: 4,
    safePromptBudget: 3,
    search: context.search
  });
  assert.equal(fetches, 0);
  assert.deepEqual(plain(get(store)), plain(initial));
  assert.equal(
    get(store).error,
    'Insufficient context: ~4 estimated tokens exceed the safe budget of 3 tokens'
  );
  assert.deepEqual(checkpoints, []);
});

test('timeout wins over a later Stop and rolls back partial output', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const original = snapshot();
  const stream = controlledFetch();
  const { store, checkpoints, generation } = harness(original);
  const pending = generation.send('timeout', context);
  await Promise.resolve();
  await Promise.resolve();
  stream.delta('temporary');
  await Promise.resolve();
  await Promise.resolve();

  t.mock.timers.tick(5 * 60_000);
  generation.stop();
  await pending;

  assert.deepEqual(plain(get(store)), plain(original));
  assert.equal(get(store).error, 'Response interrupted');
  assert.equal(get(store).telemetry, null);
  assert.deepEqual(checkpoints, []);
});

test('Stop wins over the later watchdog and checkpoints once', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  controlledFetch();
  const { store, checkpoints, generation } = harness();
  const pending = generation.send('stop first', context);
  await Promise.resolve();
  await Promise.resolve();

  generation.stop();
  t.mock.timers.tick(5 * 60_000);
  await pending;

  assert.equal(get(store).status, 'idle');
  assert.equal(get(store).error, null);
  assert.deepEqual(checkpoints, [id]);
});

test('engine and protocol error frames roll back without leaking details', async () => {
  const original = snapshot();
  for (const frame of [
    { error: { message: 'E11 prompt has 5000 tokens but context is 4096' } },
    { choices: [{ delta: { reasoning_content: 'hidden' } }] }
  ]) {
    const stream = controlledFetch();
    const { store, checkpoints, generation } = harness(original);
    const pending = generation.send('fails', context);
    await tick();
    stream.delta('temporary');
    await tick();
    stream.frame(frame);
    await pending;
    assert.deepEqual(plain(get(store)), plain(original));
    assert.equal(get(store).error, 'Response interrupted');
    assert.deepEqual(checkpoints, []);
  }
});
