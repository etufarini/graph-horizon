/*
 * Controlled HTTP/watchdog tests cover pre-header inactivity, heartbeat reset,
 * request capacity, failed or bodyless responses, external Stop, and cleanup.
 * They use mocked timers and in-memory streams, never real waits or networking.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { MAX_REQUEST_BYTES, streamAssistant } from './client.ts';

const messages = [{ role: 'user' as const, content: 'hello' }];
const originalFetch = globalThis.fetch;
const finished = 'data: {"stats":{"prompt_tokens":2,"prefill_tokens":2,"completion_tokens":1,"prefill_ms":10,"decode_ms":20}}\n\ndata: {"done":true}\n\n';

test.after(() => { globalThis.fetch = originalFetch; });

test('inactivity before response headers aborts after five minutes', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  let internalSignal: AbortSignal | undefined;
  globalThis.fetch = async (_input, init) => {
    internalSignal = init?.signal ?? undefined;
    return await new Promise<Response>((_resolve, reject) => {
      internalSignal?.addEventListener('abort', () => reject(internalSignal?.reason), { once: true });
    });
  };

  const pending = streamAssistant(messages, () => {}, new AbortController().signal);
  t.mock.timers.tick(5 * 60_000);

  await assert.rejects(pending, { message: 'Connection interrupted' });
  assert.equal(internalSignal?.aborted, true);
});

test('non-empty chunks reset inactivity beyond five minutes total', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  let stream!: ReadableStreamDefaultController<Uint8Array>;
  let request: any;
  let cacheKey: string | null = null;
  let internalSignal: AbortSignal | undefined;
  globalThis.fetch = async (_input, init) => {
    request = JSON.parse(String(init?.body));
    cacheKey = new Headers(init?.headers).get('x-graph-horizon-cache');
    internalSignal = init?.signal ?? undefined;
    return new Response(new ReadableStream<Uint8Array>({ start(controller) { stream = controller; } }));
  };

  const external = new AbortController();
  const pending = streamAssistant(messages, () => {}, external.signal);
  await Promise.resolve();
  for (let heartbeat = 0; heartbeat < 3; heartbeat += 1) {
    t.mock.timers.tick(4 * 60_000);
    stream.enqueue(new TextEncoder().encode(': heartbeat\n\n'));
    await Promise.resolve();
    await Promise.resolve();
  }
  stream.enqueue(new TextEncoder().encode(finished));
  await pending;

  assert.deepEqual(request, { messages });
  assert.match(cacheKey ?? '', /^[0-9a-f]{32}$/);
  assert.equal(internalSignal?.aborted, false);
  external.abort();
  t.mock.timers.tick(5 * 60_000);
  assert.equal(internalSignal?.aborted, false);
});

test('cache key remains stable across requests in one page session', async () => {
  const keys: Array<string | null> = [];
  globalThis.fetch = async (_input, init) => {
    keys.push(new Headers(init?.headers).get('x-graph-horizon-cache'));
    return new Response(new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(finished));
        controller.close();
      }
    }));
  };

  await streamAssistant(messages, () => {}, new AbortController().signal);
  await streamAssistant(messages, () => {}, new AbortController().signal);

  assert.match(keys[0] ?? '', /^[0-9a-f]{32}$/);
  assert.equal(keys[1], keys[0]);
});

test('unsuccessful and bodyless responses use request failure', async () => {
  for (const response of [new Response(null, { status: 500 }), new Response(null)]) {
    globalThis.fetch = async () => response;
    await assert.rejects(
      streamAssistant(messages, () => {}, new AbortController().signal),
      { message: 'Request failed' }
    );
  }
});

test('Web search adds only the bounded query field and reports upstream failure', async () => {
  let request: unknown;
  globalThis.fetch = async (_input, init) => {
    request = JSON.parse(String(init?.body));
    return new Response(new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(finished));
        controller.close();
      }
    }));
  };
  await streamAssistant(messages, () => {}, new AbortController().signal, 'fresh query');
  assert.deepEqual(request, { messages, search_query: 'fresh query' });

  globalThis.fetch = async () => new Response(null, { status: 502 });
  await assert.rejects(
    streamAssistant(messages, () => {}, new AbortController().signal, 'fresh query'),
    { message: 'Web search unavailable' }
  );
});

test('invalid Web query is rejected before fetch', async () => {
  let fetches = 0;
  globalThis.fetch = async () => { fetches += 1; throw new Error('unexpected fetch'); };
  await assert.rejects(
    streamAssistant(messages, () => {}, new AbortController().signal, 'x'.repeat(513)),
    { message: 'Request failed' }
  );
  assert.equal(fetches, 0);
});

test('an oversized assembled JSON body is rejected before fetch', async () => {
  let fetches = 0;
  globalThis.fetch = async () => {
    fetches += 1;
    throw new Error('unexpected fetch');
  };
  await assert.rejects(
    streamAssistant(
      [{ role: 'user', content: 'x'.repeat(MAX_REQUEST_BYTES) }],
      () => {},
      new AbortController().signal
    ),
    { message: 'Request failed' }
  );
  assert.equal(fetches, 0);
});

test('external abort remains a recognizable voluntary Stop', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  let internalSignal: AbortSignal | undefined;
  globalThis.fetch = async (_input, init) => {
    internalSignal = init?.signal ?? undefined;
    return await new Promise<Response>((_resolve, reject) => {
      internalSignal?.addEventListener('abort', () => reject(internalSignal?.reason), { once: true });
    });
  };

  const external = new AbortController();
  const pending = streamAssistant(messages, () => {}, external.signal);
  external.abort();

  await assert.rejects(pending, error => error instanceof DOMException && error.name === 'AbortError');
  assert.equal(internalSignal?.aborted, true);
});
