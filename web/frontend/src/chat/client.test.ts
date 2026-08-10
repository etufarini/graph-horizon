/*
 * Controlled HTTP/watchdog tests cover pre-header inactivity, heartbeat reset,
 * request capacity, failed or bodyless responses, external Stop, and cleanup.
 * They use mocked timers and in-memory streams, never real waits or networking.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { streamAssistant } from './client.ts';

const messages = [{ role: 'user' as const, content: 'ciao' }];
const originalFetch = globalThis.fetch;

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

  const pending = streamAssistant(messages, 4096, () => {}, new AbortController().signal);
  t.mock.timers.tick(5 * 60_000);

  await assert.rejects(pending, { message: 'Connessione interrotta' });
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
  const pending = streamAssistant(messages, 8192, () => {}, external.signal);
  await Promise.resolve();
  for (let heartbeat = 0; heartbeat < 3; heartbeat += 1) {
    t.mock.timers.tick(4 * 60_000);
    stream.enqueue(new TextEncoder().encode(': heartbeat\n\n'));
    await Promise.resolve();
    await Promise.resolve();
  }
  stream.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
  await pending;

  assert.equal(request.max_tokens, 8192);
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
        controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
        controller.close();
      }
    }));
  };

  await streamAssistant(messages, 4096, () => {}, new AbortController().signal);
  await streamAssistant(messages, 4096, () => {}, new AbortController().signal);

  assert.match(keys[0] ?? '', /^[0-9a-f]{32}$/);
  assert.equal(keys[1], keys[0]);
});

test('unsuccessful and bodyless responses use request failure', async () => {
  for (const response of [new Response(null, { status: 500 }), new Response(null)]) {
    globalThis.fetch = async () => response;
    await assert.rejects(
      streamAssistant(messages, 4096, () => {}, new AbortController().signal),
      { message: 'Richiesta non riuscita' }
    );
  }
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
  const pending = streamAssistant(messages, 4096, () => {}, external.signal);
  external.abort();

  await assert.rejects(pending, error => error instanceof DOMException && error.name === 'AbortError');
  assert.equal(internalSignal?.aborted, true);
});
