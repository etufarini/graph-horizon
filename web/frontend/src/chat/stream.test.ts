/*
 * In-memory SSE tests cover split/CRLF framing, chunk activity, prohibited
 * protocol data, exact terminal usage, mandatory completion, and the immediate
 * terminal boundary. Fetch and chat-state behavior are excluded.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { readChatStream } from './stream.ts';

const encoder = new TextEncoder();

function body(chunks: string[], onCancel: () => void = () => {}): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
    cancel: onCancel
  });
}

test('split CRLF content and done report activity and complete', async () => {
  const events: string[] = [];
  let activity = 0;
  await readChatStream(
    body([
      ': ignored\r\n',
      'data: {"content":"hello"}\r\n',
      '\r\ndata: {"stats":{"prompt_tokens":2,"prefill_tokens":2,"completion_tokens":1,"prefill_ms":10,"decode_ms":20}}\r\n',
      'data: {"do',
      'ne":true}\r\n\r\n'
    ]),
    event => events.push(event.type),
    () => { activity += 1; }
  );
  assert.deepEqual(events, ['content', 'stats']);
  assert.equal(activity, 5);
});

test('done cancels immediately and ignores later bytes', async () => {
  const content: string[] = [];
  let cancelled = 0;
  await readChatStream(
    body([
      'data: {"stats":{"prompt_tokens":0,"prefill_tokens":0,"completion_tokens":0,"prefill_ms":0,"decode_ms":0}}\n',
      'data: {"done":true}\n',
      'data: {"content":"forbidden"}\n'
    ], () => { cancelled += 1; }),
    event => { if (event.type === 'content') content.push(event.content); }
  );
  assert.deepEqual(content, []);
  assert.equal(cancelled, 1);
});

test('EOF before done rejects even after content', async () => {
  const content: string[] = [];
  await assert.rejects(
    readChatStream(
      body(['data: {"content":"partial"}\n\n']),
      event => { if (event.type === 'content') content.push(event.content); }
    ),
    { message: 'Connection interrupted' }
  );
  assert.deepEqual(content, ['partial']);
});

test('invalid and prohibited data frames reject uniformly', async t => {
  const frames = [
    '',
    '{',
    'null',
    '{"error":"failed"}',
    '{"tool_event":{}}',
    '{"choices":[]}',
    '{"content":"x","extra":true}'
  ];
  for (const frame of frames) {
    await t.test(frame || 'empty data', async () => {
      await assert.rejects(
        readChatStream(body([`data: ${frame}\n\n`]), () => {}),
        { message: 'Connection interrupted' }
      );
    });
  }
});

test('phase, exact stats and done frames emit typed telemetry', async () => {
  const events: string[] = [];
  await readChatStream(
    body([
      'data: {"phase":"prefill"}\n',
      'data: {"phase":"decode"}\n',
      'data: {"stats":{"prompt_tokens":2,"prefill_tokens":1,"completion_tokens":1,"prefill_ms":10,"decode_ms":20}}\n',
      'data: {"done":true}\n'
    ]),
    event => events.push(event.type)
  );
  assert.deepEqual(events, ['phase', 'phase', 'stats']);
});

test('stats reject later content or phase frames', async t => {
  const stats = 'data: {"stats":{"prompt_tokens":2,"prefill_tokens":1,"completion_tokens":1,"prefill_ms":10,"decode_ms":20}}\n';
  for (const later of [
    'data: {"content":"late"}\n',
    'data: {"phase":"decode"}\n'
  ]) {
    await t.test(later.includes('content') ? 'content' : 'phase', async () => {
      await assert.rejects(
        readChatStream(body([stats, later, 'data: {"done":true}\n']), () => {}),
        { message: 'Connection interrupted' }
      );
    });
  }
});
