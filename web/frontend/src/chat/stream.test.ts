/*
 * In-memory SSE tests cover split/CRLF framing, chunk activity, prohibited
 * protocol data, neutral usage/final frames, mandatory completion, and the
 * immediate terminal boundary. Fetch and chat-state behavior are excluded.
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

test('split CRLF content and DONE report activity and complete', async () => {
  const content: string[] = [];
  let activity = 0;
  await readChatStream(
    body([
      ': ignored\r\n',
      'data: {"choices":[{"delta":{"content":"ciao"}}]}\r\n',
      '\r\ndata: [DO',
      'NE]\r\n\r\n'
    ]),
    delta => content.push(delta.content),
    () => { activity += 1; }
  );
  assert.deepEqual(content, ['ciao']);
  assert.equal(activity, 4);
});

test('DONE cancels immediately and ignores later bytes', async () => {
  const content: string[] = [];
  let cancelled = 0;
  await readChatStream(
    body([
      'data: [DONE]\n',
      'data: {"choices":[{"delta":{"content":"vietato"}}]}\n'
    ], () => { cancelled += 1; }),
    delta => content.push(delta.content)
  );
  assert.deepEqual(content, []);
  assert.equal(cancelled, 1);
});

test('EOF before DONE rejects even after content', async () => {
  const content: string[] = [];
  await assert.rejects(
    readChatStream(
      body(['data: {"choices":[{"delta":{"content":"parziale"}}]}\n\n']),
      delta => content.push(delta.content)
    ),
    { message: 'Connessione interrotta' }
  );
  assert.deepEqual(content, ['parziale']);
});

test('invalid and prohibited data frames reject uniformly', async t => {
  const frames = [
    '',
    '{',
    'null',
    '{"error":{"message":"secret"}}',
    '{"tool_event":{}}',
    '{"choices":[{"delta":{"tool_calls":[]}}]}',
    '{"choices":[{"delta":{"reasoning_content":"hidden"}}]}'
  ];
  for (const frame of frames) {
    await t.test(frame || 'empty data', async () => {
      await assert.rejects(
        readChatStream(body([`data: ${frame}\n\n`]), () => {}),
        { message: 'Connessione interrotta' }
      );
    });
  }
});

test('usage and final-stop frames are presentation-neutral', async () => {
  const content: string[] = [];
  await readChatStream(
    body([
      'data: {"usage":{"prompt_tokens":2}}\n',
      'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
      'data: [DONE]\n'
    ]),
    delta => content.push(delta.content)
  );
  assert.deepEqual(content, []);
});
