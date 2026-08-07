/*
 * DOM-free acceptance tests for transcript validation, hydration, wire/prior
 * projection, append, and final-pair reads/replacements/removal. Lifecycle,
 * storage, and rendering are excluded.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  appendAssistant,
  beforeFinalPair,
  finalPair,
  hydrateTranscript,
  removeTrailingTurn,
  replaceFinalPair,
  validateTranscript,
  wireMessages
} from './transcript.ts';
import { parseChatFile, serializeChat } from './transfer.ts';
import type { ChatMessage } from './types.ts';

const plain = [
  { role: 'user' as const, content: '  ciao 🧠\n' },
  { role: 'assistant' as const, content: '[THINK]π[/THINK]\n ✓  ' }
];

test('validation preserves complete alternating Unicode transcripts exactly', () => {
  assert.deepEqual(validateTranscript(plain), plain);
  assert.deepEqual(validateTranscript([]), []);
  assert.deepEqual(
    validateTranscript([
      { role: 'user', content: 'x', ignored: true },
      { role: 'assistant', content: '', ignored: true }
    ]),
    [{ role: 'user', content: 'x' }, { role: 'assistant', content: '' }]
  );
});

test('validation rejects the whole invalid transcript without a valid prefix', () => {
  for (const value of [
    [{ role: 'user', content: 'odd' }],
    [{ role: 'assistant', content: 'first' }, { role: 'user', content: 'second' }],
    [{ role: 'user', content: 'a' }, { role: 'user', content: 'b' }],
    [{ role: 'user', content: 1 }, { role: 'assistant', content: 'b' }],
    [{ role: 'tool', content: 'a' }, { role: 'assistant', content: 'b' }],
    [...plain, { role: 'user', content: 'valid prefix' }, null]
  ]) {
    assert.equal(validateTranscript(value), null);
  }
  assert.equal(validateTranscript({}), null);
});

test('file transfer keeps version 1 and delegates complete-pair validation', () => {
  const text = serializeChat(hydrateTranscript(plain), '  sistema  ');
  assert.match(text, /^\{\n  "version": 1,/);
  assert.deepEqual(parseChatFile(text), {
    ok: true,
    payload: { systemPrompt: '  sistema  ', messages: plain }
  });
  assert.deepEqual(parseChatFile('{'), { ok: false, error: 'invalid-json' });
  assert.deepEqual(
    parseChatFile(JSON.stringify({ version: 1, systemPrompt: '', messages: [plain[0]] })),
    { ok: false, error: 'invalid-format' }
  );
});

test('hydration creates fresh runtime IDs without changing plain messages', () => {
  const first = hydrateTranscript(plain);
  const second = hydrateTranscript(plain);
  assert.deepEqual(first.map(({ role, content }) => ({ role, content })), plain);
  assert.notDeepEqual(first.map(message => message.id), second.map(message => message.id));
  assert.equal('id' in plain[0], false);
});

test('hydration has a deterministic unique fallback without randomUUID', () => {
  const crypto = globalThis.crypto;
  Object.defineProperty(globalThis, 'crypto', { value: {}, configurable: true });
  try {
    const messages = hydrateTranscript(plain);
    assert.notEqual(messages[0].id, messages[1].id);
    assert.match(messages[0].id, /^user-/);
    assert.match(messages[1].id, /^assistant-/);
  } finally {
    Object.defineProperty(globalThis, 'crypto', { value: crypto, configurable: true });
  }
});

test('wire projection orders trimmed system and draft around unchanged history', () => {
  const messages = hydrateTranscript(plain);
  assert.deepEqual(wireMessages(messages, '  sistema  ', '  domanda  '), [
    { role: 'system', content: 'sistema' },
    ...plain,
    { role: 'user', content: 'domanda' }
  ]);
  assert.deepEqual(wireMessages(messages, '  ', '  '), plain);
});

test('assistant append is pure and supports empty content', () => {
  const messages = hydrateTranscript(plain);
  const appended = appendAssistant(messages, ' altro');
  assert.notEqual(appended, messages);
  assert.equal(messages[1].content, plain[1].content);
  assert.equal(appended[1].content, `${plain[1].content} altro`);
  const userOnly = messages.slice(0, 1);
  assert.equal(appendAssistant(userOnly, ''), userOnly);
});

test('rollback removes only the expected trailing pair', () => {
  const earlier: ChatMessage[] = hydrateTranscript(plain);
  const active = hydrateTranscript([
    { role: 'user', content: 'active' },
    { role: 'assistant', content: '' }
  ]);
  const messages = [...earlier, ...active];
  assert.deepEqual(removeTrailingTurn(messages, active[0].id, active[1].id), earlier);
  assert.equal(removeTrailingTurn(messages, 'wrong', active[1].id), messages);
  assert.equal(removeTrailingTurn(messages, active[0].id, 'wrong'), messages);
});

test('final-pair operations preserve IDs, Unicode, and immutability', () => {
  const earlier = hydrateTranscript(plain);
  const active = hydrateTranscript([
    { role: 'user', content: 'ultima 🧠' },
    { role: 'assistant', content: '' }
  ]);
  const messages = [...earlier, ...active];

  assert.deepEqual(finalPair(messages), active);
  assert.deepEqual(beforeFinalPair(messages), earlier);
  assert.notEqual(beforeFinalPair(messages), messages);

  const replaced = replaceFinalPair(
    messages,
    active[0].id,
    active[1].id,
    'nuova domanda π',
    '[THINK]✓[/THINK]'
  );
  assert.notEqual(replaced, messages);
  assert.equal(replaced.at(-2)?.id, active[0].id);
  assert.equal(replaced.at(-1)?.id, active[1].id);
  assert.equal(replaced.at(-2)?.content, 'nuova domanda π');
  assert.equal(replaced.at(-1)?.content, '[THINK]✓[/THINK]');
  assert.equal(messages.at(-2)?.content, 'ultima 🧠');
  assert.equal(messages.at(-1)?.content, '');
});

test('final-pair transformations reject failed preconditions without allocation', () => {
  const messages = hydrateTranscript(plain);
  const userOnly = messages.slice(0, 1);

  assert.equal(finalPair([]), null);
  assert.equal(finalPair(userOnly), null);
  assert.equal(beforeFinalPair(userOnly), userOnly);
  assert.equal(replaceFinalPair(messages, 'wrong', messages[1].id, 'x', 'y'), messages);
  assert.equal(replaceFinalPair(messages, messages[0].id, 'wrong', 'x', 'y'), messages);
  assert.equal(removeTrailingTurn(userOnly, userOnly[0].id, 'missing'), userOnly);
});
