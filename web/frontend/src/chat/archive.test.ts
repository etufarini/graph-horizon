/*
 * Acceptance tests for untrusted private archive strings: exact parsing,
 * invariant rejection, legacy recognition, and UTF-8 serialization bounds.
 * Browser storage I/O is excluded.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  FORMAT_VERSION,
  MAX_RECORD_BYTES,
  parseArchive,
  serializeArchive,
  STORAGE_KEY
} from './archive.ts';
import { createCollection, replaceActiveTranscript } from './sessions.ts';
import { hydrateTranscript } from './transcript.ts';

const firstId = '00000000-0000-4000-8000-000000000001';
const secondId = '00000000-0000-4000-8000-000000000002';
const source = (id: string) => () => id;
const plain = [
  { role: 'user' as const, content: 'ciao 🧠' },
  { role: 'assistant' as const, content: '[THINK]π[/THINK]' }
];

function validRecord() {
  return {
    version: FORMAT_VERSION,
    activeChatId: firstId,
    chats: [{
      id: firstId,
      title: 'Titolo',
      messages: plain,
      updatedAt: 42
    }]
  };
}

test('version-2 parsing accepts only the exact collection shape', () => {
  assert.equal(STORAGE_KEY, 'graph-horizon.conversation');
  const result = parseArchive(JSON.stringify(validRecord()));
  assert.equal(result.kind, 'current');
  if (result.kind !== 'current') return;
  assert.equal(result.collection.activeChatId, firstId);
  assert.deepEqual(
    result.collection.chats[0].messages.map(({ role, content }) => ({ role, content })),
    plain
  );
  assert.equal(result.collection.chats[0].messages.every(message => message.id.length > 0), true);
});

test('malformed, unknown, missing, and extra top-level fields are invalid', () => {
  const record = validRecord();
  const invalid = [
    '{',
    'null',
    JSON.stringify({ ...record, version: 3 }),
    JSON.stringify({ version: 2, activeChatId: firstId }),
    JSON.stringify({ ...record, extra: true }),
    JSON.stringify({ ...record, chats: [] })
  ];
  for (const raw of invalid) {
    assert.deepEqual(parseArchive(raw), { kind: 'invalid' });
  }
});

test('chat IDs must be unique UUIDs and include the active ID', () => {
  const record = validRecord();
  for (const changed of [
    { ...record, activeChatId: 'invalid' },
    { ...record, activeChatId: secondId },
    { ...record, chats: [{ ...record.chats[0], id: 'invalid' }] },
    { ...record, chats: [record.chats[0], { ...record.chats[0] }] }
  ]) {
    assert.deepEqual(parseArchive(JSON.stringify(changed)), { kind: 'invalid' });
  }
});

test('chat fields reject invalid title, timestamp, transcript, and extra keys', () => {
  const record = validRecord();
  const chat = record.chats[0];
  const variants = [
    { ...chat, title: '' },
    { ...chat, title: '🧠'.repeat(81) },
    { ...chat, updatedAt: -1 },
    { ...chat, updatedAt: 1.5 },
    { ...chat, updatedAt: Number.MAX_SAFE_INTEGER + 1 },
    { ...chat, messages: [{ role: 'user', content: 'odd' }] },
    { ...chat, messages: [{ ...plain[0], extra: true }, plain[1]] },
    { ...chat, extra: true }
  ];
  for (const changed of variants) {
    assert.deepEqual(
      parseArchive(JSON.stringify({ ...record, chats: [changed] })),
      { kind: 'invalid' }
    );
  }
});

test('legacy version 1 migrates exact messages and derives one active chat', () => {
  const result = parseArchive(
    JSON.stringify({ version: 1, messages: plain }),
    77,
    source(secondId)
  );
  assert.equal(result.kind, 'legacy');
  if (result.kind !== 'legacy') return;
  assert.equal(result.collection.activeChatId, secondId);
  assert.deepEqual(result.collection.chats[0], {
    id: secondId,
    title: 'ciao 🧠',
    messages: result.collection.chats[0].messages,
    updatedAt: 77
  });
  assert.deepEqual(
    result.collection.chats[0].messages.map(({ role, content }) => ({ role, content })),
    plain
  );
});

test('legacy recognition rejects extra fields and invalid message objects', () => {
  for (const value of [
    { version: 1, messages: plain, extra: true },
    { version: 1, messages: [{ ...plain[0], id: 'runtime' }, plain[1]] },
    { version: 1, messages: [plain[0]] }
  ]) {
    assert.deepEqual(parseArchive(JSON.stringify(value)), { kind: 'invalid' });
  }
});

test('serialization strips runtime IDs and emits no fields beyond version 2', () => {
  let collection = createCollection(42, source(firstId));
  collection = replaceActiveTranscript(collection, hydrateTranscript(plain), 43);
  const serialized = serializeArchive(collection);
  assert.equal(serialized.ok, true);
  if (!serialized.ok) return;
  assert.deepEqual(JSON.parse(serialized.raw), {
    version: 2,
    activeChatId: firstId,
    chats: [{ id: firstId, title: 'ciao 🧠', messages: plain, updatedAt: 43 }]
  });
});

test('serialization accepts exactly 4 MiB and rejects one UTF-8 byte over', () => {
  let collection = createCollection(1, source(firstId));
  collection = replaceActiveTranscript(collection, hydrateTranscript([
    { role: 'user', content: '' },
    { role: 'assistant', content: '' }
  ]), 1);
  const empty = serializeArchive(collection);
  assert.equal(empty.ok, true);
  if (!empty.ok) return;
  const overhead = new TextEncoder().encode(empty.raw).byteLength;
  collection.chats[0].messages[0].content = 'x'.repeat(MAX_RECORD_BYTES - overhead);
  const exact = serializeArchive(collection);
  assert.equal(exact.ok, true);
  if (!exact.ok) return;
  assert.equal(new TextEncoder().encode(exact.raw).byteLength, MAX_RECORD_BYTES);

  collection.chats[0].messages[0].content += 'x';
  assert.deepEqual(serializeArchive(collection), { ok: false, error: 'oversized' });
  assert.deepEqual(parseArchive(`${exact.raw} `), { kind: 'invalid' });
});
