/*
 * Markdown-file contract tests: cover strict names, UTF-8 decoding, byte limits,
 * exact durable records, and preservation of accepted text.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import {
  MAX_FILE_BYTES,
  parseMarkdownFileRecord,
  readMarkdownFile
} from './record.ts';

const CHAT_ID = '00000000-0000-4000-8000-000000000001';
const FILE_ID = '00000000-0000-4000-8000-000000000002';

function selected(name: string, bytes: Uint8Array): File {
  return {
    name,
    size: bytes.byteLength,
    arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
  } as File;
}

test('reads an exact UTF-8 Markdown record without trusting MIME metadata', async () => {
  const bytes = new TextEncoder().encode('# Titolo\n\nTesto');
  const result = await readMarkdownFile(selected('Specifica.MD', bytes), CHAT_ID, 7, () => FILE_ID);
  assert.deepEqual(result, {
    ok: true,
    record: {
      id: FILE_ID,
      chatId: CHAT_ID,
      name: 'Specifica.MD',
      content: '# Titolo\n\nTesto',
      utf8Bytes: bytes.byteLength,
      addedAt: 7
    }
  });
});

test('rejects wrong extensions, unsafe names, empty data, and invalid UTF-8', async () => {
  const valid = new TextEncoder().encode('x');
  assert.equal((await readMarkdownFile(selected('note.txt', valid), CHAT_ID)).ok, false);
  assert.equal((await readMarkdownFile(selected('../note.md', valid), CHAT_ID)).ok, false);
  assert.equal((await readMarkdownFile(selected(' note.md', valid), CHAT_ID)).ok, false);
  assert.equal((await readMarkdownFile(selected('empty.md', new Uint8Array()), CHAT_ID)).ok, false);
  const invalid = await readMarkdownFile(selected('bad.md', new Uint8Array([0xc3, 0x28])), CHAT_ID);
  assert.deepEqual(invalid, { ok: false, error: 'encoding' });
});

test('rejects files beyond the hard byte limit before reading', async () => {
  let read = false;
  const file = {
    name: 'large.md',
    size: MAX_FILE_BYTES + 1,
    arrayBuffer: async () => {
      read = true;
      return new ArrayBuffer(0);
    }
  } as File;
  assert.deepEqual(await readMarkdownFile(file, CHAT_ID), { ok: false, error: 'oversized' });
  assert.equal(read, false);
});

test('durable parser rejects extra fields and byte-count mismatches', () => {
  const record = {
    id: FILE_ID,
    chatId: CHAT_ID,
    name: 'note.md',
    content: 'é',
    utf8Bytes: 2,
    addedAt: 3
  };
  assert.deepEqual(parseMarkdownFileRecord(record), record);
  assert.equal(parseMarkdownFileRecord({ ...record, utf8Bytes: 1 }), null);
  assert.equal(parseMarkdownFileRecord({ ...record, extra: true }), null);
});
