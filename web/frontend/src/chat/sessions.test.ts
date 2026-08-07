/*
 * Deterministic acceptance tests for pure chat-collection transformations.
 * Clock and UUID sources are controlled in memory; persistence, transport,
 * Svelte stores, and rendering are excluded.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  activeChat,
  appendChat,
  createCollection,
  deleteChat,
  MAX_TITLE_CODE_POINTS,
  newChat,
  orderedChats,
  renameChat,
  replaceActiveTranscript,
  selectChat
} from './sessions.ts';
import { hydrateTranscript } from './transcript.ts';

const ids = [
  '00000000-0000-4000-8000-000000000001',
  '00000000-0000-4000-8000-000000000002',
  '00000000-0000-4000-8000-000000000003',
  '00000000-0000-4000-8000-000000000004'
];
const source = (...values: string[]) => {
  let index = 0;
  return () => values[index++];
};
const turn = (user = 'ciao', assistant = 'risposta') => hydrateTranscript([
  { role: 'user', content: user },
  { role: 'assistant', content: assistant }
]);

test('constructor creates exactly one empty active chat', () => {
  const collection = createCollection(10, source(ids[0]));
  assert.deepEqual(collection, {
    activeChatId: ids[0],
    chats: [{ id: ids[0], title: 'Nuova chat', messages: [], updatedAt: 10 }]
  });
});

test('new chat is a no-op for an empty active chat and retries UUID collisions', () => {
  const initial = createCollection(10, source(ids[0]));
  assert.equal(newChat(initial, 20, source(ids[1])), initial);

  const stable = replaceActiveTranscript(initial, turn(), 15);
  const created = newChat(stable, 20, source('invalid', ids[0], ids[1]));
  assert.equal(created.activeChatId, ids[1]);
  assert.equal(created.chats.length, 2);
  assert.deepEqual(activeChat(created), {
    id: ids[1], title: 'Nuova chat', messages: [], updatedAt: 20
  });
});

test('stable replacement derives at most 48 Unicode code points once', () => {
  const initial = createCollection(1, source(ids[0]));
  const user = `  ${'🧠'.repeat(47)}   fine   ignorato  `;
  const messages = turn(user, '[THINK]π[/THINK]');
  const stable = replaceActiveTranscript(initial, messages, 2);
  assert.equal(Array.from(activeChat(stable).title).length, 48);
  assert.equal(activeChat(stable).title, `${'🧠'.repeat(47)} `);
  assert.equal(activeChat(stable).updatedAt, 2);
  assert.equal(activeChat(stable).messages, messages);

  const next = replaceActiveTranscript(stable, turn('titolo diverso'), 3);
  assert.equal(activeChat(next).title, activeChat(stable).title);
  const blank = replaceActiveTranscript(initial, turn(' \n\t '), 4);
  assert.equal(activeChat(blank).title, 'Nuova chat');
});

test('ordering is descending timestamp with ascending ID tie-break', () => {
  let collection = createCollection(10, source(ids[2]));
  collection = appendChat(collection, turn('second'), 30, source(ids[1]));
  collection = appendChat(collection, turn('third'), 30, source(ids[0]));
  assert.deepEqual(orderedChats(collection).map(chat => chat.id), [ids[0], ids[1], ids[2]]);
  assert.deepEqual(collection.chats.map(chat => chat.id), [ids[2], ids[1], ids[0]]);
});

test('selection and rename preserve timestamps and reject invalid titles', () => {
  let collection = createCollection(10, source(ids[0]));
  collection = appendChat(collection, turn(), 20, source(ids[1]));
  const selected = selectChat(collection, ids[0]);
  assert.equal(selected.activeChatId, ids[0]);
  assert.deepEqual(selected.chats.map(chat => chat.updatedAt), [10, 20]);
  assert.equal(selectChat(selected, ids[0]), selected);
  assert.equal(selectChat(selected, ids[3]), selected);

  const renamed = renameChat(selected, ids[0], '  titolo   interno  ');
  assert.equal(renamed.chats[0].title, 'titolo   interno');
  assert.equal(renamed.chats[0].updatedAt, 10);
  assert.equal(renameChat(renamed, ids[0], '   '), renamed);
  assert.equal(renameChat(renamed, ids[0], '🧠'.repeat(MAX_TITLE_CODE_POINTS + 1)), renamed);
  assert.equal(renameChat(renamed, ids[3], 'valido'), renamed);
  assert.equal(renameChat(renamed, ids[0], '🧠'.repeat(MAX_TITLE_CODE_POINTS)).chats[0].title,
    '🧠'.repeat(MAX_TITLE_CODE_POINTS));
});

test('deletion preserves or replaces active selection deterministically', () => {
  let collection = createCollection(10, source(ids[0]));
  collection = appendChat(collection, turn('newer'), 30, source(ids[1]));
  collection = appendChat(collection, turn('tie'), 30, source(ids[2]));

  const nonActive = deleteChat(collection, ids[0]);
  assert.equal(nonActive.activeChatId, ids[2]);
  assert.equal(nonActive.chats.length, 2);
  const active = deleteChat(nonActive, ids[2]);
  assert.equal(active.activeChatId, ids[1]);
  assert.equal(deleteChat(active, ids[3]), active);

  const replacement = deleteChat(active, ids[1], 40, source(ids[3]));
  assert.deepEqual(replacement, {
    activeChatId: ids[3],
    chats: [{ id: ids[3], title: 'Nuova chat', messages: [], updatedAt: 40 }]
  });
});

test('invalid stable transcripts and imported chat inputs are rejected', () => {
  const collection = createCollection(1, source(ids[0]));
  const odd = hydrateTranscript([{ role: 'user', content: 'odd' }]);
  assert.equal(replaceActiveTranscript(collection, odd, 2), collection);
  assert.equal(appendChat(collection, odd, 2, source(ids[1])), collection);

  const imported = appendChat(collection, turn('  import title  '), 3, source(ids[1]));
  assert.equal(imported.activeChatId, ids[1]);
  assert.equal(activeChat(imported).title, 'import title');
});
