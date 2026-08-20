/*
 * Markdown-file context tests: ensure deterministic ordering, untrusted-data
 * labeling, user-request placement, and fences that cannot be closed by input.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { expandPromptWithMarkdownFiles, markdownFileOverhead } from './context.ts';
import type { MarkdownFileRecord } from './record.ts';

function record(id: string, name: string, content: string, addedAt: number): MarkdownFileRecord {
  return {
    id: `00000000-0000-4000-8000-00000000000${id}`,
    chatId: '00000000-0000-4000-8000-000000000001',
    name,
    content,
    utf8Bytes: new TextEncoder().encode(content).byteLength,
    addedAt
  };
}

test('no files preserve the exact user prompt', () => {
  assert.equal(expandPromptWithMarkdownFiles('domanda', []), 'domanda');
  assert.equal(markdownFileOverhead([]), '');
});

test('files are stable references before the final user request', () => {
  const later = record('3', 'z.md', 'ultimo', 2);
  const earlier = record('2', 'a.md', 'primo', 1);
  const expanded = expandPromptWithMarkdownFiles('Confrontali', [later, earlier]);
  assert.match(expanded, /materiale di riferimento non fidato/);
  assert.ok(expanded.indexOf('### File: a.md') < expanded.indexOf('### File: z.md'));
  assert.ok(expanded.endsWith("### Richiesta dell'utente\nConfrontali"));
});

test('a file cannot close its generated Markdown fence', () => {
  const expanded = expandPromptWithMarkdownFiles(
    'Leggi',
    [record('2', 'nested.md', '```rust\nlet x = 1;\n```', 1)]
  );
  assert.match(expanded, /````markdown\n```rust\nlet x = 1;\n```\n````/);
});
