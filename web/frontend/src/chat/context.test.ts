/*
 * Pure Web-capacity tests covering private payload validation, Unicode
 * estimation, inclusive prompt-budget boundaries, and overflow-safe arithmetic.
 * Browser rendering and transport are intentionally excluded.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { admitMessages, contextUsage, parseRuntimeContext } from './context.ts';
import type { RuntimeContext, WireMessage } from './types.ts';

const context: RuntimeContext = {
  contextLimit: 2000,
  safePromptBudget: 1800,
  search: { provider: 'search.example', maxQueryCharacters: 512, maxContextCharacters: 2800 }
};

const capability = {
  provider: 'search.example',
  max_query_characters: 512,
  max_context_characters: 2800
};

function user(content: string): WireMessage[] {
  return [{ role: 'user', content }];
}

test('context accepts positive safe capacities and ignores extra fields', () => {
  assert.deepEqual(
    parseRuntimeContext({
      context_limit: 8192,
      search: capability,
      model_path: 'ignored'
    }),
    {
      ok: true,
      context: {
        contextLimit: 8192,
        safePromptBudget: 7372,
        search: { provider: 'search.example', maxQueryCharacters: 512, maxContextCharacters: 2800 }
      }
    }
  );
  for (const context_limit of [0, 1, -1, 1.5, Number.MAX_SAFE_INTEGER + 1, '4096', null]) {
    assert.deepEqual(parseRuntimeContext({ context_limit, search: capability }), {
      ok: false,
      error: 'unavailable'
    });
  }
  for (const search of [0, null, {}, { ...capability, provider: null }, { ...capability, provider: '' }, { ...capability, provider: 1 }, { ...capability, max_context_characters: 0 }]) {
    assert.deepEqual(parseRuntimeContext({ context_limit: 8192, search }), {
      ok: false,
      error: 'unavailable'
    });
  }
  for (const payload of [null, [], {}, { context_limit: null }, { context_limit: 8192 }]) {
    assert.deepEqual(parseRuntimeContext(payload), { ok: false, error: 'unavailable' });
  }
});

test('search reserve participates in the same Unicode estimate', () => {
  assert.equal(contextUsage(user('xxxx'), context, 4).estimatedTokens, 2);
  assert.equal(admitMessages(user('x'.repeat(1799 * 4)), context, 4).ok, true);
  assert.equal(admitMessages(user('x'.repeat(1800 * 4)), context, 4).ok, false);
  assert.equal(contextUsage([], context, -1).estimatedTokens, Number.MAX_SAFE_INTEGER);
});

test('Unicode code points are summed before one floor division', () => {
  const messages: WireMessage[] = [
    { role: 'system', content: 'abc' },
    { role: 'user', content: '😀éx' }
  ];
  assert.equal(contextUsage(messages, context).estimatedTokens, 1);
});

test('admission accepts equality and rejects one token over', () => {
  assert.equal(admitMessages(user('x'.repeat(1800 * 4)), context).ok, true);
  const rejected = admitMessages(user('x'.repeat(1801 * 4)), context);
  assert.equal(rejected.ok, false);
  if (!rejected.ok) {
    assert.equal(rejected.estimatedTokens, 1801);
    assert.equal(rejected.safePromptBudget, 1800);
    assert.deepEqual(rejected.usage, {
      estimatedTokens: 1801,
      contextLimit: 2000,
      percent: 91,
      progress: 91
    });
  }
});

test('usage includes its context denominator and clamps only graphical progress', () => {
  assert.deepEqual(contextUsage([], context), {
    estimatedTokens: 0,
    contextLimit: 2000,
    percent: 0,
    progress: 0
  });
  assert.deepEqual(contextUsage(user('x'.repeat(4)), context), {
    estimatedTokens: 1,
    contextLimit: 2000,
    percent: 1,
    progress: 1
  });
  const over = contextUsage(user('x'.repeat(2001 * 4)), context);
  assert.equal(over.percent, 101);
  assert.equal(over.progress, 100);
});

test('maximum safe capacity uses exact quotient arithmetic', () => {
  assert.deepEqual(
    parseRuntimeContext({
      context_limit: Number.MAX_SAFE_INTEGER,
      search: capability
    }),
    {
      ok: true,
      context: {
        contextLimit: Number.MAX_SAFE_INTEGER,
        safePromptBudget: 8106479329266891,
        search: { provider: 'search.example', maxQueryCharacters: 512, maxContextCharacters: 2800 }
      }
    }
  );
});
