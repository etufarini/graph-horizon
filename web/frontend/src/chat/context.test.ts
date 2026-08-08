/*
 * Pure Web-capacity tests covering `/props` validation and equality, Unicode
 * estimation, inclusive prompt-budget boundaries, and overflow-safe arithmetic.
 * Browser rendering and transport are intentionally excluded.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { admitMessages, contextUsage, parseRuntimeContext } from './context.ts';
import type { RuntimeContext, WireMessage } from './types.ts';

const context: RuntimeContext = {
  contextLimit: 2000,
  safePromptBudget: 1800
};

function user(content: string): WireMessage[] {
  return [{ role: 'user', content }];
}

test('properties accept equal positive safe capacities and ignore extra fields', () => {
  assert.deepEqual(
    parseRuntimeContext({
      default_generation_settings: { n_ctx: 8192, max_tokens: 8192, ignored: true },
      model_path: 'ignored'
    }),
    {
      ok: true,
      context: { contextLimit: 8192, safePromptBudget: 7372 }
    }
  );
  for (const n_ctx of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1, '4096', null]) {
    assert.deepEqual(parseRuntimeContext({ default_generation_settings: { n_ctx, max_tokens: n_ctx } }), {
      ok: false,
      error: 'unavailable'
    });
  }
  for (const max_tokens of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1, '4096', null]) {
    assert.deepEqual(parseRuntimeContext({ default_generation_settings: { n_ctx: max_tokens, max_tokens } }), {
      ok: false,
      error: 'unavailable'
    });
  }
  for (const settings of [
    { n_ctx: 8192 },
    { max_tokens: 8192 },
    { n_ctx: 8192, max_tokens: 4096 },
    { n_ctx: 1, max_tokens: 1 }
  ]) {
    assert.deepEqual(parseRuntimeContext({ default_generation_settings: settings }), {
      ok: false,
      error: 'unavailable'
    });
  }
  for (const payload of [null, [], {}, { default_generation_settings: null }, { default_generation_settings: [] }]) {
    assert.deepEqual(parseRuntimeContext(payload), { ok: false, error: 'unavailable' });
  }
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
    assert.deepEqual(rejected.usage, { estimatedTokens: 1801, percent: 91, progress: 91 });
  }
});

test('percentage uses ceiling while graphical progress clamps to 100', () => {
  assert.deepEqual(contextUsage([], context), { estimatedTokens: 0, percent: 0, progress: 0 });
  assert.deepEqual(contextUsage(user('x'.repeat(4)), context), {
    estimatedTokens: 1,
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
      default_generation_settings: {
        n_ctx: Number.MAX_SAFE_INTEGER,
        max_tokens: Number.MAX_SAFE_INTEGER
      }
    }),
    {
      ok: true,
      context: {
        contextLimit: Number.MAX_SAFE_INTEGER,
        safePromptBudget: 8106479329266891
      }
    }
  );
});
