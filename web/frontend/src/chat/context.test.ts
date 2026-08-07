/*
 * Node tests for the pure browser context model.
 * They cover properties validation, Unicode estimation, budget boundaries,
 * percentage clamping, and arithmetic rejection without browser fixtures.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { admitMessages, contextUsage, parseRuntimeContext } from './context.ts';
import type { RuntimeContext, WireMessage } from './types.ts';

const context: RuntimeContext = {
  contextLimit: 2000,
  maxTokens: 1024,
  safeTotalBudget: 1800
};

function user(content: string): WireMessage[] {
  return [{ role: 'user', content }];
}

test('properties accept positive safe capacity integers and ignore extra fields', () => {
  assert.deepEqual(
    parseRuntimeContext({
      default_generation_settings: { n_ctx: 8192, max_tokens: 4096, ignored: true },
      model_path: 'ignored'
    }),
    {
      ok: true,
      context: { contextLimit: 8192, maxTokens: 4096, safeTotalBudget: 7372 }
    }
  );
  for (const n_ctx of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1, '4096', null]) {
    assert.deepEqual(parseRuntimeContext({ default_generation_settings: { n_ctx, max_tokens: 1 } }), {
      ok: false,
      error: 'unavailable'
    });
  }
  for (const max_tokens of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1, '4096', null]) {
    assert.deepEqual(parseRuntimeContext({ default_generation_settings: { n_ctx: 8192, max_tokens } }), {
      ok: false,
      error: 'unavailable'
    });
  }
  assert.deepEqual(parseRuntimeContext({}), { ok: false, error: 'unavailable' });
});

test('reserve must leave prompt space', () => {
  assert.deepEqual(parseRuntimeContext({ default_generation_settings: { n_ctx: 4096, max_tokens: 4096 } }), {
    ok: false,
    error: 'no-prompt-space'
  });
});

test('Unicode code points are summed before one floor division', () => {
  const messages: WireMessage[] = [
    { role: 'system', content: 'abc' },
    { role: 'user', content: '😀éx' }
  ];
  assert.equal(contextUsage(messages, context).estimatedTokens, 1);
});

test('admission accepts equality and rejects one token over', () => {
  assert.equal(admitMessages(user('x'.repeat((1800 - 1024) * 4)), context).ok, true);
  const rejected = admitMessages(user('x'.repeat((1801 - 1024) * 4)), context);
  assert.equal(rejected.ok, false);
  if (!rejected.ok) {
    assert.equal(rejected.estimatedTokens, 777);
    assert.equal(rejected.safeTotalBudget, 1800);
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

test('unsafe required-budget arithmetic rejects', () => {
  const unsafeContext: RuntimeContext = {
    contextLimit: Number.MAX_SAFE_INTEGER,
    maxTokens: Number.MAX_SAFE_INTEGER - 1,
    safeTotalBudget: Number.MAX_SAFE_INTEGER
  };
  assert.equal(admitMessages(user('xxxxxxxx'), unsafeContext).ok, false);
});
