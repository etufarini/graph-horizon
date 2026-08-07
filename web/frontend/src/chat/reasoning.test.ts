/*
 * This Node test suite verifies the display-only Reasoning parser. It does not
 * test Svelte rendering, chat persistence, or wire behavior.
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import { splitReasoning } from './reasoning.ts';

test('empty and whitespace-only streams remain pending', () => {
  assert.deepEqual(splitReasoning('', true), { answer: '', pending: true });
  assert.deepEqual(splitReasoning(' \n\t', true), { answer: '', pending: true });
});

test('empty and whitespace-only completed content remains an ordinary answer', () => {
  assert.deepEqual(splitReasoning('', false), { answer: '', pending: false });
  assert.deepEqual(splitReasoning(' \n\t', false), { answer: ' \n\t', pending: false });
});

test('every strict opening-marker prefix remains pending while streaming', () => {
  const marker = '[THINK]';
  for (let length = 0; length < marker.length; length += 1) {
    assert.deepEqual(splitReasoning(marker.slice(0, length), true), {
      answer: '',
      pending: true,
    });
  }
});

test('leading whitespace before a strict prefix remains pending', () => {
  assert.deepEqual(splitReasoning('\n  [TH', true), { answer: '', pending: true });
});

test('a contradicted streaming prefix immediately becomes an ordinary answer', () => {
  assert.deepEqual(splitReasoning('  [THX', true), {
    answer: '  [THX',
    pending: false,
  });
});

test('an incomplete prefix after completion is an ordinary answer', () => {
  assert.deepEqual(splitReasoning('[THIN', false), {
    answer: '[THIN',
    pending: false,
  });
});

test('an ordinary response is unchanged', () => {
  assert.deepEqual(splitReasoning('Risposta ordinaria', true), {
    answer: 'Risposta ordinaria',
    pending: false,
  });
});

test('exact complete markers produce empty THINK and empty answer', () => {
  assert.deepEqual(splitReasoning('[THINK][/THINK]', false), {
    thinking: '',
    answer: '',
    pending: false,
  });
});

test('leading whitespace is omitted only from a recognized Reasoning view', () => {
  const raw = ' \n[THINK]  ragiona \n[/THINK]  rispondi ';
  assert.deepEqual(splitReasoning(raw, false), {
    thinking: '  ragiona \n',
    answer: '  rispondi ',
    pending: false,
  });
  assert.equal(raw, ' \n[THINK]  ragiona \n[/THINK]  rispondi ');
});

test('an opening marker without a close exposes THINK while streaming', () => {
  assert.deepEqual(splitReasoning('[THINK]passo', true), {
    thinking: 'passo',
    answer: '',
    pending: false,
  });
});

test('an opening marker without a close exposes THINK after completion', () => {
  assert.deepEqual(splitReasoning('[THINK]passo', false), {
    thinking: 'passo',
    answer: '',
    pending: false,
    incomplete: true,
  });
});

test('lowercase and mixed-case near-matches remain ordinary answers', () => {
  for (const raw of ['[think]x[/think]y', '[Think]x[/THINK]y']) {
    assert.deepEqual(splitReasoning(raw, true), { answer: raw, pending: false });
  }
});

test('an opening marker after non-whitespace text remains literal', () => {
  const raw = 'intro [THINK]x[/THINK]y';
  assert.deepEqual(splitReasoning(raw, false), { answer: raw, pending: false });
});

test('a closing marker without an opening marker remains literal', () => {
  const raw = '[/THINK]risposta';
  assert.deepEqual(splitReasoning(raw, false), { answer: raw, pending: false });
});

test('an extra opening marker remains inside THINK', () => {
  assert.deepEqual(splitReasoning('[THINK]a[THINK]b[/THINK]c', false), {
    thinking: 'a[THINK]b',
    answer: 'c',
    pending: false,
  });
});

test('markers after the first close remain literal answer content', () => {
  assert.deepEqual(splitReasoning('[THINK]a[/THINK]b[/THINK][THINK]c[/THINK]', false), {
    thinking: 'a',
    answer: 'b[/THINK][THINK]c[/THINK]',
    pending: false,
  });
});

test('newlines and Unicode are preserved exactly in both sections', () => {
  assert.deepEqual(splitReasoning('[THINK]\nπensa 🧠\n[/THINK]\nrisposta ✓\n', false), {
    thinking: '\nπensa 🧠\n',
    answer: '\nrisposta ✓\n',
    pending: false,
  });
});
