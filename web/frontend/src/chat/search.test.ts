/* Provider-neutral search request tests preserve exact terms without date UI. */
import test from 'node:test';
import assert from 'node:assert/strict';

import { defaultSearch, validSearch, wireSearch } from './search.ts';

const now = new Date(2026, 7, 24, 12);

test('any-time Web search preserves arbitrary-language terms', () => {
  const terms = 'noticias de hace diez días error class latest';
  const request = wireSearch(terms, defaultSearch(), now)!;
  assert.equal(request.terms, terms);
  assert.equal(request.category, 'web');
  assert.equal(request.reference_date, '2026-08-24');
  assert.equal(request.published, null);
  assert.ok(request.language.length >= 2);
});

test('new browser searches never add a publication range', () => {
  const terms = 'Notizie Tuscania (VT, Italia) ultima settimana';
  for (const category of ['web', 'news'] as const) {
    const selection = { ...defaultSearch(), category };
    const request = wireSearch(terms, selection, now)!;
    assert.equal(request.terms, terms);
    assert.equal(request.category, category);
    assert.equal(request.published, null);
  }
});

test('selection validation only bounds the optional explicit query', () => {
  assert.equal(validSearch(defaultSearch()), true);
  assert.equal(validSearch({ ...defaultSearch(), query: 'x'.repeat(512) }), true);
  assert.equal(validSearch({ ...defaultSearch(), query: 'x'.repeat(513) }), false);
});
