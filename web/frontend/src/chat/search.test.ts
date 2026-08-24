/* Provider-neutral search request tests use fixed local calendar values only. */
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

test('calendar presets produce exact half-open intervals', () => {
  for (const [period, from, to] of [
    ['day', '2026-08-24', '2026-08-25'],
    ['week', '2026-08-18', '2026-08-25'],
    ['month', '2026-07-26', '2026-08-25']
  ] as const) {
    const selection = { ...defaultSearch(), period, category: 'news' as const };
    assert.deepEqual(wireSearch('query', selection, now)?.published, { from, to });
  }
});

test('custom dates are inclusive in the UI and exclusive on the wire', () => {
  const selection = {
    ...defaultSearch(),
    category: 'news' as const,
    period: 'custom' as const,
    from: '2026-08-14',
    to: '2026-08-14'
  };
  assert.equal(validSearch(selection), true);
  assert.deepEqual(wireSearch('query', selection, now)?.published, {
    from: '2026-08-14',
    to: '2026-08-15'
  });

  assert.equal(validSearch({ ...selection, from: '2026-02-29' }), false);
  assert.equal(validSearch({ ...selection, from: '2026-08-15' }), false);
});
