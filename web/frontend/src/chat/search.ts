/*
 * Provider-neutral browser search options.
 * Validates explicit category and calendar controls, then builds the strict
 * wire request without interpreting or rewriting the user's terms.
 */
import type { SearchSelection, WireSearch } from './types.ts';

export function defaultSearch(): SearchSelection {
  return { category: 'web', period: 'any', from: '', to: '' };
}

export function validSearch(selection: SearchSelection): boolean {
  if (selection.period !== 'custom') return true;
  const from = parseDate(selection.from);
  const to = parseDate(selection.to);
  return from !== null && to !== null && selection.from <= selection.to;
}

export function wireSearch(
  terms: string,
  selection: SearchSelection,
  now = new Date()
): WireSearch | null {
  if (!terms || Array.from(terms).length > 512 || !validSearch(selection)) return null;
  const reference = localDate(now);
  let published: WireSearch['published'] = null;
  if (selection.period === 'day') {
    published = { from: reference, to: localDate(addDays(now, 1)) };
  } else if (selection.period === 'week') {
    published = { from: localDate(addDays(now, -6)), to: localDate(addDays(now, 1)) };
  } else if (selection.period === 'month') {
    published = { from: localDate(addDays(now, -29)), to: localDate(addDays(now, 1)) };
  } else if (selection.period === 'custom') {
    const inclusiveTo = parseDate(selection.to)!;
    published = { from: selection.from, to: localDate(addDays(inclusiveTo, 1)) };
  }
  return {
    terms,
    category: selection.category,
    language: browserLanguage(),
    reference_date: reference,
    published
  };
}

function browserLanguage(): string {
  const language = globalThis.navigator?.language || 'en-US';
  try {
    return Intl.getCanonicalLocales(language)[0] ?? 'en-US';
  } catch {
    return 'en-US';
  }
}

function parseDate(value: string): Date | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return null;
  const date = new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
  return localDate(date) === value ? date : null;
}

function addDays(date: Date, days: number): Date {
  // Local calendar arithmetic preserves the user's definition of each day
  // across daylight-saving transitions.
  return new Date(date.getFullYear(), date.getMonth(), date.getDate() + days);
}

function localDate(date: Date): string {
  const year = String(date.getFullYear()).padStart(4, '0');
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}
