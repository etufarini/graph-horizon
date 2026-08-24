/*
 * Provider-neutral browser search options.
 * Validates explicit category and calendar controls, then builds the strict
 * wire request without interpreting or rewriting the user's terms.
 */
import type { SearchReport, SearchSelection, SearchSource, WireSearch } from './types.ts';

const MAX_TIMESTAMP_BOUND_MS = 253_402_300_800_000;

export function defaultSearch(): SearchSelection {
  return { query: '', category: 'web', period: 'any', from: '', to: '' };
}

export function validSearch(selection: SearchSelection): boolean {
  if (Array.from(selection.query.trim()).length > 512) return false;
  if (selection.period !== 'custom') return true;
  const from = parseDate(selection.from);
  const to = parseDate(selection.to);
  return from !== null && to !== null && selection.from <= selection.to &&
    selection.from >= '1970-01-01' && selection.to < '9999-12-31';
}

export function wireSearch(
  terms: string,
  selection: SearchSelection,
  now = new Date()
): WireSearch | null {
  const query = terms.trim();
  if (!query || Array.from(query).length > 512 || !validSearch(selection)) return null;
  const reference = localDate(now);
  let published: WireSearch['published'] = null;
  if (selection.period === 'day') {
    published = interval(dayStart(now), addDays(dayStart(now), 1));
  } else if (selection.period === 'week') {
    published = interval(addDays(dayStart(now), -6), addDays(dayStart(now), 1));
  } else if (selection.period === 'month') {
    published = interval(addDays(dayStart(now), -29), addDays(dayStart(now), 1));
  } else if (selection.period === 'custom') {
    const inclusiveTo = parseDate(selection.to)!;
    published = interval(parseDate(selection.from)!, addDays(inclusiveTo, 1));
  }
  return {
    terms: query,
    category: selection.category,
    language: browserLanguage(),
    reference_date: reference,
    published
  };
}

export function parseSearchReport(value: unknown): SearchReport | null {
  if (!record(value) || !exact(value, ['query', 'category', 'reference_date', 'published', 'provider', 'sources']) ||
      typeof value.query !== 'string' || !value.query.trim() || Array.from(value.query).length > 512 ||
      (value.category !== 'web' && value.category !== 'news') ||
      typeof value.reference_date !== 'string' || parseDate(value.reference_date) === null ||
      typeof value.provider !== 'string' || !value.provider || value.provider.length > 253 ||
      !Array.isArray(value.sources) || value.sources.length < 1 || value.sources.length > 5) {
    return null;
  }
  const published = parsePublished(value.published);
  if (value.published !== null && published === null) return null;
  const sources = value.sources.map(parseSource);
  if (sources.some(source => source === null) ||
      sources.some((source, index) => source!.id !== `S${index + 1}`)) return null;
  return {
    query: value.query,
    category: value.category,
    referenceDate: value.reference_date,
    published,
    provider: value.provider,
    sources: sources as SearchSource[]
  };
}

export function parseStoredSearch(value: unknown): SearchReport | null {
  if (!record(value) || !exact(value, ['query', 'category', 'referenceDate', 'published', 'provider', 'sources']) ||
      typeof value.query !== 'string' || !value.query.trim() || Array.from(value.query).length > 512 ||
      (value.category !== 'web' && value.category !== 'news') ||
      typeof value.referenceDate !== 'string' || parseDate(value.referenceDate) === null ||
      typeof value.provider !== 'string' || !value.provider || value.provider.length > 253 ||
      !Array.isArray(value.sources) || value.sources.length < 1 || value.sources.length > 5) return null;
  const published = parseStoredPublished(value.published);
  if (value.published !== null && published === null) return null;
  const sources = value.sources.map(parseStoredSource);
  if (sources.some(source => source === null) ||
      sources.some((source, index) => source!.id !== `S${index + 1}`)) return null;
  return {
    query: value.query,
    category: value.category,
    referenceDate: value.referenceDate,
    published,
    provider: value.provider,
    sources: sources as SearchSource[]
  };
}

function parsePublished(value: unknown): SearchReport['published'] {
  if (value === null) return null;
  if (!record(value) || !exact(value, ['from_ms', 'to_ms']) ||
      !safeTime(value.from_ms) || !safeTime(value.to_ms) || value.from_ms >= value.to_ms) return null;
  return { fromMs: value.from_ms, toMs: value.to_ms };
}

function parseStoredPublished(value: unknown): SearchReport['published'] {
  if (value === null) return null;
  if (!record(value) || !exact(value, ['fromMs', 'toMs']) ||
      !safeTime(value.fromMs) || !safeTime(value.toMs) || value.fromMs >= value.toMs) return null;
  return { fromMs: value.fromMs, toMs: value.toMs };
}

function parseSource(value: unknown): SearchSource | null {
  if (!record(value) || !exact(value, ['id', 'title', 'url', 'publisher', 'published_at_ms']) ||
      typeof value.id !== 'string' || !/^S[1-5]$/.test(value.id) ||
      typeof value.title !== 'string' || !value.title || Array.from(value.title).length > 160 ||
      !(value.publisher === null || (typeof value.publisher === 'string' &&
        Array.from(value.publisher).length <= 100)) ||
      !(value.published_at_ms === null || safeTime(value.published_at_ms)) || !safeUrl(value.url)) return null;
  return {
    id: value.id,
    title: value.title,
    url: value.url,
    publisher: value.publisher,
    publishedAtMs: value.published_at_ms
  };
}

function parseStoredSource(value: unknown): SearchSource | null {
  if (!record(value) || !exact(value, ['id', 'title', 'url', 'publisher', 'publishedAtMs']) ||
      typeof value.id !== 'string' || !/^S[1-5]$/.test(value.id) ||
      typeof value.title !== 'string' || !value.title || Array.from(value.title).length > 160 ||
      !(value.publisher === null || (typeof value.publisher === 'string' &&
        Array.from(value.publisher).length <= 100)) ||
      !(value.publishedAtMs === null || safeTime(value.publishedAtMs)) || !safeUrl(value.url)) return null;
  return value as unknown as SearchSource;
}

function safeTime(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0 &&
    (value as number) <= MAX_TIMESTAMP_BOUND_MS;
}

function safeUrl(value: unknown): value is string {
  if (typeof value !== 'string' || Array.from(value).length > 2048) return false;
  try {
    const url = new URL(value);
    return (url.protocol === 'https:' || url.protocol === 'http:') && !!url.hostname &&
      !url.username && !url.password;
  } catch {
    return false;
  }
}

function record(value: unknown): value is Record<string, any> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function exact(value: Record<string, unknown>, keys: string[]): boolean {
  const actual = Object.keys(value);
  return actual.length === keys.length && keys.every(key => actual.includes(key));
}

function interval(from: Date, to: Date): NonNullable<WireSearch['published']> {
  return { from_ms: from.getTime(), to_ms: to.getTime() };
}

function dayStart(date: Date): Date {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate());
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
  const date = new Date(2000, Number(match[2]) - 1, Number(match[3]));
  date.setFullYear(Number(match[1]));
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
