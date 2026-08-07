/*
 * Private chat-archive format boundary: owns exact version schemas, strict
 * object validation, the UTF-8 size limit, and version-1 migration. Browser
 * storage I/O and application mutations remain outside this module.
 */
import { createCollection, replaceActiveTranscript } from './sessions.ts';
import { hydrateTranscript, validateTranscript } from './transcript.ts';
import type { ChatArchiveRecord, ChatCollection, TranscriptMessage } from './types.ts';

export const STORAGE_KEY = 'graph-horizon.conversation';
export const FORMAT_VERSION = 2;
export const MAX_RECORD_BYTES = 4_194_304;

export type ArchiveParseResult =
  | { kind: 'current'; collection: ChatCollection }
  | { kind: 'legacy'; collection: ChatCollection }
  | { kind: 'invalid' };

export type ArchiveSerializeResult =
  | { ok: true; raw: string }
  | { ok: false; error: 'oversized' };

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export function parseArchive(
  raw: string,
  migratedAt = Date.now(),
  idSource: () => string = () => globalThis.crypto.randomUUID()
): ArchiveParseResult {
  if (bytes(raw) > MAX_RECORD_BYTES) {
    return { kind: 'invalid' };
  }
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return { kind: 'invalid' };
  }
  if (!isObject(value)) {
    return { kind: 'invalid' };
  }
  if (value.version === FORMAT_VERSION) {
    const collection = parseCurrent(value);
    return collection ? { kind: 'current', collection } : { kind: 'invalid' };
  }
  if (value.version === 1) {
    const transcript = parseLegacy(value);
    if (transcript) {
      const collection = createCollection(migratedAt, idSource);
      return {
        kind: 'legacy',
        collection: replaceActiveTranscript(collection, hydrateTranscript(transcript), migratedAt)
      };
    }
  }
  return { kind: 'invalid' };
}

export function serializeArchive(collection: ChatCollection): ArchiveSerializeResult {
  const record: ChatArchiveRecord = {
    version: FORMAT_VERSION,
    activeChatId: collection.activeChatId,
    chats: collection.chats.map(chat => ({
      id: chat.id,
      title: chat.title,
      messages: chat.messages.map(({ role, content }) => ({ role, content })),
      updatedAt: chat.updatedAt
    }))
  };
  const raw = JSON.stringify(record);
  return bytes(raw) <= MAX_RECORD_BYTES
    ? { ok: true, raw }
    : { ok: false, error: 'oversized' };
}

function parseCurrent(value: Record<string, unknown>): ChatCollection | null {
  if (!exact(value, ['version', 'activeChatId', 'chats']) ||
      typeof value.activeChatId !== 'string' || !UUID.test(value.activeChatId) ||
      !Array.isArray(value.chats) || value.chats.length === 0) {
    return null;
  }
  const ids = new Set<string>();
  const chats = [];
  for (const entry of value.chats) {
    if (!isObject(entry) || !exact(entry, ['id', 'title', 'messages', 'updatedAt']) ||
        typeof entry.id !== 'string' || !UUID.test(entry.id) || ids.has(entry.id) ||
        typeof entry.title !== 'string' || !validTitle(entry.title) ||
        !Number.isSafeInteger(entry.updatedAt) || (entry.updatedAt as number) < 0) {
      return null;
    }
    const messages = parseMessages(entry.messages);
    if (!messages) {
      return null;
    }
    ids.add(entry.id);
    chats.push({
      id: entry.id,
      title: entry.title,
      messages: hydrateTranscript(messages),
      updatedAt: entry.updatedAt as number
    });
  }
  return ids.has(value.activeChatId)
    ? { activeChatId: value.activeChatId, chats }
    : null;
}

function parseLegacy(value: Record<string, unknown>): TranscriptMessage[] | null {
  return exact(value, ['version', 'messages']) ? parseMessages(value.messages) : null;
}

function parseMessages(value: unknown): TranscriptMessage[] | null {
  if (!Array.isArray(value) || !value.every(message =>
    isObject(message) && exact(message, ['role', 'content']))) {
    return null;
  }
  return validateTranscript(value);
}

function validTitle(title: string): boolean {
  const length = Array.from(title).length;
  return length >= 1 && length <= 80;
}

function exact(value: Record<string, unknown>, keys: string[]): boolean {
  const actual = Object.keys(value);
  return actual.length === keys.length && keys.every(key => actual.includes(key));
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function bytes(value: string): number {
  return new TextEncoder().encode(value).byteLength;
}
