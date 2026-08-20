/*
 * Browser Markdown-file contract: define the exact durable record and turn an
 * untrusted File into validated UTF-8 text. Storage, chat ownership, prompt
 * projection, and UI behavior remain outside this module.
 */

export const MAX_FILES_PER_CHAT = 10;
export const MAX_FILE_BYTES = 1024 * 1024;
export const MAX_CHAT_FILE_BYTES = 2 * 1024 * 1024;
export const MAX_FILE_NAME_CODE_POINTS = 255;

export interface MarkdownFileRecord {
  id: string;
  chatId: string;
  name: string;
  content: string;
  utf8Bytes: number;
  addedAt: number;
}

export type MarkdownFileError =
  | 'extension'
  | 'name'
  | 'empty'
  | 'encoding'
  | 'oversized'
  | 'unreadable';

export type MarkdownFileResult =
  | { ok: true; record: MarkdownFileRecord }
  | { ok: false; error: MarkdownFileError };

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const FORBIDDEN_NAME = /[\u0000-\u001f\u007f-\u009f/\\]/u;

export async function readMarkdownFile(
  file: File,
  chatId: string,
  addedAt = Date.now(),
  idSource: () => string = () => globalThis.crypto.randomUUID()
): Promise<MarkdownFileResult> {
  if (!validName(file.name)) return { ok: false, error: nameError(file.name) };
  if (file.size === 0) return { ok: false, error: 'empty' };
  if (file.size > MAX_FILE_BYTES) return { ok: false, error: 'oversized' };
  if (!UUID.test(chatId) || !Number.isSafeInteger(addedAt) || addedAt < 0) {
    return { ok: false, error: 'unreadable' };
  }

  let bytes: ArrayBuffer;
  try {
    bytes = await file.arrayBuffer();
  } catch {
    return { ok: false, error: 'unreadable' };
  }
  if (bytes.byteLength === 0) return { ok: false, error: 'empty' };
  if (bytes.byteLength > MAX_FILE_BYTES) return { ok: false, error: 'oversized' };

  let content: string;
  try {
    // Fatal decoding keeps persisted, previewed, downloaded, and prompted text identical.
    content = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes);
  } catch {
    return { ok: false, error: 'encoding' };
  }
  if (!content || content.includes('\0')) return { ok: false, error: 'empty' };

  const id = nextId(idSource);
  return {
    ok: true,
    record: { id, chatId, name: file.name, content, utf8Bytes: bytes.byteLength, addedAt }
  };
}

export function parseMarkdownFileRecord(value: unknown): MarkdownFileRecord | null {
  if (!isRecord(value) || !exact(value, ['id', 'chatId', 'name', 'content', 'utf8Bytes', 'addedAt']) ||
      typeof value.id !== 'string' || !UUID.test(value.id) ||
      typeof value.chatId !== 'string' || !UUID.test(value.chatId) ||
      typeof value.name !== 'string' || !validName(value.name) ||
      typeof value.content !== 'string' || !value.content || value.content.includes('\0') ||
      !Number.isSafeInteger(value.utf8Bytes) || (value.utf8Bytes as number) <= 0 ||
      (value.utf8Bytes as number) > MAX_FILE_BYTES ||
      !Number.isSafeInteger(value.addedAt) || (value.addedAt as number) < 0) {
    return null;
  }
  const utf8Bytes = new TextEncoder().encode(value.content).byteLength;
  return utf8Bytes === value.utf8Bytes ? value as unknown as MarkdownFileRecord : null;
}

function validName(name: string): boolean {
  const length = Array.from(name).length;
  return name === name.trim() && length > 3 && length <= MAX_FILE_NAME_CODE_POINTS &&
    !FORBIDDEN_NAME.test(name) && name.toLocaleLowerCase('en-US').endsWith('.md');
}

function nameError(name: string): MarkdownFileError {
  return name.toLocaleLowerCase('en-US').endsWith('.md') ? 'name' : 'extension';
}

function nextId(idSource: () => string): string {
  let id = idSource();
  while (!UUID.test(id)) id = idSource();
  return id;
}

function exact(value: Record<string, unknown>, keys: string[]): boolean {
  const actual = Object.keys(value);
  return actual.length === keys.length && keys.every(key => actual.includes(key));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
