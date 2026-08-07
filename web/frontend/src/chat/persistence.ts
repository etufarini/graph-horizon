/*
 * Versioned conversation-storage boundary: owns the exact localStorage record,
 * UTF-8 bound, and exception-safe load/save/clear operations. System prompts,
 * UI text, lifecycle checkpoints, and alternative stores are excluded.
 */
import { validateTranscript } from './transcript.ts';
import type {
  ChatMessage,
  ConversationLoadResult,
  PersistenceWarning,
  TranscriptMessage
} from './types';

export const STORAGE_KEY = 'graph-horizon.conversation';
export const FORMAT_VERSION = 1;
export const MAX_RECORD_BYTES = 4_194_304;

function exactObject(value: unknown, keys: string[]): value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return false;
  }
  const actual = Object.keys(value);
  return actual.length === keys.length && keys.every(key => actual.includes(key));
}

function invalidRecord(storage: Storage): ConversationLoadResult {
  try {
    storage.removeItem(STORAGE_KEY);
    return { messages: [], warning: 'invalid-record' };
  } catch {
    return { messages: [], warning: 'unavailable' };
  }
}

export function loadConversation(): ConversationLoadResult {
  try {
    const storage = window.localStorage;
    const raw = storage.getItem(STORAGE_KEY);
    if (raw === null) {
      return { messages: [], warning: null };
    }
    if (new TextEncoder().encode(raw).byteLength > MAX_RECORD_BYTES) {
      return invalidRecord(storage);
    }
    let value: unknown;
    try {
      value = JSON.parse(raw);
    } catch {
      return invalidRecord(storage);
    }
    if (!exactObject(value, ['version', 'messages']) || value.version !== FORMAT_VERSION ||
        !Array.isArray(value.messages) ||
        !value.messages.every(message => exactObject(message, ['role', 'content']))) {
      return invalidRecord(storage);
    }
    const messages = validateTranscript(value.messages);
    return messages === null ? invalidRecord(storage) : { messages, warning: null };
  } catch {
    return { messages: [], warning: 'unavailable' };
  }
}

export function saveConversation(
  messages: ChatMessage[] | TranscriptMessage[]
): PersistenceWarning | null {
  const transcript = validateTranscript(messages);
  if (transcript === null) {
    return 'unavailable';
  }
  const raw = JSON.stringify({ version: FORMAT_VERSION, messages: transcript });
  if (new TextEncoder().encode(raw).byteLength > MAX_RECORD_BYTES) {
    return 'unavailable';
  }
  try {
    window.localStorage.setItem(STORAGE_KEY, raw);
    return null;
  } catch {
    return 'unavailable';
  }
}

export function clearConversation(): PersistenceWarning | null {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
    return null;
  } catch {
    return 'unavailable';
  }
}
