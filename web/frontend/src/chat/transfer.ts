/*
 * Versioned chat-file transfer: owns pretty JSON and system-prompt schema while
 * delegating the shared complete-turn invariant to the pure transcript module.
 */
import { validateTranscript } from './transcript.ts';
import type { ChatMessage, TranscriptMessage } from './types';

export interface ChatTransferPayload {
  systemPrompt: string;
  messages: TranscriptMessage[];
}

export type ChatParseError = 'invalid-json' | 'invalid-format';

export type ChatParseResult =
  | { ok: true; payload: ChatTransferPayload }
  | { ok: false; error: ChatParseError };

export function serializeChat(messages: ChatMessage[], systemPrompt: string): string {
  const file = {
    version: 2,
    systemPrompt,
    messages: messages.map(({ role, content, search }) =>
      search ? { role, content, search } : { role, content })
  };
  return JSON.stringify(file, null, 2);
}

export function parseChatFile(text: string): ChatParseResult {
  let root: unknown;
  try {
    root = JSON.parse(text);
  } catch {
    return { ok: false, error: 'invalid-json' };
  }

  if (typeof root !== 'object' || root === null || Array.isArray(root)) {
    return { ok: false, error: 'invalid-format' };
  }
  const file = root as Record<string, unknown>;
  if ((file.version !== 1 && file.version !== 2) || typeof file.systemPrompt !== 'string') {
    return { ok: false, error: 'invalid-format' };
  }
  if (file.version === 1 && Array.isArray(file.messages) &&
      file.messages.some(message => typeof message === 'object' && message !== null && 'search' in message)) {
    return { ok: false, error: 'invalid-format' };
  }
  const messages = validateTranscript(file.messages);
  if (messages === null) return { ok: false, error: 'invalid-format' };
  return { ok: true, payload: { systemPrompt: file.systemPrompt, messages } };
}
