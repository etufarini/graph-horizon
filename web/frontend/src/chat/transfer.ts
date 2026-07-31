/*
 * Chat transfer format.
 * Single responsibility: serialize and validate imported/exported text-chat
 * transcripts. It is pure data code and rejects unsupported or incomplete turn
 * sequences before state replacement.
 */
import type { ChatMessage } from './types';

export interface ChatTransferPayload {
  systemPrompt: string;
  messages: { role: 'user' | 'assistant'; content: string }[];
}

export type ChatParseError = 'invalid-json' | 'invalid-format';

export type ChatParseResult =
  | { ok: true; payload: ChatTransferPayload }
  | { ok: false; error: ChatParseError };

export function serializeChat(messages: ChatMessage[], systemPrompt: string): string {
  const file = {
    version: 1,
    systemPrompt,
    messages: messages.map(message => ({
      role: message.role,
      content: message.content
    }))
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
  if (file.version !== 1 || typeof file.systemPrompt !== 'string' || !Array.isArray(file.messages)) {
    return { ok: false, error: 'invalid-format' };
  }

  const messages: ChatTransferPayload['messages'] = [];
  for (const entry of file.messages) {
    if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) {
      return { ok: false, error: 'invalid-format' };
    }
    const message = entry as Record<string, unknown>;
    if (message.role !== 'user' && message.role !== 'assistant') {
      return { ok: false, error: 'invalid-format' };
    }
    // Imported history must remain valid for the next request: complete turns
    // start with the user and alternate strictly through the assistant reply.
    const expectedRole = messages.length % 2 === 0 ? 'user' : 'assistant';
    if (message.role !== expectedRole) {
      return { ok: false, error: 'invalid-format' };
    }
    if (typeof message.content !== 'string') {
      return { ok: false, error: 'invalid-format' };
    }
    messages.push({ role: message.role, content: message.content });
  }
  if (messages.length % 2 !== 0) {
    return { ok: false, error: 'invalid-format' };
  }

  return { ok: true, payload: { systemPrompt: file.systemPrompt, messages } };
}
