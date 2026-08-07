/*
 * Pure transcript boundary: validates complete alternating turns, hydrates
 * runtime IDs, and transforms chat/wire messages. Lifecycle, transport,
 * browser storage, file schemas, and presentation remain outside this module.
 */
import type { ChatMessage, TranscriptMessage, WireMessage } from './types';

let fallbackId = 0;

export function validateTranscript(value: unknown): TranscriptMessage[] | null {
  if (!Array.isArray(value) || value.length % 2 !== 0) {
    return null;
  }
  const messages: TranscriptMessage[] = [];
  for (const [index, entry] of value.entries()) {
    if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) {
      return null;
    }
    const message = entry as Record<string, unknown>;
    const expected = index % 2 === 0 ? 'user' : 'assistant';
    if (message.role !== expected || typeof message.content !== 'string') {
      return null;
    }
    messages.push({ role: expected, content: message.content });
  }
  return messages;
}

export function hydrateTranscript(messages: TranscriptMessage[]): ChatMessage[] {
  return messages.map(message => {
    const random = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${fallbackId++}`;
    return { ...message, id: `${message.role}-${random}` };
  });
}

export function wireMessages(
  messages: ChatMessage[],
  systemPrompt: string,
  draft = ''
): WireMessage[] {
  const wire: WireMessage[] = messages.map(({ role, content }) => ({ role, content }));
  const user = draft.trim();
  if (user) {
    wire.push({ role: 'user', content: user });
  }
  const system = systemPrompt.trim();
  if (system) {
    wire.unshift({ role: 'system', content: system });
  }
  return wire;
}

export function appendAssistant(messages: ChatMessage[], content: string): ChatMessage[] {
  const last = messages.at(-1);
  if (!last || last.role !== 'assistant') {
    return messages;
  }
  return [...messages.slice(0, -1), { ...last, content: last.content + content }];
}

export function removeTrailingTurn(
  messages: ChatMessage[],
  userId: string,
  assistantId: string
): ChatMessage[] {
  const user = messages.at(-2);
  const assistant = messages.at(-1);
  if (user?.role !== 'user' || user.id !== userId ||
      assistant?.role !== 'assistant' || assistant.id !== assistantId) {
    return messages;
  }
  return messages.slice(0, -2);
}
