/*
 * Pure transcript boundary: validates and hydrates alternating transcripts,
 * projects wire context, and appends, locates, replaces, truncates, or removes
 * complete turns. Lifecycle, storage, transport, and presentation remain outside.
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

export function finalPair(messages: ChatMessage[]): [ChatMessage, ChatMessage] | null {
  const user = messages.at(-2);
  const assistant = messages.at(-1);
  return user?.role === 'user' && assistant?.role === 'assistant'
    ? [user, assistant]
    : null;
}

export function findTurn(
  messages: ChatMessage[],
  userId: string
): { index: number; user: ChatMessage; assistant: ChatMessage } | null {
  const index = messages.findIndex(message => message.id === userId);
  const user = messages[index];
  const assistant = messages[index + 1];
  return index >= 0 && index % 2 === 0 &&
    user?.role === 'user' && assistant?.role === 'assistant'
    ? { index, user, assistant }
    : null;
}

export function replaceFromTurn(
  messages: ChatMessage[],
  userId: string,
  userContent: string,
  assistantContent: string
): ChatMessage[] {
  const turn = findTurn(messages, userId);
  if (!turn) return messages;
  // A revised prompt invalidates its old answer and every causal successor.
  return [
    ...messages.slice(0, turn.index),
    { ...turn.user, content: userContent },
    { ...turn.assistant, content: assistantContent }
  ];
}

export function removeTrailingTurn(
  messages: ChatMessage[],
  userId: string,
  assistantId: string
): ChatMessage[] {
  const pair = finalPair(messages);
  if (!pair || pair[0].id !== userId || pair[1].id !== assistantId) {
    return messages;
  }
  return messages.slice(0, -2);
}
