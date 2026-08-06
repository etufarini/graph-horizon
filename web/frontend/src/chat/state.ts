/*
 * Chat state.
 * Single responsibility: own capacity-gated transcript mutations, transport
 * rollback, stop behavior, and monotonic generation timing.
 */
import { get, writable } from 'svelte/store';
import { streamAssistant } from './client';
import { admitMessages } from './context';
import { loadSystemPrompt, saveSystemPrompt } from './systemPrompt';
import { parseChatFile } from './transfer';
import type {
  ChatMessage,
  ChatSnapshot,
  RuntimeContext,
  StreamDelta,
  WireMessage
} from './types';

const FAILED = 'Richiesta non riuscita';
const INTERRUPTED = 'Connessione interrotta';

function emptySnapshot(): ChatSnapshot {
  return {
    messages: [],
    status: 'idle',
    error: null,
    systemPrompt: loadSystemPrompt(),
    generationStartedAt: null,
    generationMs: null
  };
}

function nextId(role: string): string {
  const random = crypto.randomUUID?.() ?? Math.random().toString(16).slice(2);
  return `${role}-${Date.now()}-${random}`;
}

export function wireMessages(
  messages: ChatMessage[],
  systemPrompt: string,
  draft = ''
): WireMessage[] {
  const wire: WireMessage[] = messages.map(message => ({
    role: message.role,
    content: message.content
  }));
  const trimmedDraft = draft.trim();
  if (trimmedDraft) {
    wire.push({ role: 'user', content: trimmedDraft });
  }
  const trimmed = systemPrompt.trim();
  if (trimmed) {
    wire.unshift({ role: 'system', content: trimmed });
  }
  return wire;
}

function createChatState() {
  const store = writable<ChatSnapshot>(emptySnapshot());
  let controller: AbortController | null = null;

  async function send(text: string, context: RuntimeContext): Promise<void> {
    const prompt = text.trim();
    const current = get(store);
    if (!prompt || current.status === 'streaming') {
      return;
    }

    const wire = wireMessages(current.messages, current.systemPrompt, prompt);
    const admission = admitMessages(wire, context);
    if (!admission.ok) {
      store.set({
        ...current,
        status: 'error',
        error: `Contesto insufficiente: ~${admission.estimatedTokens} token + ${admission.maxTokens} riservati superano il budget sicuro di ${admission.safeTotalBudget} token`
      });
      return;
    }

    const user: ChatMessage = { id: nextId('user'), role: 'user', content: prompt };
    const assistant: ChatMessage = { id: nextId('assistant'), role: 'assistant', content: '' };
    const outgoing = [...current.messages, user];
    controller = new AbortController();
    const request = controller;
    const generationStartedAt = performance.now();
    store.set({
      ...current,
      messages: [...outgoing, assistant],
      status: 'streaming',
      error: null,
      generationStartedAt,
      generationMs: null
    });

    try {
      await streamAssistant(
        wire,
        context.maxTokens,
        appendAssistant,
        request.signal
      );
      const generationMs = performance.now() - generationStartedAt;
      store.update(snapshot => ({
        ...snapshot,
        status: 'idle',
        error: null,
        generationStartedAt: null,
        generationMs
      }));
    } catch (error) {
      const aborted =
        request.signal.aborted || (error instanceof DOMException && error.name === 'AbortError');
      if (aborted) {
        // Keep the stopped turn, including an empty or partial assistant message,
        // so every later request still sees an alternating transcript.
        store.update(snapshot => ({
          ...snapshot,
          status: 'idle',
          error: null,
          generationStartedAt: null,
          generationMs: null
        }));
      } else {
        removeTrailingTurn(user.id, assistant.id);
        const message = error instanceof Error && error.message === FAILED ? FAILED : INTERRUPTED;
        store.update(snapshot => ({
          ...snapshot,
          status: 'error',
          error: message,
          generationStartedAt: null,
          generationMs: null
        }));
      }
    } finally {
      controller = null;
    }
  }

  function stop(): void {
    controller?.abort();
  }

  function setSystemPrompt(text: string): void {
    store.update(snapshot => ({ ...snapshot, systemPrompt: text }));
    saveSystemPrompt(text);
  }

  function importChat(text: string): void {
    const result = parseChatFile(text);
    if (!result.ok) {
      const message =
        result.error === 'invalid-json'
          ? 'File non valido: JSON non riconosciuto'
          : 'File non valido: formato chat non riconosciuto';
      store.update(snapshot => ({ ...snapshot, status: 'error', error: message }));
      return;
    }
    const messages: ChatMessage[] = result.payload.messages.map(message => ({
      id: nextId(message.role),
      role: message.role,
      content: message.content
    }));
    store.update(snapshot => ({
      ...snapshot,
      messages,
      status: 'idle',
      error: null,
      systemPrompt: result.payload.systemPrompt,
      generationStartedAt: null,
      generationMs: null
    }));
    saveSystemPrompt(result.payload.systemPrompt);
  }

  function removeTrailingTurn(userId: string, assistantId: string): void {
    store.update(snapshot => {
      const user = snapshot.messages[snapshot.messages.length - 2];
      const assistant = snapshot.messages[snapshot.messages.length - 1];
      if (
        !user ||
        user.role !== 'user' ||
        user.id !== userId ||
        !assistant ||
        assistant.role !== 'assistant' ||
        assistant.id !== assistantId
      ) {
        return snapshot;
      }
      // Roll back only the in-flight pair; completed history is immutable here.
      return { ...snapshot, messages: snapshot.messages.slice(0, -2) };
    });
  }

  function appendAssistant(delta: StreamDelta): void {
    store.update(snapshot => ({
      ...snapshot,
      messages: snapshot.messages.map((message, index) => {
        const isLast = index === snapshot.messages.length - 1;
        if (!isLast || message.role !== 'assistant') {
          return message;
        }
        return { ...message, content: message.content + delta.content };
      })
    }));
  }

  return {
    subscribe: store.subscribe,
    send,
    stop,
    setSystemPrompt,
    importChat
  };
}

export const chat = createChatState();
