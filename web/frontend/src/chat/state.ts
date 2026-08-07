/*
 * Chat state.
 * Single responsibility: own capacity-gated transcript mutations, transport
 * rollback, stop behavior, and monotonic generation timing.
 */
import { get, writable } from 'svelte/store';
import { streamAssistant } from './client';
import { admitMessages } from './context';
import { loadSystemPrompt, saveSystemPrompt } from './systemPrompt';
import {
  appendAssistant as appendAssistantContent,
  hydrateTranscript,
  removeTrailingTurn,
  wireMessages
} from './transcript';
import { parseChatFile } from './transfer';
import type { ChatSnapshot, RuntimeContext, StreamDelta } from './types';

export { wireMessages } from './transcript';

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

    const [user, assistant] = hydrateTranscript([
      { role: 'user', content: prompt },
      { role: 'assistant', content: '' }
    ]);
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
        store.update(snapshot => ({
          ...snapshot,
          messages: removeTrailingTurn(snapshot.messages, user.id, assistant.id)
        }));
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
    const messages = hydrateTranscript(result.payload.messages);
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

  function appendAssistant(delta: StreamDelta): void {
    store.update(snapshot => ({
      ...snapshot,
      messages: appendAssistantContent(snapshot.messages, delta.content)
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
