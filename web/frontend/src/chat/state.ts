/*
 * Chat lifecycle orchestration: owns capacity admission, transport rollback,
 * timing, and stable persistence checkpoints. Record format and presentation
 * remain in the persistence/transcript modules and Svelte components.
 */
import { get, writable } from 'svelte/store';
import { streamAssistant } from './client.ts';
import { admitMessages } from './context.ts';
import { clearConversation, loadConversation, saveConversation } from './persistence.ts';
import { loadSystemPrompt, saveSystemPrompt } from './systemPrompt.ts';
import {
  appendAssistant as appendAssistantContent,
  hydrateTranscript,
  removeTrailingTurn,
  wireMessages
} from './transcript.ts';
import { parseChatFile } from './transfer.ts';
import type { ChatSnapshot, RuntimeContext, StreamDelta } from './types';

export { wireMessages } from './transcript.ts';

const FAILED = 'Richiesta non riuscita';
const INTERRUPTED = 'Connessione interrotta';

function emptySnapshot(): ChatSnapshot {
  const restored = loadConversation();
  return {
    messages: hydrateTranscript(restored.messages),
    status: 'idle',
    error: null,
    persistenceWarning: restored.warning,
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
      await streamAssistant(wire, context.maxTokens, appendAssistant, request.signal);
      const generationMs = performance.now() - generationStartedAt;
      store.update(snapshot => ({
        ...snapshot,
        status: 'idle',
        error: null,
        generationStartedAt: null,
        generationMs
      }));
      checkpoint();
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
        checkpoint();
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
    checkpoint();
  }

  function newChat(): void {
    const current = get(store);
    if (current.status === 'streaming') return;
    store.set({
      ...current,
      messages: [],
      status: 'idle',
      error: null,
      generationStartedAt: null,
      generationMs: null
    });
    const persistenceWarning = clearConversation();
    store.update(snapshot => ({ ...snapshot, persistenceWarning }));
  }

  function checkpoint(): void {
    const persistenceWarning = saveConversation(get(store).messages);
    store.update(snapshot => ({ ...snapshot, persistenceWarning }));
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
    importChat,
    newChat
  };
}

export const chat = createChatState();
