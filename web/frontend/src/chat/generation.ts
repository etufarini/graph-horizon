/*
 * Transactional generation lifecycle: admits prompt occupancy before visible
 * mutation, then owns one request, timing, deltas, append/replacement rollback,
 * voluntary Stop commit, and stable checkpoint signals. Chat-list, storage,
 * Reasoning presentation, and UI are excluded.
 */
import { get, type Writable } from 'svelte/store';
import { streamAssistant } from './client.ts';
import { admitMessages } from './context.ts';
import { activeChat } from './sessions.ts';
import {
  appendAssistant,
  beforeFinalPair,
  finalPair,
  hydrateTranscript,
  replaceFinalPair,
  wireMessages
} from './transcript.ts';
import type {
  ChatCollection,
  ChatMessage,
  ChatSnapshot,
  RuntimeContext,
  StreamDelta
} from './types.ts';

const FAILED = 'Richiesta non riuscita';
const INTERRUPTED = 'Risposta interrotta';

export function createGeneration(
  store: Writable<ChatSnapshot>,
  checkpoint: (chatId: string) => void
) {
  let controller: AbortController | null = null;

  async function send(text: string, context: RuntimeContext): Promise<void> {
    await generate('append', text.trim(), context);
  }

  async function regenerate(context: RuntimeContext): Promise<void> {
    const snapshot = get(store);
    const pair = finalPair(activeChat(snapshot.collection).messages);
    if (pair) {
      await generate('replace', pair[0].content, context);
    }
  }

  async function editLastPrompt(text: string, context: RuntimeContext): Promise<void> {
    await generate('replace', text.trim(), context);
  }

  function stop(): void {
    controller?.abort();
  }

  async function generate(
    mode: 'append' | 'replace',
    prompt: string,
    context: RuntimeContext
  ): Promise<void> {
    const current = get(store);
    if (!prompt || current.status === 'streaming') {
      return;
    }
    const chat = activeChat(current.collection);
    const previousPair = mode === 'replace' ? finalPair(chat.messages) : null;
    if (mode === 'replace' && !previousPair) {
      return;
    }
    const previousMessages = chat.messages;
    const prior = mode === 'replace' ? beforeFinalPair(previousMessages) : previousMessages;
    const wire = wireMessages(prior, chat.systemPrompt);
    wire.push({ role: 'user', content: prompt });
    const admission = admitMessages(wire, context);
    if (!admission.ok) {
      store.set({
        ...current,
        status: 'error',
        error: `Contesto insufficiente: ~${admission.estimatedTokens} token stimati superano il budget sicuro di ${admission.safePromptBudget} token`
      });
      return;
    }

    const pair = mode === 'append'
      ? hydrateTranscript([
          { role: 'user', content: prompt },
          { role: 'assistant', content: '' }
        ])
      : previousPair!;
    const messages = mode === 'append'
      ? [...previousMessages, ...pair]
      : replaceFinalPair(previousMessages, pair[0].id, pair[1].id, prompt, '');
    const chatId = chat.id;
    const assistantId = pair[1].id;
    controller = new AbortController();
    const request = controller;
    const generationStartedAt = performance.now();
    store.set({
      ...current,
      collection: withMessages(current.collection, chatId, messages),
      status: 'streaming',
      error: null,
      generationStartedAt,
      generationMs: null
    });

    try {
      await streamAssistant(
        wire,
        context.contextLimit,
        delta => applyDelta(chatId, assistantId, delta),
        request.signal
      );
      store.update(snapshot => ({
        ...snapshot,
        status: 'idle',
        error: null,
        generationStartedAt: null,
        generationMs: performance.now() - generationStartedAt
      }));
      checkpoint(chatId);
    } catch (error) {
      const stopped = request.signal.aborted &&
        error instanceof DOMException && error.name === 'AbortError';
      if (stopped) {
        store.update(snapshot => ({
          ...snapshot,
          status: 'idle',
          error: null,
          generationStartedAt: null,
          generationMs: null
        }));
        checkpoint(chatId);
      } else {
        store.update(snapshot => ({
          ...snapshot,
          collection: withMessages(snapshot.collection, chatId, previousMessages),
          status: 'error',
          error: error instanceof Error && error.message === FAILED ? FAILED : INTERRUPTED,
          generationStartedAt: null,
          generationMs: null
        }));
      }
    } finally {
      if (controller === request) {
        controller = null;
      }
    }
  }

  function applyDelta(chatId: string, assistantId: string, delta: StreamDelta): void {
    store.update(snapshot => {
      const chat = snapshot.collection.chats.find(candidate => candidate.id === chatId);
      if (!chat || chat.messages.at(-1)?.id !== assistantId) {
        return snapshot;
      }
      return {
        ...snapshot,
        collection: withMessages(
          snapshot.collection,
          chatId,
          appendAssistant(chat.messages, delta.content)
        )
      };
    });
  }

  return { send, regenerate, editLastPrompt, stop };
}

function withMessages(
  collection: ChatCollection,
  chatId: string,
  messages: ChatMessage[]
): ChatCollection {
  return {
    ...collection,
    chats: collection.chats.map(chat => chat.id === chatId ? { ...chat, messages } : chat)
  };
}
