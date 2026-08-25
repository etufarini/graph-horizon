/* Transactional generation lifecycle: admits local plus optional search context,
 * then owns one streamed request, rollback, Stop, timing, and checkpoints. */
import { get, type Writable } from 'svelte/store';
import { isExpectedFailure, streamAssistant } from './client.ts';
import { admitMessages } from './context.ts';
import { expandPromptWithMarkdownFiles } from './files/context.ts';
import { activeChat, replaceChatMessages } from './sessions.ts';
import {
  appendAssistant,
  attachAssistantSearch,
  findTurn,
  finalPair,
  hydrateTranscript,
  replaceFromTurn,
  wireMessages
} from './transcript.ts';
import type { ChatSnapshot, RuntimeContext, SearchSelection, StreamEvent } from './types.ts';
import type { MarkdownFileRecord } from './files/record.ts';

const INTERRUPTED = 'Response interrupted';

export function createGeneration(
  store: Writable<ChatSnapshot>,
  checkpoint: (chatId: string) => void
) {
  let controller: AbortController | null = null;

  async function send(
    text: string,
    context: RuntimeContext,
    files: MarkdownFileRecord[] = [],
    search: SearchSelection | null = null
  ): Promise<void> {
    await generate('append', text.trim(), context, '', files, search);
  }

  async function regenerate(
    context: RuntimeContext,
    files: MarkdownFileRecord[] = [],
    search: SearchSelection | null = null
  ): Promise<void> {
    const snapshot = get(store);
    const pair = finalPair(activeChat(snapshot.collection).messages);
    if (pair) {
      await generate('replace', pair[0].content, context, pair[0].id, files, search);
    }
  }

  async function editPrompt(
    userId: string,
    text: string,
    context: RuntimeContext,
    files: MarkdownFileRecord[] = [],
    search: SearchSelection | null = null
  ): Promise<void> {
    await generate('replace', text.trim(), context, userId, files, search);
  }

  function stop(): void {
    controller?.abort();
  }

  async function generate(
    mode: 'append' | 'replace',
    prompt: string,
    context: RuntimeContext,
    userId = '',
    files: MarkdownFileRecord[] = [],
    search: SearchSelection | null = null
  ): Promise<void> {
    const current = get(store);
    if (!prompt || current.status === 'streaming') {
      return;
    }
    const chat = activeChat(current.collection);
    const turn = mode === 'replace' ? findTurn(chat.messages, userId) : null;
    if (mode === 'replace' && !turn) {
      return;
    }
    const previousMessages = chat.messages;
    const prior = turn ? previousMessages.slice(0, turn.index) : previousMessages;
    const wire = wireMessages(prior, chat.systemPrompt);
    wire.push({ role: 'user', content: expandPromptWithMarkdownFiles(prompt, files) });
    const admission = admitMessages(wire, context, search ? context.search.maxContextCharacters : 0);
    if (!admission.ok) {
      store.set({
        ...current,
        status: 'error',
        error: `Insufficient context: ~${admission.estimatedTokens} estimated tokens exceed the safe budget of ${admission.safePromptBudget} tokens`
      });
      return;
    }

    // The user identity locates the causal cut; replaceFromTurn creates a fresh
    // assistant identity so old request events cannot target the new version.
    const messages = mode === 'append'
      ? [...previousMessages, ...hydrateTranscript([
          { role: 'user', content: prompt },
          { role: 'assistant', content: '' }
        ])]
      : replaceFromTurn(previousMessages, turn!.user.id, prompt, '');
    const chatId = chat.id;
    const assistantId = messages.at(-1)!.id;
    controller = new AbortController();
    const request = controller;
    store.set({
      ...current,
      collection: replaceChatMessages(current.collection, chatId, messages),
      status: 'streaming',
      error: null,
      telemetry: {
        phase: 'waiting',
        phaseStartedAt: performance.now(),
        stats: null
      }
    });

    try {
      await streamAssistant(
        wire,
        event => applyEvent(chatId, assistantId, event),
        request.signal,
        search ? { terms: search.query.trim() || prompt.trim(), selection: search } : null
      );
      store.update(snapshot => ({
        ...snapshot,
        status: 'idle',
        error: null
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
          telemetry: null
        }));
        checkpoint(chatId);
      } else {
        const message = error instanceof Error ? error.message : '';
        store.update(snapshot => ({
          ...snapshot,
          collection: replaceChatMessages(snapshot.collection, chatId, previousMessages),
          status: 'error',
          error: isExpectedFailure(message) ? message : INTERRUPTED,
          telemetry: null
        }));
      }
    } finally {
      if (controller === request) {
        controller = null;
      }
    }
  }

  function applyEvent(chatId: string, assistantId: string, event: StreamEvent): void {
    store.update(snapshot => {
      const chat = snapshot.collection.chats.find(candidate => candidate.id === chatId);
      if (!chat || chat.messages.at(-1)?.id !== assistantId) {
        return snapshot;
      }
      if (snapshot.telemetry?.stats) throw new Error(INTERRUPTED);
      if (event.type === 'search') {
        return {
          ...snapshot,
          collection: replaceChatMessages(
            snapshot.collection,
            chatId,
            attachAssistantSearch(chat.messages, event.search)
          )
        };
      }
      if (event.type === 'phase') {
        const prior = snapshot.telemetry?.phase;
        const valid = (prior === 'waiting' && event.phase === 'prefill') ||
          (prior === 'prefill' && event.phase === 'decode');
        if (!valid) throw new Error(INTERRUPTED);
        return {
          ...snapshot,
          telemetry: { phase: event.phase, phaseStartedAt: performance.now(), stats: null }
        };
      }
      if (event.type === 'stats') {
        if (snapshot.telemetry?.phase !== 'decode') throw new Error(INTERRUPTED);
        return {
          ...snapshot,
          telemetry: { phase: null, phaseStartedAt: null, stats: event.stats }
        };
      }
      return {
        ...snapshot,
        collection: replaceChatMessages(
          snapshot.collection,
          chatId,
          appendAssistant(chat.messages, event.content)
        )
      };
    });
  }

  return { send, regenerate, editPrompt, stop };
}
