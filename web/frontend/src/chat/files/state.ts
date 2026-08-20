/*
 * Per-chat Markdown-file lifecycle coordinator: own active loading, bounded
 * additions, replacement, deletion, memory fallback, and persistence warnings.
 * Rendering, chat mutation, prompt transport, and IndexedDB mechanics are excluded.
 */
import { get, writable } from 'svelte/store';
import { admitMessages } from '../context.ts';
import { markdownFileOverhead } from './context.ts';
import {
  MAX_CHAT_FILE_BYTES,
  MAX_FILES_PER_CHAT,
  readMarkdownFile,
  type MarkdownFileError,
  type MarkdownFileRecord
} from './record.ts';
import {
  browserMarkdownFileStorage,
  type MarkdownFileStorage
} from './persistence.ts';
import type { RuntimeContext, WireMessage } from '../types.ts';

export type MarkdownFileWarning = 'invalid-record' | 'unavailable';

export interface MarkdownFileSnapshot {
  chatId: string | null;
  files: MarkdownFileRecord[];
  ready: boolean;
  busy: boolean;
  error: string | null;
  warning: MarkdownFileWarning | null;
}

export function createMarkdownFileState(storage: MarkdownFileStorage) {
  const store = writable<MarkdownFileSnapshot>({
    chatId: null, files: [], ready: false, busy: false, error: null, warning: null
  });
  const memory = new Map<string, MarkdownFileRecord[]>();
  let selection = 0;
  let persistenceRequested = false;

  async function select(chatId: string): Promise<void> {
    const request = ++selection;
    const cached = memory.get(chatId) ?? [];
    store.set({ chatId, files: cached, ready: false, busy: false, error: null, warning: null });
    try {
      const result = await storage.list(chatId);
      if (request !== selection) return;
      const files = ordered(result.files);
      memory.set(chatId, files);
      store.set({
        chatId, files, ready: true, busy: false, error: null,
        warning: result.invalid ? 'invalid-record' : null
      });
    } catch {
      if (request !== selection) return;
      store.set({
        chatId, files: cached, ready: true, busy: false, error: null, warning: 'unavailable'
      });
    }
  }

  async function add(
    selected: File[],
    chatId: string,
    baseMessages: WireMessage[],
    context: RuntimeContext
  ): Promise<void> {
    const current = get(store);
    if (!current.ready || current.busy || current.chatId !== chatId || selected.length === 0) return;
    store.set({ ...current, busy: true, error: null });
    const prepared: MarkdownFileRecord[] = [];
    for (const file of selected) {
      const result = await readMarkdownFile(file, chatId);
      if (!result.ok) {
        fail(fileError(result.error));
        return;
      }
      prepared.push(result.record);
    }

    const replacements = new Set(prepared.map(file => file.name));
    const retained = current.files.filter(file => !replacements.has(file.name));
    const candidate = ordered([...retained, ...prepared]);
    if (candidate.length > MAX_FILES_PER_CHAT) {
      fail(`Limite file superato: massimo ${MAX_FILES_PER_CHAT} per chat`);
      return;
    }
    const bytes = candidate.reduce((total, file) => total + file.utf8Bytes, 0);
    if (!Number.isSafeInteger(bytes) || bytes > MAX_CHAT_FILE_BYTES) {
      fail('Dimensione complessiva dei file superiore a 2 MiB');
      return;
    }
    const overhead = markdownFileOverhead(candidate);
    const admission = admitMessages(
      overhead ? [...baseMessages, { role: 'user', content: overhead }] : baseMessages,
      context
    );
    if (!admission.ok) {
      fail(`Contesto insufficiente: i file superano il budget sicuro di ${admission.safePromptBudget} token`);
      return;
    }

    const deletedIds = current.files.filter(file => replacements.has(file.name)).map(file => file.id);
    await durable(chatId, candidate, () => storage.write(prepared, deletedIds));
    if (!persistenceRequested) {
      persistenceRequested = true;
      void storage.persist();
    }
  }

  async function remove(id: string): Promise<void> {
    const current = get(store);
    if (!current.ready || current.busy || !current.chatId) return;
    const files = current.files.filter(file => file.id !== id);
    if (files.length === current.files.length) return;
    store.set({ ...current, busy: true, error: null });
    await durable(current.chatId, files, () => storage.delete(id));
  }

  async function removeChat(chatId: string): Promise<void> {
    memory.delete(chatId);
    if (get(store).chatId === chatId) selection += 1;
    try {
      await storage.deleteChat(chatId);
    } catch {
      store.update(current => ({ ...current, warning: 'unavailable' }));
    }
  }

  async function reconcile(validChatIds: string[]): Promise<void> {
    const valid = new Set(validChatIds);
    for (const id of memory.keys()) if (!valid.has(id)) memory.delete(id);
    try {
      await storage.prune(validChatIds);
    } catch {
      store.update(current => ({ ...current, warning: 'unavailable' }));
    }
  }

  async function durable(
    chatId: string,
    files: MarkdownFileRecord[],
    operation: () => Promise<void>
  ): Promise<void> {
    let warning: MarkdownFileWarning | null = null;
    try {
      await operation();
    } catch {
      warning = 'unavailable';
    }
    memory.set(chatId, files);
    const current = get(store);
    if (current.chatId === chatId) {
      store.set({ ...current, files, busy: false, error: null, warning });
    }
  }

  function fail(error: string): void {
    store.update(current => ({ ...current, busy: false, error }));
  }

  return { subscribe: store.subscribe, select, add, remove, removeChat, reconcile };
}

function fileError(error: MarkdownFileError): string {
  switch (error) {
    case 'extension': return 'File non valido: è richiesta l’estensione .md';
    case 'name': return 'File non valido: nome non ammesso';
    case 'empty': return 'File non valido: contenuto vuoto';
    case 'encoding': return 'File non valido: il contenuto deve essere UTF-8';
    case 'oversized': return 'File non valido: dimensione superiore a 1 MiB';
    default: return 'File non leggibile';
  }
}

function ordered(files: MarkdownFileRecord[]): MarkdownFileRecord[] {
  return [...files].sort((left, right) =>
    left.addedAt - right.addedAt || (left.id < right.id ? -1 : left.id > right.id ? 1 : 0)
  );
}

export const markdownFiles = createMarkdownFileState(browserMarkdownFileStorage);
