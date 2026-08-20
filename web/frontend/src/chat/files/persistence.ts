/*
 * IndexedDB Markdown-file boundary: own the origin-scoped schema and exact
 * asynchronous list/write/delete/prune operations. In-memory fallback, prompt
 * projection, lifecycle policy, and presentation remain outside.
 */
import { parseMarkdownFileRecord, type MarkdownFileRecord } from './record.ts';

const DATABASE = 'graph-horizon';
const VERSION = 1;
const STORE = 'markdownFiles';
const CHAT_INDEX = 'chatId';

export interface MarkdownFileList {
  files: MarkdownFileRecord[];
  invalid: boolean;
}

export interface MarkdownFileStorage {
  list(chatId: string): Promise<MarkdownFileList>;
  write(files: MarkdownFileRecord[], deletedIds?: string[]): Promise<void>;
  delete(id: string): Promise<void>;
  deleteChat(chatId: string): Promise<void>;
  prune(validChatIds: string[]): Promise<void>;
  persist(): Promise<boolean>;
}

let connection: Promise<IDBDatabase> | null = null;

export const browserMarkdownFileStorage: MarkdownFileStorage = {
  async list(chatId) {
    const database = await openDatabase();
    return new Promise((resolve, reject) => {
      const transaction = database.transaction(STORE, 'readwrite');
      const cursor = transaction.objectStore(STORE).index(CHAT_INDEX).openCursor(IDBKeyRange.only(chatId));
      const files: MarkdownFileRecord[] = [];
      let invalid = false;
      cursor.onsuccess = () => {
        const row = cursor.result;
        if (!row) return;
        const parsed = parseMarkdownFileRecord(row.value);
        if (parsed) files.push(parsed);
        else {
          invalid = true;
          row.delete();
        }
        row.continue();
      };
      cursor.onerror = () => reject(cursor.error);
      transaction.oncomplete = () => resolve({ files, invalid });
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
  },

  async write(files, deletedIds = []) {
    const database = await openDatabase();
    await transaction(database, store => {
      for (const id of deletedIds) store.delete(id);
      for (const file of files) store.put(file);
    });
  },

  async delete(id) {
    const database = await openDatabase();
    await transaction(database, store => store.delete(id));
  },

  async deleteChat(chatId) {
    const database = await openDatabase();
    await cursorDelete(database, row =>
      typeof row === 'object' && row !== null &&
      (row as Record<string, unknown>).chatId === chatId
    );
  },

  async prune(validChatIds) {
    const database = await openDatabase();
    const valid = new Set(validChatIds);
    await cursorDelete(database, row => {
      const parsed = parseMarkdownFileRecord(row);
      return !parsed || !valid.has(parsed.chatId);
    });
  },

  async persist() {
    try {
      return await navigator.storage?.persist?.() ?? false;
    } catch {
      return false;
    }
  }
};

function openDatabase(): Promise<IDBDatabase> {
  if (connection) return connection;
  const opening = new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DATABASE, VERSION);
    request.onupgradeneeded = () => {
      const store = request.result.createObjectStore(STORE, { keyPath: 'id' });
      store.createIndex(CHAT_INDEX, 'chatId', { unique: false });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error('blocked'));
  }).catch((error): never => {
    connection = null;
    throw error;
  });
  connection = opening;
  return opening;
}

function transaction(
  database: IDBDatabase,
  operation: (store: IDBObjectStore) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    const current = database.transaction(STORE, 'readwrite');
    operation(current.objectStore(STORE));
    current.oncomplete = () => resolve();
    current.onerror = () => reject(current.error);
    current.onabort = () => reject(current.error);
  });
}

function cursorDelete(
  database: IDBDatabase,
  remove: (row: unknown) => boolean
): Promise<void> {
  return new Promise((resolve, reject) => {
    const current = database.transaction(STORE, 'readwrite');
    const request = current.objectStore(STORE).openCursor();
    request.onsuccess = () => {
      const cursor = request.result;
      if (!cursor) return;
      if (remove(cursor.value)) cursor.delete();
      cursor.continue();
    };
    request.onerror = () => reject(request.error);
    current.oncomplete = () => resolve();
    current.onerror = () => reject(current.error);
    current.onabort = () => reject(current.error);
  });
}
