/*
 * Browser chat types: define runtime transport, per-chat prompts and transcripts,
 * private archives, prompt capacity, timing, and bounded persistence results.
 * Storage and lifecycle behavior remain outside.
 */
export type Role = 'system' | 'user' | 'assistant';

export interface TranscriptMessage {
  role: Exclude<Role, 'system'>;
  content: string;
}

export interface ChatMessage extends TranscriptMessage {
  id: string;
}

export interface ChatRecord {
  id: string;
  title: string;
  systemPrompt: string;
  messages: ChatMessage[];
  updatedAt: number;
}

export interface ChatCollection {
  activeChatId: string;
  chats: ChatRecord[];
}

export interface ChatArchiveRecord {
  version: 3;
  activeChatId: string;
  chats: Array<{
    id: string;
    title: string;
    systemPrompt: string;
    messages: TranscriptMessage[];
    updatedAt: number;
  }>;
}

export interface WireMessage {
  role: Role;
  content: string;
}

export type ChatStatus = 'idle' | 'streaming' | 'error';

export type PersistenceWarning = 'invalid-record' | 'unavailable';

export interface ChatLoadResult {
  collection: ChatCollection;
  warning: PersistenceWarning | null;
}

export type ChatSaveResult = PersistenceWarning | null;

export interface RuntimeContext {
  contextLimit: number;
  safePromptBudget: number;
}

export interface ContextUsage {
  estimatedTokens: number;
  percent: number;
  progress: number;
}

export type ContextConfigResult =
  | { ok: true; context: RuntimeContext }
  | { ok: false; error: 'unavailable' };

export type ContextAdmission =
  | { ok: true; usage: ContextUsage }
  | {
      ok: false;
      usage: ContextUsage;
      estimatedTokens: number;
      safePromptBudget: number;
    };

export interface ChatSnapshot {
  collection: ChatCollection;
  status: ChatStatus;
  error: string | null;
  persistenceWarning: PersistenceWarning | null;
  generationStartedAt: number | null;
  generationMs: number | null;
}

export interface StreamDelta {
  content: string;
}
