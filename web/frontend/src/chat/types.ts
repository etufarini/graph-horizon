/*
 * Browser chat types: define runtime, plain transcript, context-capacity,
 * timing, and bounded persistence-result shapes without owning side effects.
 */
export type Role = 'system' | 'user' | 'assistant';

export interface TranscriptMessage {
  role: Exclude<Role, 'system'>;
  content: string;
}

export interface ChatMessage extends TranscriptMessage {
  id: string;
}

export interface WireMessage {
  role: Role;
  content: string;
}

export type ChatStatus = 'idle' | 'streaming' | 'error';

export type PersistenceWarning = 'invalid-record' | 'unavailable';

export interface ConversationLoadResult {
  messages: TranscriptMessage[];
  warning: PersistenceWarning | null;
}

export interface RuntimeContext {
  contextLimit: number;
  maxTokens: number;
  safeTotalBudget: number;
}

export interface ContextUsage {
  estimatedTokens: number;
  percent: number;
  progress: number;
}

export type ContextConfigResult =
  | { ok: true; context: RuntimeContext }
  | { ok: false; error: 'unavailable' | 'no-prompt-space' };

export type ContextAdmission =
  | { ok: true; usage: ContextUsage }
  | {
      ok: false;
      usage: ContextUsage;
      estimatedTokens: number;
      maxTokens: number;
      safeTotalBudget: number;
    };

export interface ChatSnapshot {
  messages: ChatMessage[];
  status: ChatStatus;
  error: string | null;
  systemPrompt: string;
  generationStartedAt: number | null;
  generationMs: number | null;
}

export interface StreamDelta {
  content: string;
}
