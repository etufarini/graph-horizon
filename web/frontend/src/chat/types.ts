/*
 * Chat types.
 * Single responsibility: define browser chat, immutable context-capacity, and
 * monotonic timing state plus the text-only wire shapes.
 */
export type Role = 'system' | 'user' | 'assistant';

export interface ChatMessage {
  id: string;
  role: Exclude<Role, 'system'>;
  content: string;
}

export interface WireMessage {
  role: Role;
  content: string;
}

export type ChatStatus = 'idle' | 'streaming' | 'error';

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

// Retained until the transport/status replacement task removes its final users.
export interface GenerationStats {
  promptTokens: number;
  completionTokens: number;
  prefillMs: number;
  decodeMs: number;
}

export interface ChatSnapshot {
  messages: ChatMessage[];
  status: ChatStatus;
  error: string | null;
  systemPrompt: string;
  stats: GenerationStats | null;
}

export interface StreamDelta {
  content: string;
}
