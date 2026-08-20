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
  contextLimit: number;
  percent: number;
  progress: number;
}

export type GenerationPhase = 'waiting' | 'prefill' | 'decode';

export interface GenerationStats {
  promptTokens: number;
  prefillTokens: number;
  completionTokens: number;
  prefillMs: number;
  decodeMs: number;
}

export interface GenerationTelemetry {
  phase: GenerationPhase | null;
  phaseStartedAt: number | null;
  stats: GenerationStats | null;
}

export interface RuntimeMemory {
  weights: bigint;
  kv: bigint;
  scratch: bigint;
  fixed: bigint;
  staging: bigint;
  crossing: bigint;
  reserve: bigint;
  total: bigint;
}

export interface RuntimePlacement {
  mode: string;
  cpuLayers: number;
  acceleratorLayers: number;
  cpu: RuntimeMemory;
  accelerator: RuntimeMemory;
}

export interface RuntimeInfo {
  modelName: string;
  backend: string;
  placement: RuntimePlacement | null;
}

export type RuntimeInfoResult =
  | { ok: true; info: RuntimeInfo }
  | { ok: false; error: 'unavailable' };

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
  telemetry: GenerationTelemetry | null;
}

export type StreamEvent =
  | { type: 'content'; content: string }
  | { type: 'phase'; phase: Exclude<GenerationPhase, 'waiting'> }
  | { type: 'stats'; stats: GenerationStats };
