/*
 * Chat types.
 * Single responsibility: define the browser's text-only chat state and wire
 * shapes. These types do not model tools, workspaces, confirmations, or a
 * separately rendered reasoning channel.
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
