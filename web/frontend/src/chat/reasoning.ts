/*
 * This module recognizes the current raw Ministral Reasoning markers for
 * presentation only. Chat state, persistence, transport, Markdown rendering,
 * and model selection remain owned by their existing modules.
 */

const THINK_OPEN = '[THINK]';
const THINK_CLOSE = '[/THINK]';

export type ReasoningView = {
  thinking?: string;
  answer: string;
  pending: boolean;
};

export function splitReasoning(raw: string, streaming: boolean): ReasoningView {
  const candidate = raw.trimStart();

  if (streaming && candidate !== THINK_OPEN && THINK_OPEN.startsWith(candidate)) {
    return { answer: '', pending: true };
  }

  if (!candidate.startsWith(THINK_OPEN)) {
    return { answer: raw, pending: false };
  }

  const content = candidate.slice(THINK_OPEN.length);
  const closeAt = content.indexOf(THINK_CLOSE);

  if (closeAt === -1) {
    return { thinking: content, answer: '', pending: false };
  }

  return {
    thinking: content.slice(0, closeAt),
    answer: content.slice(closeAt + THINK_CLOSE.length),
    pending: false,
  };
}
