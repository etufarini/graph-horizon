/*
 * This module recognizes the current raw Ministral Reasoning markers for
 * presentation only. Chat state, persistence, transport, Markdown rendering,
 * and model selection remain outside this presentation parser.
 */

const THINK_OPEN = '[THINK]';
const THINK_CLOSE = '[/THINK]';

export type ReasoningView = {
  thinking?: string;
  answer: string;
  pending: boolean;
  incomplete?: true;
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
    return streaming
      ? { thinking: content, answer: '', pending: false }
      : { thinking: content, answer: '', pending: false, incomplete: true };
  }

  return {
    thinking: content.slice(0, closeAt),
    answer: content.slice(closeAt + THINK_CLOSE.length),
    pending: false,
  };
}
