/*
 * Line-oriented Web SSE parser.
 * Reports every non-empty body chunk as activity, accepts content, phase, exact
 * usage, and final frames, rejects prohibited protocol data, and requires
 * terminal usage before `[DONE]`. Fetch effects remain outside.
 */
import { parseGenerationStats } from './telemetry.ts';
import type { StreamEvent } from './types';

const INTERRUPTED = 'Connection interrupted';

export async function readChatStream(
  body: ReadableStream<Uint8Array>,
  onEvent: (event: StreamEvent) => void,
  onActivity: () => void = () => {}
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finished = false;
  const emit = (event: StreamEvent) => {
    // Usage is terminal telemetry: accepting later deltas would make the
    // rendered response disagree with the exact counters just published.
    if (finished) throw new Error(INTERRUPTED);
    if (event.type === 'stats') finished = true;
    onEvent(event);
  };
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value.byteLength > 0) onActivity();
      buffer += decoder.decode(value, { stream: true });
      const consumed = consumeLines(buffer, emit);
      if (consumed.done) {
        if (!finished) throw new Error(INTERRUPTED);
        // `[DONE]` wins before any bytes that follow it can be interpreted.
        await reader.cancel().catch(() => {});
        return;
      }
      buffer = consumed.rest;
    }

    buffer += decoder.decode();
    if (consumeLines(`${buffer}\n`, emit).done && finished) {
      return;
    }
    throw new Error(INTERRUPTED);
  } finally {
    reader.releaseLock();
  }
}

function consumeLines(
  buffer: string,
  onEvent: (event: StreamEvent) => void
): { rest: string; done: boolean } {
  const lines = buffer.split(/\r?\n/);
  const rest = lines.pop() ?? '';
  for (const line of lines) {
    if (consumeDataLine(line, onEvent)) {
      return { rest: '', done: true };
    }
  }
  return { rest, done: false };
}

function consumeDataLine(
  line: string,
  onEvent: (event: StreamEvent) => void
): boolean {
  if (!line.startsWith('data:')) {
    return false;
  }

  const data = line.slice(5).trimStart();
  if (data === '[DONE]') {
    return true;
  }

  const parsed = parseJson(data);
  if (!isRecord(parsed) || 'error' in parsed || 'tool_event' in parsed) {
    throw new Error(INTERRUPTED);
  }

  if (isRecord(parsed.graph_horizon)) {
    const phase = parsed.graph_horizon.phase;
    if (phase !== 'prefill' && phase !== 'decode') throw new Error(INTERRUPTED);
    onEvent({ type: 'phase', phase });
    return false;
  }

  if ('usage' in parsed) {
    const stats = parseGenerationStats(parsed.usage);
    if (!stats) throw new Error(INTERRUPTED);
    onEvent({ type: 'stats', stats });
    return false;
  }

  const choice = Array.isArray(parsed.choices) ? parsed.choices[0] : undefined;
  const delta = isRecord(choice) && isRecord(choice.delta) ? choice.delta : undefined;
  if (delta && ('reasoning_content' in delta || 'tool_calls' in delta)) {
    throw new Error(INTERRUPTED);
  }
  if (typeof delta?.content === 'string' && delta.content.length > 0) {
    onEvent({ type: 'content', content: delta.content });
  }

  return false;
}

function parseJson(data: string): unknown {
  try {
    return JSON.parse(data);
  } catch {
    return null;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
