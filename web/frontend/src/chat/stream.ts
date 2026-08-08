/*
 * Line-oriented Web SSE parser.
 * Reports every non-empty body chunk as activity, accepts content plus neutral
 * usage/final frames, rejects error/tool/separate-reasoning fields, and requires
 * an immediate terminal `[DONE]`. Fetch and lifecycle effects remain outside.
 */
import type { StreamDelta } from './types';

const INTERRUPTED = 'Connessione interrotta';

export async function readChatStream(
  body: ReadableStream<Uint8Array>,
  onDelta: (delta: StreamDelta) => void,
  onActivity: () => void = () => {}
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value.byteLength > 0) onActivity();
      buffer += decoder.decode(value, { stream: true });
      const consumed = consumeLines(buffer, onDelta);
      if (consumed.done) {
        // `[DONE]` wins before any bytes that follow it can be interpreted.
        await reader.cancel().catch(() => {});
        return;
      }
      buffer = consumed.rest;
    }

    buffer += decoder.decode();
    if (consumeLines(`${buffer}\n`, onDelta).done) {
      return;
    }
    throw new Error(INTERRUPTED);
  } finally {
    reader.releaseLock();
  }
}

function consumeLines(
  buffer: string,
  onDelta: (delta: StreamDelta) => void
): { rest: string; done: boolean } {
  const lines = buffer.split(/\r?\n/);
  const rest = lines.pop() ?? '';
  for (const line of lines) {
    if (consumeDataLine(line, onDelta)) {
      return { rest: '', done: true };
    }
  }
  return { rest, done: false };
}

function consumeDataLine(
  line: string,
  onDelta: (delta: StreamDelta) => void
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

  const choice = Array.isArray(parsed.choices) ? parsed.choices[0] : undefined;
  const delta = isRecord(choice) && isRecord(choice.delta) ? choice.delta : undefined;
  if (delta && ('reasoning_content' in delta || 'tool_calls' in delta)) {
    throw new Error(INTERRUPTED);
  }
  if (typeof delta?.content === 'string' && delta.content.length > 0) {
    onDelta({ content: delta.content });
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
