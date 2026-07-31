/*
 * Chat stream parser.
 * Single responsibility: parse text/status/error SSE frames from the backend.
 * It treats tool or reasoning activity as a protocol error instead of exposing
 * it to the UI.
 */
import type { GenerationStats, StreamDelta } from './types';

const INTERRUPTED = 'Connessione interrotta';

export async function readChatStream(
  body: ReadableStream<Uint8Array>,
  onDelta: (delta: StreamDelta) => void,
  onStats: (stats: GenerationStats) => void
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let doneSeen = false;

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    buffer = consumeLines(buffer, line => {
      if (consumeDataLine(line, onDelta, onStats)) {
        doneSeen = true;
      }
    });
  }

  buffer += decoder.decode();
  consumeLines(`${buffer}\n`, line => {
    if (consumeDataLine(line, onDelta, onStats)) {
      doneSeen = true;
    }
  });

  if (!doneSeen) {
    throw new Error(INTERRUPTED);
  }
}

function consumeLines(buffer: string, consume: (line: string) => void): string {
  const lines = buffer.split(/\r?\n/);
  const rest = lines.pop() ?? '';
  for (const line of lines) {
    consume(line);
  }
  return rest;
}

function consumeDataLine(
  line: string,
  onDelta: (delta: StreamDelta) => void,
  onStats: (stats: GenerationStats) => void
): boolean {
  if (!line.startsWith('data:')) {
    return false;
  }

  const data = line.slice(5).trimStart();
  if (data === '[DONE]') {
    return true;
  }

  const parsed = parseJson(data);
  if (!parsed || 'error' in parsed || 'tool_event' in parsed) {
    throw new Error(INTERRUPTED);
  }

  const delta = parsed.choices?.[0]?.delta;
  if (delta?.reasoning_content !== undefined || delta?.['tool_' + 'calls'] !== undefined) {
    throw new Error(INTERRUPTED);
  }
  if (typeof delta?.content === 'string' && delta.content.length > 0) {
    onDelta({ content: delta.content });
  }

  const stats = parseUsage(parsed.usage);
  if (stats) {
    onStats(stats);
  }
  return false;
}

function parseUsage(usage: unknown): GenerationStats | null {
  if (typeof usage !== 'object' || usage === null || Array.isArray(usage)) {
    return null;
  }
  const wire = usage as Record<string, unknown>;
  const fields = ['prompt_tokens', 'completion_tokens', 'prefill_ms', 'decode_ms'];
  const values = fields.map(field => wire[field]);
  if (!values.every(value => typeof value === 'number' && Number.isFinite(value) && value >= 0)) {
    return null;
  }
  const [promptTokens, completionTokens, prefillMs, decodeMs] = values as number[];
  return { promptTokens, completionTokens, prefillMs, decodeMs };
}

function parseJson(data: string): any | null {
  try {
    return JSON.parse(data);
  } catch {
    return null;
  }
}
