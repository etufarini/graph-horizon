/*
 * Line-oriented Web SSE parser.
 * Reports every non-empty body chunk as activity, accepts only the bundled
 * backend's content, phase, stats, and done frames, and requires terminal stats
 * before done. Fetch effects remain outside.
 */
import { parseGenerationStats } from './telemetry.ts';
import { parseSearchReport } from './search.ts';
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
  let began = false;
  let searchSeen = false;
  const emit = (event: StreamEvent) => {
    // Usage is terminal telemetry: accepting later deltas would make the
    // rendered response disagree with the exact counters just published.
    if (finished) throw new Error(INTERRUPTED);
    if (event.type === 'search') {
      if (began || searchSeen) throw new Error(INTERRUPTED);
      searchSeen = true;
    } else {
      began = true;
    }
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
        // The done frame wins before later bytes can be interpreted.
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
  const parsed = parseJson(data);
  if (!isRecord(parsed) || 'error' in parsed) {
    throw new Error(INTERRUPTED);
  }

  if (only(parsed, 'phase')) {
    const phase = parsed.phase;
    if (phase !== 'prefill' && phase !== 'decode') throw new Error(INTERRUPTED);
    onEvent({ type: 'phase', phase });
    return false;
  }

  if (only(parsed, 'search')) {
    const search = parseSearchReport(parsed.search);
    if (!search) throw new Error(INTERRUPTED);
    onEvent({ type: 'search', search });
    return false;
  }

  if (only(parsed, 'stats')) {
    const stats = parseGenerationStats(parsed.stats);
    if (!stats) throw new Error(INTERRUPTED);
    onEvent({ type: 'stats', stats });
    return false;
  }

  if (only(parsed, 'content') && typeof parsed.content === 'string') {
    if (parsed.content.length > 0) onEvent({ type: 'content', content: parsed.content });
    return false;
  }

  if (only(parsed, 'done') && parsed.done === true) return true;
  throw new Error(INTERRUPTED);
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

function only(value: Record<string, unknown>, key: string): boolean {
  return Object.keys(value).length === 1 && key in value;
}
