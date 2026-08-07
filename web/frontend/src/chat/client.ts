/*
 * Chat client.
 * Single responsibility: load immutable context properties and POST admitted
 * text-only chat requests while delegating SSE parsing.
 */
import { parseRuntimeContext } from './context';
import { readChatStream } from './stream';
import type { ContextConfigResult, StreamDelta, WireMessage } from './types';

const FAILED = 'Richiesta non riuscita';

export async function loadRuntimeContext(signal: AbortSignal): Promise<ContextConfigResult> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal.addEventListener('abort', abort, { once: true });
  const timeout = setTimeout(abort, 3000);
  try {
    const response = await fetch('/props', { signal: controller.signal });
    if (!response.ok) {
      return { ok: false, error: 'unavailable' };
    }
    return parseRuntimeContext(await response.json());
  } catch {
    return { ok: false, error: 'unavailable' };
  } finally {
    clearTimeout(timeout);
    signal.removeEventListener('abort', abort);
  }
}

export async function streamAssistant(
  messages: WireMessage[],
  maxTokens: number,
  onDelta: (delta: StreamDelta) => void,
  signal: AbortSignal
): Promise<void> {
  const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: {
      'content-type': 'application/json'
    },
    body: JSON.stringify({
      messages,
      max_tokens: maxTokens,
      stream: true
    }),
    signal
  });

  if (!response.ok || !response.body) {
    throw new Error(FAILED);
  }

  await readChatStream(response.body, onDelta);
}
