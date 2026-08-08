/*
 * Web chat HTTP client.
 * Loads immutable context properties and posts admitted text-only requests.
 * Generation uses a linked internal controller so caller Stop remains distinct
 * from 60-second inactivity cancellation; activity resets that watchdog, which
 * is not a total-generation timeout.
 */
import { parseRuntimeContext } from './context.ts';
import { readChatStream } from './stream.ts';
import type { ContextConfigResult, StreamDelta, WireMessage } from './types';

const FAILED = 'Richiesta non riuscita';
const INTERRUPTED = 'Connessione interrotta';
const INACTIVITY_MS = 60_000;

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
  contextLimit: number,
  onDelta: (delta: StreamDelta) => void,
  signal: AbortSignal
): Promise<void> {
  const controller = new AbortController();
  let cancellation: 'external' | 'timeout' | null = null;
  let timeout: ReturnType<typeof setTimeout>;
  const cancel = (kind: 'external' | 'timeout') => {
    // The first terminal cancellation source owns the observable outcome.
    if (cancellation) return;
    cancellation = kind;
    controller.abort();
  };
  const resetWatchdog = () => {
    clearTimeout(timeout);
    timeout = setTimeout(() => cancel('timeout'), INACTIVITY_MS);
  };
  const stop = () => cancel('external');

  resetWatchdog();
  signal.addEventListener('abort', stop, { once: true });
  if (signal.aborted) stop();
  try {
    const response = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {
        'content-type': 'application/json'
      },
      body: JSON.stringify({
        messages,
        max_tokens: contextLimit,
        stream: true
      }),
      signal: controller.signal
    });

    if (!response.ok || !response.body) {
      throw new Error(FAILED);
    }

    await readChatStream(response.body, onDelta, resetWatchdog);
  } catch (error) {
    if (cancellation === 'timeout') {
      throw new Error(INTERRUPTED);
    }
    throw error;
  } finally {
    clearTimeout(timeout!);
    signal.removeEventListener('abort', stop);
  }
}
