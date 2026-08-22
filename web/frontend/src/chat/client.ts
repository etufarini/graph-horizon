/*
 * Web chat HTTP client.
 * Loads immutable context properties and posts admitted text-only requests.
 * Generation uses a linked internal controller so caller Stop remains distinct
 * from five-minute inactivity cancellation; activity resets that watchdog, which
 * is not a total-generation timeout.
 */
import { parseRuntimeContext } from './context.ts';
import { readChatStream } from './stream.ts';
import { parseRuntimeInfo } from './telemetry.ts';
import type { ContextConfigResult, RuntimeInfoResult, StreamEvent, WireMessage } from './types';

const FAILED = 'Request failed';
const INTERRUPTED = 'Connection interrupted';
const INACTIVITY_MS = 5 * 60_000;
export const MAX_REQUEST_BYTES = 4 * 1024 * 1024;
const CACHE_KEY = Array.from(globalThis.crypto.getRandomValues(new Uint8Array(16)), byte =>
  byte.toString(16).padStart(2, '0')
).join('');

export async function loadRuntimeContext(signal: AbortSignal): Promise<ContextConfigResult> {
  return loadProperties('/props', parseRuntimeContext, signal);
}

export async function loadRuntimeInfo(signal: AbortSignal): Promise<RuntimeInfoResult> {
  return loadProperties('/runtime', parseRuntimeInfo, signal);
}

async function loadProperties<T>(
  url: string,
  parse: (value: unknown) => T,
  signal: AbortSignal
): Promise<T | { ok: false; error: 'unavailable' }> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal.addEventListener('abort', abort, { once: true });
  const timeout = setTimeout(abort, 3000);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      return { ok: false, error: 'unavailable' };
    }
    return parse(await response.json());
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
  onEvent: (event: StreamEvent) => void,
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
    const body = JSON.stringify({
      messages,
      max_tokens: contextLimit,
      stream: true
    });
    if (new TextEncoder().encode(body).byteLength > MAX_REQUEST_BYTES) {
      throw new Error(FAILED);
    }
    const response = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-graph-horizon-cache': CACHE_KEY
      },
      body,
      signal: controller.signal
    });

    if (!response.ok || !response.body) {
      throw new Error(FAILED);
    }

    await readChatStream(response.body, onEvent, resetWatchdog);
  } catch (error) {
    if (cancellation === 'timeout') {
      throw new Error(INTERRUPTED);
    }
    if (cancellation === 'external') {
      throw new DOMException('Aborted', 'AbortError');
    }
    throw error;
  } finally {
    clearTimeout(timeout!);
    signal.removeEventListener('abort', stop);
  }
}
