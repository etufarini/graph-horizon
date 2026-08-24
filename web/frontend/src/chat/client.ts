/*
 * Web chat HTTP client.
 * Loads immutable context properties and posts admitted text requests with an
 * optional separate Web-search query and browser-local date. Its linked
 * controller keeps caller Stop distinct from inactivity cancellation.
 */
import { parseRuntimeContext } from './context.ts';
import { wireSearch } from './search.ts';
import { readChatStream } from './stream.ts';
import { parseRuntimeInfo } from './telemetry.ts';
import type { ContextConfigResult, RuntimeInfoResult, SearchInput, StreamEvent, WireMessage } from './types';

export const REQUEST_FAILED = 'Request failed';
const INTERRUPTED = 'Connection interrupted';
export const WEB_SEARCH_FAILED = 'Web search unavailable; no answer was generated';
const INACTIVITY_MS = 5 * 60_000;
export const MAX_REQUEST_BYTES = 4 * 1024 * 1024;
const CACHE_KEY = Array.from(globalThis.crypto.getRandomValues(new Uint8Array(16)), byte =>
  byte.toString(16).padStart(2, '0')
).join('');

export async function loadRuntimeContext(signal: AbortSignal): Promise<ContextConfigResult> {
  return loadProperties('/internal/context', parseRuntimeContext, signal);
}

export async function loadRuntimeInfo(signal: AbortSignal): Promise<RuntimeInfoResult> {
  return loadProperties('/internal/runtime', parseRuntimeInfo, signal);
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
  onEvent: (event: StreamEvent) => void,
  signal: AbortSignal,
  search: SearchInput | null = null
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
    const searchRequest = search === null ? null : wireSearch(
      search.terms,
      search.selection
    );
    if (search !== null && searchRequest === null) throw new Error(REQUEST_FAILED);
    const body = JSON.stringify(searchRequest === null
      ? { messages }
      : { messages, search: searchRequest });
    if (new TextEncoder().encode(body).byteLength > MAX_REQUEST_BYTES) {
      throw new Error(REQUEST_FAILED);
    }
    const response = await fetch('/internal/chat', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-graph-horizon-cache': CACHE_KEY
      },
      body,
      signal: controller.signal
    });

    if (response.status === 502) {
      throw new Error(WEB_SEARCH_FAILED);
    }
    if (!response.ok || !response.body) {
      throw new Error(REQUEST_FAILED);
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
