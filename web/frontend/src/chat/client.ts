/*
 * Chat client.
 * Single responsibility: POST a text-only chat request and delegate SSE parsing.
 * It never serializes tools, tool selection, workspace, confirmations, or
 * internal-channel controls.
 */
import { readChatStream } from './stream';
import type { GenerationStats, StreamDelta, WireMessage } from './types';

const FAILED = 'Richiesta non riuscita';

export async function streamAssistant(
  messages: WireMessage[],
  onDelta: (delta: StreamDelta) => void,
  onStats: (stats: GenerationStats) => void,
  signal: AbortSignal
): Promise<void> {
  const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: {
      'content-type': 'application/json'
    },
    body: JSON.stringify({
      messages,
      max_tokens: 1024,
      stream: true
    }),
    signal
  });

  if (!response.ok || !response.body) {
    throw new Error(FAILED);
  }

  await readChatStream(response.body, onDelta, onStats);
}
