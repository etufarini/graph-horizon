/*
 * Browser prompt-capacity model.
 * Validates the private Web context payload, estimates outgoing Unicode
 * occupancy plus optional remote-context reserve, and admits prompts under a
 * conservative budget. Arithmetic on untrusted data fails closed.
 */
import type { ContextAdmission, ContextConfigResult, ContextUsage, RuntimeContext, WireMessage } from './types';

const MAX_SAFE = Number.MAX_SAFE_INTEGER;

export function parseRuntimeContext(payload: unknown): ContextConfigResult {
  if (!isRecord(payload)) {
    return { ok: false, error: 'unavailable' };
  }
  const contextLimit = payload.context_limit;
  const search = parseSearchCapability(payload.search);
  if (
    !Number.isSafeInteger(contextLimit) ||
    (contextLimit as number) <= 1 ||
    search === null
  ) {
    return { ok: false, error: 'unavailable' };
  }
  const limit = contextLimit as number;
  // Quotient/remainder avoids overflowing a valid safe integer by multiplying it.
  const safePromptBudget = Math.floor(limit / 10) * 9 + Math.floor(((limit % 10) * 9) / 10);
  return {
    ok: true,
    context: {
      contextLimit: limit,
      safePromptBudget,
      search
    }
  };
}

function parseSearchCapability(value: unknown): RuntimeContext['search'] | null {
  if (!isRecord(value) ||
      typeof value.provider !== 'string' || !value.provider || value.provider.length > 100 ||
      !Number.isSafeInteger(value.max_query_characters) ||
      (value.max_query_characters as number) <= 0 ||
      !Number.isSafeInteger(value.max_context_characters) ||
      (value.max_context_characters as number) <= 0) {
    return null;
  }
  return {
    provider: value.provider,
    maxQueryCharacters: value.max_query_characters as number,
    maxContextCharacters: value.max_context_characters as number
  };
}

export function contextUsage(
  messages: WireMessage[],
  context: RuntimeContext,
  reservedCharacters = 0
): ContextUsage {
  return estimate(messages, context, reservedCharacters).usage;
}

export function admitMessages(
  messages: WireMessage[],
  context: RuntimeContext,
  reservedCharacters = 0
): ContextAdmission {
  const assessed = estimate(messages, context, reservedCharacters);
  if (assessed.valid && assessed.estimatedTokens <= context.safePromptBudget) {
    return { ok: true, usage: assessed.usage };
  }
  return {
    ok: false,
    usage: assessed.usage,
    estimatedTokens: assessed.estimatedTokens,
    safePromptBudget: context.safePromptBudget
  };
}

function estimate(
  messages: WireMessage[],
  context: RuntimeContext,
  reservedCharacters: number
): { valid: boolean; estimatedTokens: number; usage: ContextUsage } {
  if (!Number.isSafeInteger(reservedCharacters) || reservedCharacters < 0) {
    return overflowUsage(context.contextLimit);
  }
  let characters = reservedCharacters;
  for (const message of messages) {
    let contentCharacters = 0;
    for (const _codePoint of message.content) {
      contentCharacters += 1;
    }
    characters += contentCharacters;
    if (!Number.isSafeInteger(characters)) {
      return overflowUsage(context.contextLimit);
    }
  }
  const estimatedTokens = Math.floor(characters / 4);
  const percentBig =
    estimatedTokens === 0
      ? BigInt(0)
      : (BigInt(estimatedTokens) * BigInt(100) + BigInt(context.contextLimit) - BigInt(1)) /
        BigInt(context.contextLimit);
  if (percentBig > BigInt(MAX_SAFE)) {
    return overflowUsage(context.contextLimit);
  }
  const percent = Number(percentBig);
  return {
    valid: true,
    estimatedTokens,
    usage: {
      estimatedTokens,
      contextLimit: context.contextLimit,
      percent,
      progress: Math.min(percent, 100)
    }
  };
}

function overflowUsage(contextLimit: number): { valid: false; estimatedTokens: number; usage: ContextUsage } {
  return {
    valid: false,
    estimatedTokens: MAX_SAFE,
    usage: {
      estimatedTokens: MAX_SAFE,
      contextLimit,
      percent: MAX_SAFE,
      progress: 100
    }
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
