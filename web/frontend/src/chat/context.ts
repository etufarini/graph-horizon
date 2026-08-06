/*
 * Browser context capacity model.
 * Single responsibility: validate `/props`, estimate exact wire-message Unicode
 * code points, and apply checked occupancy and admission arithmetic.
 */
import type { ContextAdmission, ContextConfigResult, ContextUsage, RuntimeContext, WireMessage } from './types';

const MAX_SAFE = Number.MAX_SAFE_INTEGER;

export function parseRuntimeContext(payload: unknown): ContextConfigResult {
  if (!isRecord(payload) || !isRecord(payload.default_generation_settings)) {
    return { ok: false, error: 'unavailable' };
  }
  const { n_ctx: contextLimit, max_tokens: maxTokens } = payload.default_generation_settings;
  if (
    !Number.isSafeInteger(contextLimit) ||
    (contextLimit as number) <= 0 ||
    !Number.isSafeInteger(maxTokens) ||
    (maxTokens as number) <= 0
  ) {
    return { ok: false, error: 'unavailable' };
  }
  const limit = contextLimit as number;
  const generationLimit = maxTokens as number;
  // Quotient/remainder avoids overflowing a valid safe integer by multiplying it.
  const safeTotalBudget = Math.floor(limit / 10) * 9 + Math.floor(((limit % 10) * 9) / 10);
  if (generationLimit >= safeTotalBudget) {
    return { ok: false, error: 'no-prompt-space' };
  }
  return {
    ok: true,
    context: { contextLimit: limit, maxTokens: generationLimit, safeTotalBudget }
  };
}

export function contextUsage(messages: WireMessage[], context: RuntimeContext): ContextUsage {
  return estimate(messages, context).usage;
}

export function admitMessages(messages: WireMessage[], context: RuntimeContext): ContextAdmission {
  const assessed = estimate(messages, context);
  const required = assessed.valid
    ? assessed.estimatedTokens + context.maxTokens
    : Number.POSITIVE_INFINITY;
  if (Number.isSafeInteger(required) && required <= context.safeTotalBudget) {
    return { ok: true, usage: assessed.usage };
  }
  return {
    ok: false,
    usage: assessed.usage,
    estimatedTokens: assessed.estimatedTokens,
    maxTokens: context.maxTokens,
    safeTotalBudget: context.safeTotalBudget
  };
}

function estimate(messages: WireMessage[], context: RuntimeContext): { valid: boolean; estimatedTokens: number; usage: ContextUsage } {
  let characters = 0;
  for (const message of messages) {
    let contentCharacters = 0;
    for (const _codePoint of message.content) {
      contentCharacters += 1;
    }
    characters += contentCharacters;
    if (!Number.isSafeInteger(characters)) {
      return overflowUsage();
    }
  }
  const estimatedTokens = Math.floor(characters / 4);
  const percentBig =
    estimatedTokens === 0
      ? BigInt(0)
      : (BigInt(estimatedTokens) * BigInt(100) + BigInt(context.contextLimit) - BigInt(1)) /
        BigInt(context.contextLimit);
  if (percentBig > BigInt(MAX_SAFE)) {
    return overflowUsage();
  }
  const percent = Number(percentBig);
  return {
    valid: true,
    estimatedTokens,
    usage: { estimatedTokens, percent, progress: Math.min(percent, 100) }
  };
}

function overflowUsage() {
  return {
    valid: false,
    estimatedTokens: MAX_SAFE,
    usage: { estimatedTokens: MAX_SAFE, percent: MAX_SAFE, progress: 100 }
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
