/*
 * Browser inference telemetry
 * Validates immutable Web runtime properties and terminal generation usage,
 * preserves exact allocation bytes as bigint, and derives safe display rates.
 * Fetch, stream framing, stores, and component presentation remain outside.
 */
import type {
  GenerationTelemetry,
  GenerationStats,
  RuntimeInfoResult,
  RuntimeMemory,
  RuntimePlacement
} from './types.ts';

const BACKENDS = new Set(['cpu', 'vulkan', 'vulkan-hybrid', 'metal', 'metal-hybrid']);
const DECIMAL = /^(0|[1-9][0-9]*)$/;

export function liveTelemetry(value: GenerationTelemetry | null): GenerationTelemetry | null {
  return value !== null && value.phase !== null && value.phaseStartedAt !== null ? value : null;
}

export function parseRuntimeInfo(value: unknown): RuntimeInfoResult {
  if (!isRecord(value) || typeof value.backend !== 'string' || !BACKENDS.has(value.backend)) {
    return { ok: false, error: 'unavailable' };
  }
  const modelName = displayName(value.model_name);
  if (value.model_name !== null && modelName === null) {
    return { ok: false, error: 'unavailable' };
  }
  const placement = value.placement === null ? null : parsePlacement(value.placement);
  if (value.placement !== null && placement === null) {
    return { ok: false, error: 'unavailable' };
  }
  return {
    ok: true,
    info: {
      modelName: modelName ?? 'Modello locale',
      backend: value.backend,
      placement
    }
  };
}

export function parseGenerationStats(value: unknown): GenerationStats | null {
  if (!isRecord(value)) return null;
  const promptTokens = count(value.prompt_tokens);
  const prefillTokens = count(value.prefill_tokens);
  const completionTokens = count(value.completion_tokens);
  const prefillMs = count(value.prefill_ms);
  const decodeMs = count(value.decode_ms);
  if (
    promptTokens === null || prefillTokens === null || completionTokens === null ||
    prefillMs === null || decodeMs === null || prefillTokens > promptTokens
  ) {
    return null;
  }
  return { promptTokens, prefillTokens, completionTokens, prefillMs, decodeMs };
}

export function tokensPerSecond(tokens: number, milliseconds: number): number | null {
  return milliseconds > 0 ? tokens * 1000 / milliseconds : null;
}

export function formatBytes(bytes: bigint): string {
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let unit = 0;
  let divisor = BigInt(1);
  while (unit < units.length - 1 && bytes >= divisor * BigInt(1024)) {
    divisor *= BigInt(1024);
    unit += 1;
  }
  if (unit === 0) return `${bytes} B`;
  const tenths = (bytes * BigInt(10) + divisor / BigInt(2)) / divisor;
  return `${tenths / BigInt(10)},${tenths % BigInt(10)} ${units[unit]}`;
}

function parsePlacement(value: unknown): RuntimePlacement | null {
  if (!isRecord(value) || typeof value.mode !== 'string' || value.mode.length === 0) return null;
  const cpuLayers = count(value.cpu_layers);
  const acceleratorLayers = count(value.accelerator_layers);
  const cpu = parseMemory(value.cpu);
  const accelerator = parseMemory(value.accelerator);
  return cpuLayers === null || acceleratorLayers === null || !cpu || !accelerator
    ? null
    : { mode: value.mode, cpuLayers, acceleratorLayers, cpu, accelerator };
}

function parseMemory(value: unknown): RuntimeMemory | null {
  if (!isRecord(value)) return null;
  const read = (key: string): bigint | null => {
    const raw = value[key];
    return typeof raw === 'string' && DECIMAL.test(raw) ? BigInt(raw) : null;
  };
  const values = [
    read('weights_bytes'), read('kv_bytes'), read('scratch_bytes'), read('fixed_bytes'),
    read('staging_bytes'), read('crossing_bytes'), read('reserve_bytes'), read('total_bytes')
  ];
  if (values.some(item => item === null)) return null;
  const [weights, kv, scratch, fixed, staging, crossing, reserve, total] = values as bigint[];
  return { weights, kv, scratch, fixed, staging, crossing, reserve, total };
}

function displayName(value: unknown): string | null {
  if (value === null) return null;
  if (typeof value !== 'string' || /[\p{Cc}\p{Cf}]/u.test(value)) return null;
  const name = value.trim();
  return name.length > 0 && Array.from(name).length <= 128
    ? name
    : null;
}

function count(value: unknown): number | null {
  return Number.isSafeInteger(value) && (value as number) >= 0 ? value as number : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
