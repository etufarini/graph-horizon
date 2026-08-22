/*
 * Pure telemetry tests cover untrusted runtime payloads, exact bigint memory,
 * cached-prefill usage invariants, zero-duration rates, and display formatting.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  formatBytes,
  liveTelemetry,
  parseGenerationStats,
  parseRuntimeInfo,
  tokensPerSecond
} from './telemetry.ts';

test('absent or completed telemetry is never considered live', () => {
  assert.equal(liveTelemetry(null), null);
  assert.equal(liveTelemetry({ phase: null, phaseStartedAt: null, stats: null }), null);
  const waiting = { phase: 'waiting' as const, phaseStartedAt: 12, stats: null };
  assert.equal(liveTelemetry(waiting), waiting);
});

function memory(total = '18446744073709551615') {
  return {
    weights_bytes: '1', kv_bytes: '2', scratch_bytes: '3', fixed_bytes: '4',
    staging_bytes: '5', crossing_bytes: '6', reserve_bytes: '7', total_bytes: total
  };
}

test('runtime properties preserve exact bytes and bounded model identity', () => {
  const result = parseRuntimeInfo({
    model_name: '  Ministral 3B  ',
    backend: 'vulkan-hybrid',
    memory: { weights_bytes: '2', kv_bytes: '4' },
    placement: {
      mode: 'mixed', cpu_layers: 12, accelerator_layers: 20,
      cpu: memory(), accelerator: memory('0')
    }
  });
  assert.equal(result.ok, true);
  if (!result.ok) return;
  assert.equal(result.info.modelName, 'Ministral 3B');
  assert.deepEqual(result.info.memory, { weights: BigInt(2), kv: BigInt(4) });
  assert.equal(result.info.placement?.cpu.total, BigInt('18446744073709551615'));
  assert.equal(result.info.placement?.accelerator.total, BigInt(0));

  assert.deepEqual(parseRuntimeInfo({
    model_name: null,
    backend: 'cpu',
    memory: { weights_bytes: '11', kv_bytes: '12' },
    placement: null
  }), {
    ok: true,
    info: {
      modelName: 'Local model',
      backend: 'cpu',
      memory: { weights: BigInt(11), kv: BigInt(12) },
      placement: null
    }
  });
});

test('runtime properties reject controls, unknown backends and invalid decimal bytes', () => {
  const valid = {
    model_name: 'model', backend: 'vulkan-hybrid',
    memory: { weights_bytes: '2', kv_bytes: '4' },
    placement: { mode: 'mixed', cpu_layers: 1, accelerator_layers: 1, cpu: memory(), accelerator: memory() }
  };
  for (const changed of [
    { ...valid, model_name: '\nmodel' },
    { ...valid, backend: 'cuda' },
    { ...valid, backend: { toString: () => 'cpu' } },
    { ...valid, memory: { weights_bytes: '01', kv_bytes: '2' } },
    { ...valid, memory: null },
    { ...valid, memory: { weights_bytes: '3', kv_bytes: '4' } },
    { ...valid, placement: { ...valid.placement, cpu: memory('01') } }
  ]) {
    assert.deepEqual(parseRuntimeInfo(changed), { ok: false, error: 'unavailable' });
  }
});

test('generation stats distinguish prompt and cached prefill tokens', () => {
  assert.deepEqual(parseGenerationStats({
    prompt_tokens: 128, prefill_tokens: 32, completion_tokens: 42,
    prefill_ms: 400, decode_ms: 875
  }), {
    promptTokens: 128, prefillTokens: 32, completionTokens: 42,
    prefillMs: 400, decodeMs: 875
  });
  assert.equal(parseGenerationStats({
    prompt_tokens: 2, prefill_tokens: 3, completion_tokens: 1,
    prefill_ms: 1, decode_ms: 1
  }), null);
});

test('rates and byte formatting handle zero and IEC boundaries', () => {
  assert.equal(tokensPerSecond(10, 0), null);
  assert.equal(tokensPerSecond(10, 500), 20);
  assert.equal(formatBytes(BigInt(0)), '0 B');
  assert.equal(formatBytes(BigInt(1024)), '1.0 KiB');
  assert.equal(formatBytes(BigInt(1536)), '1.5 KiB');
});
