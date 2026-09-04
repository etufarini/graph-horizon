// AGENTS deroga K: kernel della sola operazione YaRN rotary embedding.
#pragma once

extern "C" __global__ void cuda_rope(
    __half *values, uint32_t heads, uint32_t head_dim, uint32_t rope_dim,
    uint32_t position, float freq_base, float factor, float beta_fast,
    float beta_slow, float original_context, float scale) {
    const uint32_t half_rope = rope_dim / 2;
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (half_rope == 0 || id >= heads * half_rope) return;
    const uint32_t head = id / half_rope;
    const uint32_t pair = id % half_rope;
    const uint64_t base = uint64_t(head) * head_dim + pair * 2;
    const float corr_fast = float(rope_dim) * logf(original_context / (beta_fast * 2.0f * CUDART_PI_F))
        / (2.0f * logf(freq_base));
    const float corr_slow = float(rope_dim) * logf(original_context / (beta_slow * 2.0f * CUDART_PI_F))
        / (2.0f * logf(freq_base));
    const float low = fmaxf(floorf(corr_fast), 0.0f);
    const float high = fminf(ceilf(corr_slow), float(rope_dim - 1));
    const float ramp = 1.0f - fminf(fmaxf((float(pair) - low) / fmaxf(high - low, 0.001f), 0.0f), 1.0f);
    const float extrapolated = float(position) * powf(freq_base, -2.0f * float(pair) / float(rope_dim));
    const float theta = (extrapolated / factor) * (1.0f - ramp) + extrapolated * ramp;
    float sine, cosine;
    sincosf(theta, &sine, &cosine);
    const float first = __half2float(values[base]);
    const float second = __half2float(values[base + 1]);
    values[base] = __float2half_rn((first * cosine - second * sine) * scale);
    values[base + 1] = __float2half_rn((first * sine + second * cosine) * scale);
}
