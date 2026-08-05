/*
 * graph_horizon_engine — generic backend buffer holders
 * The data shapes the model graph traverses, parameterized over an opaque
 * buffer handle `Buf` so the layout is defined once and every backend decides
 * how to allocate the handles. Describes only the *shape* of the buffers the
 * graph uses (residual/activations scratch, per-layer weights): no concrete GPU
 * API lives here. The per-layer weight shape mirrors the shared dense block;
 * optional Q/K norms are explicit so families without them allocate no fake
 * weights. The
 * per-request KV cache shape (`Kv`) now lives in the `kv_cache` module.
 * Backend-private storage (host-visible logit mirror, shared storage) stays in
 * the backend struct, not in these holders.
*/

// Reusable activation buffers, shared across all layers. All FP16 except `x`,
// the residual stream, which is FP32 (late layers overflow FP16).
pub(crate) struct Scratch<Buf> {
    pub x: Buf,
    pub normed: Buf,
    pub q: Buf,
    pub k: Buf,
    pub v: Buf,
    pub attn: Buf,
    pub proj: Buf,
    pub gate: Buf,
    pub up: Buf,
    pub act: Buf,
    pub ffn_out: Buf,
}

// Per-layer weights, the backend mirror of the sole dense family layout.
pub(crate) struct LayerWeights<Buf> {
    pub attn_norm: Buf,
    pub attn_q: Buf,
    pub attn_k: Buf,
    pub attn_v: Buf,
    pub attn_output: Buf,
    pub ffn_norm: Buf,
    pub ffn_gate: Buf,
    pub ffn_up: Buf,
    pub ffn_down: Buf,
}

// Full weight set: global tensors + one LayerWeights per block. `output`
// (lm_head) is optional: present for chat models, absent for embedding models
// (the GPU mirror of the weight source's tensor list).
pub(crate) struct WeightSet<Buf> {
    pub token_embd: Option<Buf>,
    pub output_norm: Option<Buf>,
    pub output: Option<Buf>,
    pub layers: Vec<LayerWeights<Buf>>,
}

// All persistent forward buffers: weights, scratch and the FP32 logits. The
// backend-private logit readback mirror is NOT here (it stays in the backend).
pub(crate) struct Buffers<Buf> {
    pub weights: WeightSet<Buf>,
    pub scratch: Scratch<Buf>,
    pub logits: Buf,
}
