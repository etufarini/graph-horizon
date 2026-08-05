/*
 * graph_horizon_engine — CPU backend host storage
 * CpuBuffer is the CPU counterpart of vulkan's GpuBuffer: host bytes instead of
 * GPU memory. It carries its own CpuFormat (like GpuBuffer.quant) so the kernels
 * pick how to interpret the bytes, and it uses RwLock because kernels write
 * through a shared `&CpuBuffer` reference and the async host (CLI/server) moves
 * `Arc<Engine>` across threads, forcing `Engine: Send + Sync` — which `RefCell`
 * (`!Sync`) cannot satisfy. The lock is never contended (the engine decodes
 * sequentially inside a single `spawn_blocking`); it is only there to satisfy the
 * type bound, and it needs no `unsafe`. The only f16<->f32 conversion of the CPU
 * module lives in the sibling `f16` module (scalar + F16C SIMD), re-exported here so
 * the kernels keep reaching it as `cpu::buffer::{f16_to_f32, …}`.
 * The storage is held behind an `Arc` so a sub-view (CpuBuffer::view) can share
 * the parent's bytes: writes through a view are seen by the parent at the same
 * window and vice versa. The `Arc` only adds a reference count, no new
 * synchronization — the lock stays uncontended (decode/prefill run sequentially
 * in one `spawn_blocking`). Each handle confines every access to its window
 * `[offset, offset + len)`; full buffers have `offset = 0` and `len` = total
 * bytes, so they behave exactly as before.
*/

use std::sync::{Arc, RwLock};

// The f16<->f32 conversion lives in `f16`; re-exported so the impl methods below and
// the kernels (`cpu::buffer::{f16_to_f32, f32_to_f16, f32_to_f16_bytes, …}`) reach it
// at the historic path.
pub(crate) use super::f16::{
    f16_slice_to_f32, f16_to_f32, f32_to_f16, f32_to_f16_bytes, narrow_f32_to_f16,
};

// The tag carried alongside the raw bytes. F32 backs the residual `x` and the
// logits; F16 backs activations and the KV cache; the three quant formats back
// the matmul weights (dequantized on the fly, see `dequant`). The variant names
// mirror ggml's canonical type names (e.g. Q4_K), hence the allow.
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub(crate) enum CpuFormat {
    F16,
    Q4_K,
    Q5_K,
    Q6_K,
    F32,
}

// Host bytes plus their format. Interior mutability: kernels receive their output
// buffer as `&CpuBuffer` and write through the RwLock. `offset`/`len` delimit the
// window of `data` this handle addresses (full buffers: `offset = 0`, `len` =
// total bytes); a view shares the same `Arc` with a sub-window.
pub(crate) struct CpuBuffer {
    data: Arc<RwLock<Vec<u8>>>,
    offset: usize,
    len: usize,
    pub format: CpuFormat,
}

impl CpuBuffer {
    // Zeroed storage of `bytes` bytes (scratch, KV, logits). Length-0 buffers are
    // allowed, mirroring `size.max(1)` on the Vulkan path (the Vec stays empty).
    pub(crate) fn zeroed(bytes: usize, format: CpuFormat) -> CpuBuffer {
        CpuBuffer {
            data: Arc::new(RwLock::new(vec![0u8; bytes])),
            offset: 0,
            len: bytes,
            format,
        }
    }

    // Wraps already-laid-out weight bytes in their original format.
    pub(crate) fn from_bytes(data: Vec<u8>, format: CpuFormat) -> CpuBuffer {
        let len = data.len();
        CpuBuffer {
            data: Arc::new(RwLock::new(data)),
            offset: 0,
            len,
            format,
        }
    }

    // A sub-view sharing this buffer's storage over `[offset_bytes, +len_bytes)`.
    // Clones the `Arc` (shared bytes) and composes the byte origin, so a view of a
    // view accumulates the offsets. No allocation, no copy; same format.
    pub(crate) fn view(&self, offset_bytes: usize, len_bytes: usize) -> CpuBuffer {
        debug_assert!(
            offset_bytes + len_bytes <= self.len,
            "CpuBuffer::view out of bounds: offset + len > window len"
        );
        CpuBuffer {
            data: Arc::clone(&self.data),
            offset: self.offset + offset_bytes,
            len: len_bytes,
            format: self.format,
        }
    }

    // The byte window addressed by this handle. All accessors confine themselves
    // to it; `bytes()` returns the whole guard, so its consumers slice with this.
    pub(crate) fn window(&self) -> std::ops::Range<usize> {
        self.offset..self.offset + self.len
    }

    pub(crate) fn byte_len(&self) -> usize {
        self.len
    }

    // Reads the window as FP32 (the residual `x` / logits).
    pub(crate) fn read_f32(&self) -> Vec<f32> {
        let data = self.data.read().expect("CpuBuffer lock poisoned");
        data[self.window()]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // Reads the window as FP16, widening every element to FP32.
    pub(crate) fn read_f16_as_f32(&self) -> Vec<f32> {
        let data = self.data.read().expect("CpuBuffer lock poisoned");
        f16_slice_to_f32(&data[self.window()])
    }

    // Writes `v` as FP32 little-endian into the first `v.len()*4` bytes of the
    // window. The caller sizes the window to fit (invariant established at alloc
    // time). Consumed by the kernels in m1/m2.
    pub(crate) fn write_f32(&self, v: &[f32]) {
        let mut data = self.data.write().expect("CpuBuffer lock poisoned");
        let win = self.window();
        for (dst, &x) in data[win].chunks_exact_mut(4).zip(v) {
            dst.copy_from_slice(&x.to_le_bytes());
        }
    }

    // Writes `v` as FP16 little-endian (narrowing from FP32) into the first
    // `v.len()*2` bytes of the window. Consumed by the kernels in m1/m2.
    pub(crate) fn write_f16_from_f32(&self, v: &[f32]) {
        let mut data = self.data.write().expect("CpuBuffer lock poisoned");
        let win = self.window();
        narrow_f32_to_f16(v, &mut data[win]);
    }

    // Runs `f` with exclusive access to this buffer's byte window, holding the write
    // lock for its duration. The scoped form lets a caller fill the window in place
    // (e.g. a parallel transpose+narrow) without copying through a temporary Vec; the
    // closure may itself parallelize over disjoint sub-slices of `dst`.
    pub(crate) fn with_bytes_mut<R>(&self, f: impl FnOnce(&mut [u8]) -> R) -> R {
        let mut data = self.data.write().expect("CpuBuffer lock poisoned");
        let win = self.window();
        f(&mut data[win])
    }

    // Raw byte guard for the dequant of the weight buffers (read-only path).
    // Returns the whole shared `Vec` guard; the consumer slices it with
    // `window()` so a view touches only its window (chosen over reshaping the
    // dequant/kernel signatures, which would touch more call-sites).
    pub(crate) fn bytes(&self) -> std::sync::RwLockReadGuard<'_, Vec<u8>> {
        self.data.read().expect("CpuBuffer lock poisoned")
    }

    // Copies `count` FP16 elements from `src` (its window) into this buffer's
    // window starting at element `offset` — a raw 2-bytes-per-element byte copy,
    // no f16<->f32 round trip, so the cached bits are identical to the source
    // (mirror of kv_write.comp). Used by `kv_write`; `offset`/`count` arrive
    // already validated from kv_cache. `offset` is relative to this window.
    pub(crate) fn copy_f16_from(&self, src: &CpuBuffer, offset: usize, count: usize) {
        let src_data = src.data.read().expect("CpuBuffer lock poisoned");
        let mut dst = self.data.write().expect("CpuBuffer lock poisoned");
        let s = src.offset;
        let d = self.offset + offset * 2;
        dst[d..d + count * 2].copy_from_slice(&src_data[s..s + count * 2]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_buffer_read_write() {
        let buf = CpuBuffer::zeroed(3 * 4, CpuFormat::F32);
        buf.write_f32(&[1.0, -2.5, 3.25]);
        assert_eq!(buf.read_f32(), vec![1.0, -2.5, 3.25]);
    }

    #[test]
    fn f16_buffer_read_write() {
        let buf = CpuBuffer::zeroed(2 * 2, CpuFormat::F16);
        buf.write_f16_from_f32(&[1.0, -0.5]);
        assert_eq!(buf.read_f16_as_f32(), vec![1.0, -0.5]);
    }

    // Writing through a view is observed by the parent at the same window; the
    // rest of the parent stays untouched, and the view reads back only its window.
    #[test]
    fn view_write_seen_by_parent() {
        let parent = CpuBuffer::zeroed(4 * 2, CpuFormat::F16); // 4 FP16 elements
        let view = parent.view(2 * 2, 2 * 2); // elements 2..4
        view.write_f16_from_f32(&[1.0, -2.0]);
        assert_eq!(parent.read_f16_as_f32(), vec![0.0, 0.0, 1.0, -2.0]);
        assert_eq!(view.read_f16_as_f32(), vec![1.0, -2.0]);
        assert_eq!(view.byte_len(), 4);
    }

    // Two disjoint views of the same parent write to non-overlapping windows.
    #[test]
    fn disjoint_views_do_not_overlap() {
        let parent = CpuBuffer::zeroed(4 * 2, CpuFormat::F16);
        let lo = parent.view(0, 2 * 2);
        let hi = parent.view(2 * 2, 2 * 2);
        lo.write_f16_from_f32(&[1.0, 2.0]);
        hi.write_f16_from_f32(&[3.0, 4.0]);
        assert_eq!(lo.read_f16_as_f32(), vec![1.0, 2.0]);
        assert_eq!(hi.read_f16_as_f32(), vec![3.0, 4.0]);
        assert_eq!(parent.read_f16_as_f32(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    // A view of a view composes the offsets: inner addresses parent element 2.
    #[test]
    fn view_of_view_composes_offset() {
        let parent = CpuBuffer::zeroed(4 * 2, CpuFormat::F16);
        let outer = parent.view(2, 3 * 2); // parent elements 1..4
        let inner = outer.view(2, 2); // outer element 1 == parent element 2
        inner.write_f16_from_f32(&[7.0]);
        assert_eq!(parent.read_f16_as_f32(), vec![0.0, 0.0, 7.0, 0.0]);
    }
}
