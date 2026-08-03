/*
 * gh_zero_engine — Vulkan module boundary
 * Owns the concrete backend state and exposes the Vulkan domains for
 * initialization, memory, execution, pipelines, and kernels. Loading and trait
 * delegation stay in their dedicated modules; hybrid policy stays in the
 * feature-gated Mistral domain.
 */

#![deny(clippy::undocumented_unsafe_blocks)]

pub(crate) mod backend;
mod exec;
mod init;
mod loader;
mod mem;

pub(crate) mod coopmat;
pub(crate) mod kernels;
pub(crate) mod pipeline;

use crate::backend::buffers::Buffers;
use buffers::GpuBuffer;
#[cfg(not(feature = "vulcan-hybrid"))]
use init::device::Device;
use pipeline::PipelineRegistry;

// Persistent resources owned by one Vulkan model backend.
pub(crate) struct VulkanBackend {
    pub(crate) dev: Device,
    pub(crate) reg: PipelineRegistry,
    pub(crate) buf: Buffers<GpuBuffer>,
    pub(crate) logits_host: GpuBuffer,
    pub(crate) reduce: GpuBuffer,
    pub(crate) mmvq_qs: GpuBuffer,
    pub(crate) mmvq_ds: GpuBuffer,
}

impl VulkanBackend {
    // Raw test read-back is concrete Vulkan tooling, not part of the graph
    // backend contract.
    #[cfg(test)]
    pub(crate) fn read_bytes(&self, buf: &GpuBuffer, bytes: usize) -> color_eyre::Result<Vec<u8>> {
        exec::readback::read_bytes(self, buf, bytes)
    }

    // The hybrid residual boundary and Vulkan tests share the loader's bounded
    // staging upload primitive.
    #[cfg(any(test, feature = "vulcan-hybrid"))]
    pub(crate) fn upload_bytes(&self, buf: &GpuBuffer, data: &[u8]) -> color_eyre::Result<()> {
        buf.upload(&self.dev, data)
    }
}

// Upper activation width covered by the persistent Q8_1 MMVQ scratch.
pub(crate) const MMVQ_SCRATCH_IN_DIM: u64 = 32768;

#[cfg(test)]
thread_local! {
    static SUBMIT_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_submit_count() {
    SUBMIT_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn submit_count() -> usize {
    SUBMIT_COUNT.with(std::cell::Cell::get)
}

#[cfg(test)]
fn record_submit() {
    SUBMIT_COUNT.with(|count| count.set(count.get() + 1));
}

pub(crate) use init::device;
#[cfg(feature = "vulcan-hybrid")]
pub(crate) use init::device::Device;
#[cfg(feature = "vulcan-hybrid")]
pub(crate) use init::hybrid_device;
#[cfg(all(test, feature = "vulcan-hybrid"))]
pub(crate) use init::{probe_count, reset_probe_count};
#[cfg(feature = "vulcan-hybrid")]
pub(crate) use mem::budget::vram_for_auto;
#[cfg(not(feature = "vulcan-hybrid"))]
pub(crate) use mem::budget::{set_reserve_mib, set_weights_percent};
pub(crate) use mem::{buffers, weights};

#[cfg(test)]
pub(crate) mod fault {
    use std::cell::{Cell, RefCell};

    use color_eyre::eyre::{Result, bail};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Point {
        Initialization,
        Allocation,
        Pipeline,
        Record,
        Submit,
        Transfer,
        Readback,
    }

    thread_local! {
        static ARMED: Cell<Option<Point>> = const { Cell::new(None) };
        static LIVE: RefCell<[usize; 7]> = const { RefCell::new([0; 7]) };
    }

    pub(crate) struct Tracked(Point);

    impl Drop for Tracked {
        fn drop(&mut self) {
            LIVE.with(|live| live.borrow_mut()[self.0 as usize] -= 1);
        }
    }

    pub(crate) fn track(point: Point) -> Tracked {
        LIVE.with(|live| live.borrow_mut()[point as usize] += 1);
        Tracked(point)
    }

    pub(crate) fn arm(point: Point) {
        ARMED.with(|armed| armed.set(Some(point)));
    }

    pub(crate) fn hit(point: Point) -> Result<()> {
        let fail = ARMED.with(|armed| {
            let fail = armed.get() == Some(point);
            if fail {
                armed.set(None);
            }
            fail
        });
        if fail {
            bail!("vulkan: injected test failure");
        }
        Ok(())
    }

    pub(crate) fn live() -> [usize; 7] {
        LIVE.with(|live| *live.borrow())
    }
}

#[cfg(test)]
mod format_coverage {
    use crate::backend::Backend;

    use super::VulkanBackend;
    use super::buffers::WeightFormat;

    #[test]
    fn retained_quant_formats_execute_dense_smoke_when_device_is_available() {
        let backend = match VulkanBackend::bare() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("external verification: no Vulkan device");
                return;
            }
        };
        let in_dim = 256usize;
        let out_dim = 2usize;
        let activation: Vec<f32> = (0..in_dim).map(|i| i as f32 / 256.0 - 0.5).collect();
        let activation_bytes = f32_slice_to_f16_bytes(&activation);
        let a = backend
            .alloc_buffer(activation_bytes.len() as u64)
            .expect("activation");
        backend
            .upload_bytes(&a, &activation_bytes)
            .expect("upload activation");

        for format in [WeightFormat::Q4K, WeightFormat::Q5K, WeightFormat::Q6K] {
            let bytes = quant_weight_bytes(format, out_dim);
            let mut w = backend.alloc_buffer(bytes.len() as u64).expect("weight");
            w.quant = format;
            backend.upload_bytes(&w, &bytes).expect("upload weight");

            let x = backend
                .alloc_buffer((in_dim * 4) as u64)
                .expect("embed out");
            let enc = backend.begin().expect("begin embed");
            backend
                .embed(&enc, &x, &w, 0, in_dim as u32)
                .expect("embed");
            backend.submit(enc).expect("submit embed");
            assert_finite_f32(&backend.read_bytes(&x, in_dim * 4).expect("read embed"));
            backend.free_buffer(x);

            let y = backend
                .alloc_buffer((out_dim * 2) as u64)
                .expect("matmul out");
            let enc = backend.begin().expect("begin matmul");
            backend.matmul(&enc, &y, &a, &w, in_dim as u32, out_dim as u32);
            backend.submit(enc).expect("submit matmul");
            assert_finite_f16(&backend.read_bytes(&y, out_dim * 2).expect("read matmul"));
            backend.free_buffer(y);

            let logits = backend
                .alloc_buffer((out_dim * 4) as u64)
                .expect("logits out");
            let enc = backend.begin().expect("begin logits");
            backend.logits(&enc, &logits, &a, &w, in_dim as u32, out_dim as u32);
            backend.submit(enc).expect("submit logits");
            assert_finite_f32(
                &backend
                    .read_bytes(&logits, out_dim * 4)
                    .expect("read logits"),
            );
            backend.free_buffer(logits);
            backend.free_buffer(w);
        }
        backend.free_buffer(a);
    }

    fn quant_weight_bytes(format: WeightFormat, out_dim: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for row in 0..out_dim {
            match format {
                WeightFormat::Q4K => out.extend(q4_block(row)),
                WeightFormat::Q5K => out.extend(q5_block(row)),
                WeightFormat::Q6K => out.extend(q6_block(row)),
                WeightFormat::F16 => unreachable!("quant smoke only covers quantized formats"),
            }
        }
        out
    }

    fn q4_block(seed: usize) -> Vec<u8> {
        let mut b = vec![0u8; 144];
        b[0..2].copy_from_slice(&f16le(0.03));
        for (i, v) in b[4..16].iter_mut().enumerate() {
            *v = if i < 8 { 1 } else { 0 };
        }
        for (i, v) in b[16..144].iter_mut().enumerate() {
            *v = ((i + seed) as u8 & 0x0f) | ((((i + seed + 3) as u8) & 0x0f) << 4);
        }
        b
    }

    fn q5_block(seed: usize) -> Vec<u8> {
        let mut b = vec![0u8; 176];
        b[0..2].copy_from_slice(&f16le(0.03));
        for (i, v) in b[4..16].iter_mut().enumerate() {
            *v = if i < 8 { 1 } else { 0 };
        }
        for (i, v) in b[16..48].iter_mut().enumerate() {
            *v = if (i + seed).is_multiple_of(3) {
                0x55
            } else {
                0
            };
        }
        for (i, v) in b[48..176].iter_mut().enumerate() {
            *v = ((i + seed) as u8 & 0x0f) | ((((i + seed + 5) as u8) & 0x0f) << 4);
        }
        b
    }

    fn q6_block(seed: usize) -> Vec<u8> {
        let mut b = vec![0u8; 210];
        for (i, v) in b[0..192].iter_mut().enumerate() {
            *v = (i + seed) as u8;
        }
        for v in &mut b[192..208] {
            *v = 1;
        }
        b[208..210].copy_from_slice(&f16le(0.01));
        b
    }

    fn assert_finite_f16(bytes: &[u8]) {
        let values: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        assert!(values.iter().all(|v| v.is_finite()), "{values:?}");
        assert!(values.iter().any(|v| v.abs() > 0.0), "{values:?}");
    }

    fn assert_finite_f32(bytes: &[u8]) {
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!(values.iter().all(|v| v.is_finite()), "{values:?}");
        assert!(values.iter().any(|v| v.abs() > 0.0), "{values:?}");
    }

    fn f32_slice_to_f16_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for &v in values {
            out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
        }
        out
    }

    fn f16le(v: f32) -> [u8; 2] {
        f32_to_f16(v).to_le_bytes()
    }

    fn f32_to_f16(f: f32) -> u16 {
        let bits = f.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mant = bits & 0x7f_ffff;
        if exp == 0xff {
            return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
        }
        let e = exp - 127 + 15;
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        if e <= 0 {
            if e < -10 {
                return sign;
            }
            let m = mant | 0x80_0000;
            let shift = (14 - e) as u32;
            let half = (m >> shift) as u16;
            let rem = m & ((1 << shift) - 1);
            let round =
                u16::from(rem > (1 << (shift - 1)) || rem == (1 << (shift - 1)) && half & 1 == 1);
            return sign | (half + round);
        }
        let half = (e as u16) << 10 | (mant >> 13) as u16;
        let rem = mant & 0x1fff;
        let round = u16::from(rem > 0x1000 || rem == 0x1000 && half & 1 == 1);
        sign | (half + round)
    }

    fn f16_to_f32(h: u16) -> f32 {
        let sign = ((h >> 15) & 1) as u32;
        let exp = ((h >> 10) & 0x1f) as u32;
        let mant = (h & 0x3ff) as u32;
        let bits = if exp == 0 {
            if mant == 0 {
                sign << 31
            } else {
                let mut e = -1i32;
                let mut m = mant;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x3ff;
                let exp32 = (127 - 15 + 2 + e) as u32;
                (sign << 31) | (exp32 << 23) | (m << 13)
            }
        } else if exp == 0x1f {
            (sign << 31) | (0xff << 23) | (mant << 13)
        } else {
            let exp32 = exp + (127 - 15);
            (sign << 31) | (exp32 << 23) | (mant << 13)
        };
        f32::from_bits(bits)
    }
}

#[cfg(test)]
mod rope {
    use crate::backend::Backend;
    use crate::backend::rope::{RopeRole, Yarn};
    use crate::backend::vulkan::VulkanBackend;
    use crate::backend::vulkan::kernels::elementwise;

    fn yarn() -> Yarn {
        Yarn {
            rope_dim: 8,
            original_context: 128,
            freq_base: 1_000_000.0,
            factor: 8.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            log_multiplier: 0.1,
            q_temperature_scale: 1.25,
        }
    }

    fn input() -> Vec<f32> {
        (0..16).map(|i| i as f32 / 8.0 - 1.0).collect()
    }

    fn cpu_rope(y: &Yarn, role: RopeRole, pos: usize, x: &[f32]) -> Vec<f32> {
        let mut out = x.to_vec();
        let heads = 2usize;
        let head_dim = 8usize;
        let half = y.rope_dim / 2;
        for head in 0..heads {
            let base = head * head_dim;
            for pair in 0..half {
                let p = y.pair(role, pair, pos).expect("valid yarn pair");
                let first = base + pair * 2;
                let second = first + 1;
                let a = out[first];
                let b = out[second];
                let scale = y.post_scale(role, pos);
                out[first] = (a * p.cos - b * p.sin) * scale;
                out[second] = (a * p.sin + b * p.cos) * scale;
            }
        }
        out
    }

    #[test]
    fn query_temperature_scale_changes_only_query_role() {
        let y = yarn();
        assert_eq!(y.post_scale(RopeRole::Key, 512), 1.0);
        assert_eq!(y.post_scale(RopeRole::Query, 0), 1.0);
        assert!(y.post_scale(RopeRole::Query, 128) > 1.0);
    }

    #[test]
    fn vulkan_yarn_rope_matches_cpu_oracle_when_device_is_available() {
        let backend = match VulkanBackend::bare() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("external verification: no Vulkan device");
                return;
            }
        };
        let y = yarn();
        for (role, pos) in [
            (RopeRole::Key, 64usize),
            (RopeRole::Query, 128usize),
            (RopeRole::Query, 512usize),
        ] {
            let rounded_input: Vec<f32> = input()
                .into_iter()
                .map(|v| f16_to_f32(f32_to_f16(v)))
                .collect();
            let raw = f32_slice_to_f16_bytes(&rounded_input);
            let buf = backend.alloc_buffer(raw.len() as u64).expect("rope buffer");
            backend.upload_bytes(&buf, &raw).expect("upload rope input");
            let enc = backend.begin().expect("begin rope");
            elementwise::rope_yarn(
                &backend.dev,
                &backend.reg,
                enc,
                &buf,
                2,
                8,
                y.rope_dim as u32,
                pos as u32,
                y.freq_base,
                y.factor,
                y.beta_fast,
                y.beta_slow,
                y.original_context as u32,
                y.post_scale(role, pos),
            );
            backend.submit(enc).expect("submit rope");
            let got_raw = backend.read_bytes(&buf, raw.len()).expect("read rope");
            let got: Vec<f32> = got_raw
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let want = cpu_rope(&y, role, pos, &rounded_input);
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                let tol = 2e-3 + 2e-3 * w.abs();
                assert!(
                    (*g - w).abs() <= tol,
                    "role={:?} pos={pos} i={i} got={g} want={w} tol={tol}",
                    role as u8
                );
            }
            backend.free_buffer(buf);
        }
    }

    fn f32_slice_to_f16_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for &v in values {
            out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
        }
        out
    }

    fn f32_to_f16(f: f32) -> u16 {
        let bits = f.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mant = bits & 0x7f_ffff;
        if exp == 0xff {
            return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
        }
        let e = exp - 127 + 15;
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        if e <= 0 {
            if e < -10 {
                return sign;
            }
            let m = mant | 0x80_0000;
            let shift = (14 - e) as u32;
            let half = (m >> shift) as u16;
            let rem = m & ((1 << shift) - 1);
            let round =
                u16::from(rem > (1 << (shift - 1)) || rem == (1 << (shift - 1)) && half & 1 == 1);
            return sign | (half + round);
        }
        let half = (e as u16) << 10 | (mant >> 13) as u16;
        let rem = mant & 0x1fff;
        let round = u16::from(rem > 0x1000 || rem == 0x1000 && half & 1 == 1);
        sign | (half + round)
    }

    fn f16_to_f32(h: u16) -> f32 {
        let sign = ((h >> 15) & 1) as u32;
        let exp = ((h >> 10) & 0x1f) as u32;
        let mant = (h & 0x3ff) as u32;
        let bits = if exp == 0 {
            if mant == 0 {
                sign << 31
            } else {
                let mut e = -1i32;
                let mut m = mant;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x3ff;
                let exp32 = (127 - 15 + 2 + e) as u32;
                (sign << 31) | (exp32 << 23) | (m << 13)
            }
        } else if exp == 0x1f {
            (sign << 31) | (0xff << 23) | (mant << 13)
        } else {
            let exp32 = exp + (127 - 15);
            (sign << 31) | (exp32 << 23) | (mant << 13)
        };
        f32::from_bits(bits)
    }
}

#[cfg(test)]
mod vulkan_failure_lifecycle {
    use crate::backend::Backend;
    use crate::backend::vulkan::VulkanBackend;

    #[test]
    fn real_owner_failpoints_release_acquired_resources() {
        use super::fault::{self, Point};

        for point in [Point::Initialization, Point::Pipeline, Point::Allocation] {
            fault::arm(point);
            let error = VulkanBackend::bare()
                .err()
                .expect("injected backend construction failure");
            assert_eq!(error.to_string(), "vulkan: injected test failure");
            assert_eq!(fault::live(), [0; 7], "{point:?}");
        }

        let backend = match VulkanBackend::bare() {
            Ok(backend) => backend,
            Err(_) => {
                eprintln!("external verification: no Vulkan device");
                return;
            }
        };
        let buffer = backend.alloc_buffer(8).expect("buffer");
        for point in [Point::Record, Point::Transfer, Point::Readback] {
            fault::arm(point);
            let error = match point {
                Point::Record => backend.begin().expect_err("record failpoint"),
                Point::Transfer => backend
                    .upload_bytes(&buffer, &[1, 2, 3, 4])
                    .expect_err("transfer failpoint"),
                Point::Readback => backend
                    .read_bytes(&buffer, 4)
                    .expect_err("readback failpoint"),
                _ => unreachable!(),
            };
            assert_eq!(error.to_string(), "vulkan: injected test failure");
            assert_eq!(fault::live(), [0; 7], "{point:?}");
        }

        let command = backend.begin().expect("command before submit failpoint");
        fault::arm(Point::Submit);
        let error = backend.submit(command).expect_err("submit failpoint");
        assert_eq!(error.to_string(), "vulkan: injected test failure");
        assert_eq!(fault::live(), [0; 7]);
        backend.free_buffer(buffer);
    }

    #[test]
    fn readback_and_reduce_validation_fail_before_backend_submit() {
        let backend = match VulkanBackend::bare() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("external verification: no Vulkan device");
                return;
            }
        };
        let buf = backend.alloc_buffer(4).expect("buffer");
        let view = backend.view(&buf, 0, 4);

        super::reset_submit_count();
        let err = backend.read_bytes(&view, 8).expect_err("bounded readback");
        assert_eq!(err.to_string(), "vulkan: read_bytes past buffer window");
        assert_eq!(super::submit_count(), 0);

        super::reset_submit_count();
        let err = backend.read_argmax(&buf, 0).expect_err("empty argmax");
        assert_eq!(err.to_string(), "vulkan: read_argmax on empty logits");
        assert_eq!(super::submit_count(), 0);

        backend.free_buffer(buf);
    }

    #[test]
    fn transfer_validation_and_cancel_boundary_do_not_submit_work() {
        let backend = match VulkanBackend::bare() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("external verification: no Vulkan device");
                return;
            }
        };
        let buf = backend.alloc_buffer(4).expect("buffer");
        let view = backend.view(&buf, 0, 4);

        super::reset_submit_count();
        let err = backend
            .upload_bytes(&view, &[1, 2, 3, 4, 5])
            .expect_err("bounded upload");
        assert_eq!(err.to_string(), "vulkan: upload larger than buffer");
        assert_eq!(super::submit_count(), 0);

        super::reset_submit_count();
        let cancelled = true;
        if !cancelled {
            let enc = backend.begin().expect("begin");
            backend.submit(enc).expect("submit");
        }
        assert_eq!(super::submit_count(), 0);

        backend.free_buffer(buf);
    }
}

#[cfg(test)]
mod cpu_vulkan_parity {
    use crate::backend::Backend;
    use crate::backend::rope::{RopeRole, Yarn};
    use crate::backend::vulkan::VulkanBackend;
    use crate::backend::vulkan::kernels::elementwise;
    use crate::gguf::metadata::ModelMetadata;
    use crate::kv_cache;
    use crate::kv_cache::scheme::KvQuant;

    #[test]
    fn cpu_and_vulkan_match_dense_greedy_id_and_intermediate() {
        let backend = match VulkanBackend::bare() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("external verification: no Vulkan device");
                dense_host_oracle_smoke();
                return;
            }
        };
        let hidden = 8usize;
        let vocab = 4usize;
        let token = 2u32;
        let token_embd = f16_matrix_bytes(&[
            [-0.4, 0.1, 0.0, 0.2, -0.1, 0.3, 0.2, -0.2],
            [0.2, -0.2, 0.3, -0.3, 0.4, -0.1, 0.1, 0.0],
            [0.6, 0.5, -0.2, 0.1, 0.3, -0.4, 0.2, 0.7],
            [-0.1, 0.0, 0.2, 0.2, -0.2, 0.1, -0.4, 0.3],
        ]);
        let output = f16_matrix_bytes(&[
            [0.1, -0.2, 0.1, 0.0, -0.1, 0.2, 0.0, -0.1],
            [-0.3, -0.1, 0.2, 0.1, 0.0, 0.1, -0.2, 0.2],
            [0.4, 0.5, -0.1, 0.1, 0.3, -0.2, 0.2, 0.6],
            [-0.2, 0.1, 0.4, -0.3, 0.1, 0.0, 0.3, -0.1],
        ]);

        let embd_buf = backend.alloc_buffer(token_embd.len() as u64).expect("embd");
        let out_buf = backend.alloc_buffer(output.len() as u64).expect("output");
        let x = backend.alloc_buffer((hidden * 4) as u64).expect("x");
        let normed = backend.alloc_buffer((hidden * 2) as u64).expect("normed");
        let logits = backend.alloc_buffer((vocab * 4) as u64).expect("logits");
        backend
            .upload_bytes(&embd_buf, &token_embd)
            .expect("upload embd");
        backend
            .upload_bytes(&out_buf, &output)
            .expect("upload output");

        let enc = backend.begin().expect("begin");
        backend
            .embed(&enc, &x, &embd_buf, token, hidden as u32)
            .expect("embed");
        backend.submit(enc).expect("submit");

        let got_x = f32_bytes_to_f32(&backend.read_bytes(&x, hidden * 4).expect("read x"));
        let (want_x, want_logits, want_id) =
            dense_oracle(&token_embd, &output, token as usize, hidden, vocab);
        backend
            .upload_bytes(&normed, &f32_slice_to_f16_bytes(&want_x))
            .expect("upload normed");
        let enc = backend.begin().expect("begin logits");
        backend.logits(
            &enc,
            &logits,
            &normed,
            &out_buf,
            hidden as u32,
            vocab as u32,
        );
        backend.submit(enc).expect("submit logits");

        let got_logits = backend.read_logits(&logits, vocab).expect("read logits");
        let got_id = backend.read_argmax(&logits, vocab).expect("argmax");

        assert_vec_close("x", &got_x, &want_x, 2e-3);
        assert_vec_close("logits", &got_logits, &want_logits, 2e-3);
        assert_eq!(got_id, want_id);

        backend.free_buffer(logits);
        backend.free_buffer(normed);
        backend.free_buffer(x);
        backend.free_buffer(out_buf);
        backend.free_buffer(embd_buf);
    }

    #[test]
    fn vulkan_prefill_attention_matches_sequential_decode() {
        let backend = match VulkanBackend::bare() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("external verification: no Vulkan device");
                return;
            }
        };
        for scheme in [KvQuant::F16, KvQuant::Int8] {
            attention_prefill_decode_for_scheme(&backend, scheme);
        }
    }

    fn attention_prefill_decode_for_scheme(backend: &VulkanBackend, scheme: KvQuant) {
        let meta = attention_meta();
        let n = 3usize;
        let q_heads = meta.head_count;
        let head_dim = meta.head_dim;
        let row_elems = q_heads * head_dim;
        let row_bytes = row_elems * 2;
        let q_rows: Vec<f32> = (0..n * row_elems)
            .map(|i| (i % 17) as f32 * 0.03 - 0.22)
            .collect();
        let q = backend.alloc_buffer((n * row_bytes) as u64).expect("q");
        let prefill = backend
            .alloc_buffer((n * row_bytes) as u64)
            .expect("prefill");
        let decode = backend
            .alloc_buffer((n * row_bytes) as u64)
            .expect("decode");
        let token_k = backend
            .alloc_buffer((head_dim * 2) as u64)
            .expect("token k");
        let token_v = backend
            .alloc_buffer((head_dim * 2) as u64)
            .expect("token v");
        let kv = kv_cache::alloc_shape(
            backend,
            meta.block_count,
            3,
            meta.head_count_kv,
            meta.head_dim,
            meta.head_dim,
            scheme,
        )
        .expect("kv");

        backend
            .upload_bytes(&q, &f32_slice_to_f16_bytes(&q_rows))
            .expect("upload q");
        for pos in 0..n {
            let k: Vec<f32> = (0..head_dim)
                .map(|i| ((pos * 7 + i) % 13) as f32 * 0.025 - 0.12)
                .collect();
            let v: Vec<f32> = (0..head_dim)
                .map(|i| ((pos * 11 + i) % 19) as f32 * 0.02 - 0.15)
                .collect();
            backend
                .upload_bytes(&token_k, &f32_slice_to_f16_bytes(&k))
                .expect("upload k");
            backend
                .upload_bytes(&token_v, &f32_slice_to_f16_bytes(&v))
                .expect("upload v");
            let enc = backend.begin().expect("begin append");
            kv_cache::append(backend, &enc, &kv, 0, pos, &token_k, &token_v).expect("append kv");
            backend.submit(enc).expect("submit append");
        }

        let enc = backend.begin().expect("begin prefill");
        backend.attention_prefill(&enc, &prefill, &q, &kv, q_heads as u32, 0, n as u32, 0);
        backend.submit(enc).expect("submit prefill");

        for pos in 0..n {
            let qi = backend.view(&q, (pos * row_bytes) as u64, row_bytes as u64);
            let oi = backend.view(&decode, (pos * row_bytes) as u64, row_bytes as u64);
            let enc = backend.begin().expect("begin decode");
            backend.attention_decode(&enc, &oi, &qi, &kv, q_heads as u32, pos as u32, 0);
            backend.submit(enc).expect("submit decode");
        }

        let got_prefill = f16_bytes_to_f32(
            &backend
                .read_bytes(&prefill, n * row_bytes)
                .expect("read prefill"),
        );
        let got_decode = f16_bytes_to_f32(
            &backend
                .read_bytes(&decode, n * row_bytes)
                .expect("read decode"),
        );
        let (label, tol) = match scheme {
            KvQuant::F16 => ("attention_f16", 2e-2),
            KvQuant::Int8 => ("attention_int8", 5e-2),
        };
        assert_vec_close(label, &got_prefill, &got_decode, tol);

        kv_cache::free(backend, kv);
        backend.free_buffer(token_v);
        backend.free_buffer(token_k);
        backend.free_buffer(decode);
        backend.free_buffer(prefill);
        backend.free_buffer(q);
    }

    #[test]
    fn cpu_and_vulkan_match_yarn_rope_intermediate_tensor() {
        let backend = match VulkanBackend::bare() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("external verification: no Vulkan device");
                host_oracle_smoke();
                return;
            }
        };
        let yarn = yarn();
        let input: Vec<f32> = (0..16).map(|i| i as f32 / 7.0 - 1.0).collect();
        let rounded_input: Vec<f32> = input.into_iter().map(round_f16).collect();
        let raw = f32_slice_to_f16_bytes(&rounded_input);
        let buf = backend.alloc_buffer(raw.len() as u64).expect("buffer");
        backend.upload_bytes(&buf, &raw).expect("upload");

        let enc = backend.begin().expect("begin");
        elementwise::rope_yarn(
            &backend.dev,
            &backend.reg,
            enc,
            &buf,
            2,
            8,
            yarn.rope_dim as u32,
            512,
            yarn.freq_base,
            yarn.factor,
            yarn.beta_fast,
            yarn.beta_slow,
            yarn.original_context as u32,
            yarn.post_scale(RopeRole::Query, 512),
        );
        backend.submit(enc).expect("submit");
        let got_raw = backend.read_bytes(&buf, raw.len()).expect("read");
        let got: Vec<f32> = got_raw
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let want = cpu_rope(&yarn, RopeRole::Query, 512, &rounded_input);
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let tol = 2e-3 + 2e-3 * w.abs();
            assert!((*g - w).abs() <= tol, "i={i} got={g} want={w} tol={tol}");
        }
        backend.free_buffer(buf);
    }

    fn host_oracle_smoke() {
        let y = yarn();
        let x = vec![0.0f32; 16];
        let out = cpu_rope(&y, RopeRole::Query, 512, &x);
        assert_eq!(out.len(), x.len());
        assert!(out.iter().all(|v| v.is_finite()));
    }

    fn dense_host_oracle_smoke() {
        let embd = f16_matrix_bytes(&[[1.0, 0.0], [0.0, 1.0]]);
        let output = f16_matrix_bytes(&[[0.0, 1.0], [1.0, 0.0]]);
        let (_, logits, id) = dense_oracle(&embd, &output, 0, 2, 2);
        assert_eq!(logits.len(), 2);
        assert_eq!(id, 1);
    }

    fn attention_meta() -> ModelMetadata {
        ModelMetadata {
            block_count: 1,
            embedding_length: 128,
            head_count: 2,
            head_count_kv: 1,
            head_dim: 64,
            feed_forward_length: 256,
            vocab_size: 4,
        }
    }

    fn dense_oracle(
        token_embd: &[u8],
        output: &[u8],
        token: usize,
        hidden: usize,
        vocab: usize,
    ) -> (Vec<f32>, Vec<f32>, u32) {
        let embd = f16_bytes_to_f32(token_embd);
        let weights = f16_bytes_to_f32(output);
        let x = embd[token * hidden..token * hidden + hidden].to_vec();
        let mut logits = vec![0.0f32; vocab];
        for row in 0..vocab {
            let w = &weights[row * hidden..row * hidden + hidden];
            logits[row] = w.iter().zip(&x).map(|(a, b)| a * b).sum();
        }
        let id = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap();
        (x, logits, id)
    }

    fn f16_matrix_bytes<const C: usize>(rows: &[[f32; C]]) -> Vec<u8> {
        let mut out = Vec::with_capacity(rows.len() * C * 2);
        for row in rows {
            for &v in row {
                out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
            }
        }
        out
    }

    fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    }

    fn f32_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn assert_vec_close(label: &str, got: &[f32], want: &[f32], base_tol: f32) {
        assert_eq!(got.len(), want.len());
        let mut max_err = 0.0f32;
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let err = (*g - *w).abs();
            max_err = max_err.max(err);
            let tol = base_tol + base_tol * w.abs();
            assert!(err <= tol, "{label}[{i}] got={g} want={w} tol={tol}");
        }
        println!("{label} max_err={max_err}");
    }

    fn yarn() -> Yarn {
        Yarn {
            rope_dim: 8,
            original_context: 128,
            freq_base: 1_000_000.0,
            factor: 8.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            log_multiplier: 0.1,
            q_temperature_scale: 1.25,
        }
    }

    fn cpu_rope(y: &Yarn, role: RopeRole, pos: usize, x: &[f32]) -> Vec<f32> {
        let mut out = x.to_vec();
        let heads = 2usize;
        let head_dim = 8usize;
        let half = y.rope_dim / 2;
        for head in 0..heads {
            let base = head * head_dim;
            for pair in 0..half {
                let p = y.pair(role, pair, pos).expect("valid yarn pair");
                let first = base + pair * 2;
                let second = first + 1;
                let a = out[first];
                let b = out[second];
                let scale = y.post_scale(role, pos);
                out[first] = (a * p.cos - b * p.sin) * scale;
                out[second] = (a * p.sin + b * p.cos) * scale;
            }
        }
        out
    }

    fn f32_slice_to_f16_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for &v in values {
            out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
        }
        out
    }

    fn round_f16(v: f32) -> f32 {
        f16_to_f32(f32_to_f16(v))
    }

    fn f32_to_f16(f: f32) -> u16 {
        let bits = f.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mant = bits & 0x7f_ffff;
        if exp == 0xff {
            return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
        }
        let e = exp - 127 + 15;
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        if e <= 0 {
            if e < -10 {
                return sign;
            }
            let m = mant | 0x80_0000;
            let shift = (14 - e) as u32;
            let half = (m >> shift) as u16;
            let rem = m & ((1 << shift) - 1);
            let round =
                u16::from(rem > (1 << (shift - 1)) || rem == (1 << (shift - 1)) && half & 1 == 1);
            return sign | (half + round);
        }
        let half = (e as u16) << 10 | (mant >> 13) as u16;
        let rem = mant & 0x1fff;
        let round = u16::from(rem > 0x1000 || rem == 0x1000 && half & 1 == 1);
        sign | (half + round)
    }

    fn f16_to_f32(h: u16) -> f32 {
        let sign = ((h >> 15) & 1) as u32;
        let exp = ((h >> 10) & 0x1f) as u32;
        let mant = (h & 0x3ff) as u32;
        let bits = if exp == 0 {
            if mant == 0 {
                sign << 31
            } else {
                let mut e = -1i32;
                let mut m = mant;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x3ff;
                let exp32 = (127 - 15 + 2 + e) as u32;
                (sign << 31) | (exp32 << 23) | (m << 13)
            }
        } else if exp == 0x1f {
            (sign << 31) | (0xff << 23) | (mant << 13)
        } else {
            let exp32 = exp + (127 - 15);
            (sign << 31) | (exp32 << 23) | (mant << 13)
        };
        f32::from_bits(bits)
    }
}
