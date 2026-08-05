/*
 * graph_horizon_engine — GGUF loader
 * Low-level GGUF parsing over a read-only mmap (memmap2): exposes the header,
 * the typed metadata KV map and the tensor table, with zero-copy access to the
 * raw weight bytes. Knows nothing about any model architecture. The byte
 * reader and the value model live in sibling modules (`cursor`, `value`).
 *
 * Security: the file is untrusted input. GGUF is little-endian; every read
 * advances a bounds-checked cursor, every tensor offset/length is validated
 * against the file size before a slice is produced, and no malformed file may
 * panic — failures are always Result. Error messages carry no system paths.
*/

use std::collections::HashMap;
use std::path::Path;

use color_eyre::eyre::{Result, bail, eyre};
use memmap2::Mmap;

use super::tensor_index::{GgmlType, TensorInfo};
use cursor::{Cursor, align_up, read_value};

mod cursor;
mod value;

pub use value::GgufValue;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
// GGML supports at most 4 dimensions; a larger count signals a malformed file.
const MAX_DIMS: usize = 4;

// A parsed GGUF file: owns the mmap and exposes the header, the metadata KV map
// and the tensor table. Weight bytes are sliced zero-copy out of the mmap.
pub struct GgufFile {
    mmap: Mmap,
    version: u32,
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<TensorInfo>,
    data_offset: usize,
    file_len: usize,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<GgufFile> {
        let file = std::fs::File::open(path).map_err(|_| eyre!("gguf: cannot open model file"))?;
        // SAFETY: read-only mapping; we treat the file as immutable for the
        // lifetime of this GgufFile and never write through the mapping.
        let mmap = unsafe { Mmap::map(&file).map_err(|_| eyre!("gguf: cannot mmap model file"))? };
        let file_len = mmap.len();
        let mut cur = Cursor::new(&mmap);

        // Magic + version before anything else: reject non-GGUF / unknown layout.
        if cur.take(4)? != GGUF_MAGIC {
            bail!("gguf: not a GGUF file (bad magic)");
        }
        let version = cur.u32()?;
        if version != 2 && version != 3 {
            bail!("gguf: unsupported GGUF version {version}");
        }

        let tensor_count = cur.u64()?;
        let kv_count = cur.u64()?;

        // Metadata KV map. Duplicate keys: last one wins (documented choice).
        let mut metadata = HashMap::new();
        for _ in 0..kv_count {
            let key = cur.string()?;
            let vtype = cur.u32()?;
            let value = read_value(&mut cur, vtype)?;
            metadata.insert(key, value);
        }

        // Tensor table: name, shape, dtype, data-relative offset.
        let mut tensors = Vec::new();
        for _ in 0..tensor_count {
            let name = cur.string()?;
            let n_dims = cur.u32()? as usize;
            if n_dims > MAX_DIMS {
                bail!("gguf: tensor '{name}' has too many dimensions");
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(cur.u64()?);
            }
            let ggml_type = GgmlType::from_u32(cur.u32()?);
            let offset = cur.u64()?;
            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }

        // The data section starts after the tensor table, padded up to
        // general.alignment (default 32). Must stay within the file.
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .filter(|a| *a > 0)
            .unwrap_or(DEFAULT_ALIGNMENT);
        let data_offset = align_up(cur.pos as u64, alignment)
            .filter(|o| (*o as usize) <= file_len)
            .ok_or_else(|| eyre!("gguf: data section beyond end of file"))?
            as usize;

        let f = GgufFile {
            mmap,
            version,
            metadata,
            tensors,
            data_offset,
            file_len,
        };

        // Validate every tensor span now, so later slicing is total and no
        // overlapping or out-of-bounds weight range survives past open().
        f.validate_tensor_spans()?;
        Ok(f)
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn metadata(&self) -> &HashMap<String, GgufValue> {
        &self.metadata
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    // Zero-copy slice of a tensor's raw bytes within the mmap. Every bound is
    // checked: an offset or length outside the file is rejected, never sliced.
    pub fn tensor_bytes(&self, info: &TensorInfo) -> Result<&[u8]> {
        let (start, end) = self.tensor_span(info)?;
        Ok(&self.mmap[start..end])
    }

    fn tensor_span(&self, info: &TensorInfo) -> Result<(usize, usize)> {
        let nbytes = info
            .byte_len()
            .ok_or_else(|| eyre!("gguf: cannot size tensor '{}'", info.name))?;
        let offset =
            usize::try_from(info.offset).map_err(|_| eyre!("gguf: tensor offset overflow"))?;
        let nbytes = usize::try_from(nbytes).map_err(|_| eyre!("gguf: tensor length overflow"))?;
        let start = self
            .data_offset
            .checked_add(offset)
            .ok_or_else(|| eyre!("gguf: tensor offset overflow"))?;
        let end = start
            .checked_add(nbytes)
            .ok_or_else(|| eyre!("gguf: tensor length overflow"))?;
        if end > self.file_len {
            bail!("gguf: tensor '{}' extends beyond end of file", info.name);
        }
        Ok((start, end))
    }

    fn validate_tensor_spans(&self) -> Result<()> {
        let mut spans = Vec::with_capacity(self.tensors.len());
        for tensor in &self.tensors {
            spans.push((self.tensor_span(tensor)?, tensor.name.as_str()));
        }
        spans.sort_by_key(|((start, _), _)| *start);
        for pair in spans.windows(2) {
            let ((_, prev_end), prev_name) = pair[0];
            let ((next_start, _), next_name) = pair[1];
            if prev_end > next_start {
                bail!("gguf: tensor '{next_name}' overlaps tensor '{prev_name}'");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    // Minimal GGUF (v3) with a single tensor and no metadata, padded to the data
    // section and followed by `data_bytes` zero bytes. The header fields are valid;
    // only `dims`/`gtype`/`offset`/`data_bytes` vary to drive the bound checks.
    fn build_gguf_one_tensor(dims: &[u64], gtype: u32, offset: u64, data_bytes: usize) -> Vec<u8> {
        build_gguf_tensors(&[("w", dims, gtype, offset)], data_bytes)
    }

    fn build_gguf_tensors(tensors: &[(&str, &[u64], u32, u64)], data_bytes: usize) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(GGUF_MAGIC);
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&(tensors.len() as u64).to_le_bytes()); // tensor_count
        b.extend_from_slice(&0u64.to_le_bytes()); // kv_count
        for &(name, dims, gtype, offset) in tensors {
            push_str(&mut b, name);
            b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                b.extend_from_slice(&d.to_le_bytes());
            }
            b.extend_from_slice(&gtype.to_le_bytes());
            b.extend_from_slice(&offset.to_le_bytes());
        }
        while b.len() % DEFAULT_ALIGNMENT as usize != 0 {
            b.push(0);
        }
        b.extend(std::iter::repeat_n(0u8, data_bytes));
        b
    }

    // open() requires a real (mmap-able) file; write the bytes to a unique temp path,
    // run `f`, and always remove the file afterwards.
    fn with_temp_gguf<R>(tag: &str, bytes: &[u8], f: impl FnOnce(Result<GgufFile>) -> R) -> R {
        let path = std::env::temp_dir().join(format!("ndg_gguf_{tag}_{}.gguf", std::process::id()));
        std::fs::write(&path, bytes).expect("write temp gguf");
        let r = f(GgufFile::open(&path));
        let _ = std::fs::remove_file(&path);
        r
    }

    // A well-formed F32 [256] tensor with its 1024 data bytes present opens cleanly —
    // proves the builder is valid so the rejection tests below aren't trivially passing.
    #[test]
    fn well_formed_tensor_opens() {
        let g = build_gguf_one_tensor(&[256], 0, 0, 1024);
        with_temp_gguf("ok", &g, |r| assert!(r.is_ok(), "valid GGUF must open"));
    }

    // offset + byte_len beyond the mapped file is rejected at open() (the per-tensor
    // validation), never sliced out of bounds (SEC-INV at the load boundary).
    #[test]
    fn tensor_offset_plus_len_beyond_file_is_rejected() {
        // F32 [256] ⇒ byte_len 1024, but no data bytes follow the header.
        let g = build_gguf_one_tensor(&[256], 0, 0, 0);
        with_temp_gguf("oob", &g, |r| {
            assert!(r.is_err(), "OOB tensor must be rejected")
        });
    }

    // A dtype whose element count is not a whole number of blocks (Q4_K [100],
    // 100 % 256 ≠ 0) has no coherent byte_len, so the load rejects it before any
    // byte reaches a kernel.
    #[test]
    fn incoherent_byte_len_is_rejected() {
        let g = build_gguf_one_tensor(&[100], 12, 0, 4096);
        with_temp_gguf("bad", &g, |r| {
            assert!(r.is_err(), "incoherent byte_len must be rejected")
        });
    }

    #[test]
    fn overlapping_tensor_ranges_are_rejected() {
        let g = build_gguf_tensors(&[("a", &[256], 0, 0), ("b", &[256], 0, 512)], 2048);
        with_temp_gguf("overlap", &g, |r| {
            let err = match r {
                Ok(_) => panic!("overlapping tensors must fail"),
                Err(err) => err.to_string(),
            };
            assert!(err.contains("overlaps tensor"), "{err}");
        });
    }

    #[test]
    fn tensor_offset_overflow_is_rejected() {
        let g = build_gguf_one_tensor(&[256], 0, u64::MAX, 1024);
        with_temp_gguf("overflow", &g, |r| {
            let err = match r {
                Ok(_) => panic!("overflowing tensor offset must fail"),
                Err(err) => err.to_string(),
            };
            assert!(err.contains("tensor offset overflow"), "{err}");
        });
    }

    #[test]
    fn known_bf16_tensor_layout_is_indexed() {
        let g = build_gguf_one_tensor(&[256], 30, 0, 512);
        with_temp_gguf("bf16", &g, |r| {
            let file = r.expect("BF16 is a known GGML layout");
            assert!(matches!(file.tensors()[0].ggml_type, GgmlType::BF16));
            assert_eq!(file.tensor_bytes(&file.tensors()[0]).unwrap().len(), 512);
        });
    }
}
