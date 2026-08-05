/*
 * graph_horizon_engine — GGUF byte cursor
 * Bounds-checked little-endian reader over the (untrusted) GGUF bytes, plus the
 * metadata-value reader and the alignment helper. Every read validates the
 * requested range against the slice length before advancing, so untrusted counts
 * and lengths can never read out of bounds and a truncated file never panics.
*/

use color_eyre::eyre::{Result, bail, eyre};

use super::value::GgufValue;

// Bounds-checked sequential reader over the mmap bytes. Every read validates the
// requested range against the slice length before advancing, so untrusted counts
// and lengths can never read out of bounds.
pub(super) struct Cursor<'a> {
    buf: &'a [u8],
    pub(super) pos: usize,
}

impl<'a> Cursor<'a> {
    pub(super) fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub(super) fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| eyre!("gguf: read length overflow"))?;
        if end > self.buf.len() {
            bail!("gguf: file is truncated");
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    // The fixed-width helpers below convert a slice of exactly the right length,
    // so `try_into` never fails: the unwraps are on lengths guaranteed by `take`.
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    pub(super) fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(super) fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()? as i8)
    }
    fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()? as i16)
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    // GGUF string: u64 length prefix + UTF-8 bytes, no null terminator.
    pub(super) fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| eyre!("gguf: invalid utf-8 in string"))
    }
}

// Reads one metadata value of the given GGUF value-type tag.
pub(super) fn read_value(cur: &mut Cursor, vtype: u32) -> Result<GgufValue> {
    Ok(match vtype {
        0 => GgufValue::U8(cur.u8()?),
        1 => GgufValue::I8(cur.i8()?),
        2 => GgufValue::U16(cur.u16()?),
        3 => GgufValue::I16(cur.i16()?),
        4 => GgufValue::U32(cur.u32()?),
        5 => GgufValue::I32(cur.i32()?),
        6 => GgufValue::F32(cur.f32()?),
        7 => GgufValue::Bool(cur.u8()? != 0),
        8 => GgufValue::String(cur.string()?),
        9 => {
            let inner = cur.u32()?;
            if inner == 9 {
                bail!("gguf: nested arrays are not supported");
            }
            let n = cur.u64()? as usize;
            // Not pre-sized: the count is untrusted, so the per-element reads
            // (each bounds-checked) gate growth instead of an up-front alloc.
            let mut items = Vec::new();
            for _ in 0..n {
                items.push(read_value(cur, inner)?);
            }
            GgufValue::Array(items)
        }
        10 => GgufValue::U64(cur.u64()?),
        11 => GgufValue::I64(cur.i64()?),
        12 => GgufValue::F64(cur.f64()?),
        other => bail!("gguf: unknown metadata value type {other}"),
    })
}

// Rounds `value` up to a multiple of `align`. None on overflow.
pub(super) fn align_up(value: u64, align: u64) -> Option<u64> {
    let rem = value % align;
    if rem == 0 {
        Some(value)
    } else {
        value.checked_add(align - rem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A read past the end of a truncated buffer fails cleanly (no panic).
    #[test]
    fn take_past_end_is_error_not_panic() {
        let mut cur = Cursor::new(&[1, 2, 3]);
        assert!(cur.take(2).is_ok());
        assert!(cur.take(2).is_err(), "reading beyond the buffer must error");
    }

    #[test]
    fn align_up_rounds_to_multiple() {
        assert_eq!(align_up(0, 32), Some(0));
        assert_eq!(align_up(1, 32), Some(32));
        assert_eq!(align_up(32, 32), Some(32));
        assert_eq!(align_up(33, 32), Some(64));
    }
}
