/*
 * graph_horizon_engine — qualified Vulkan device profile
 * Reduces immutable vendor identity and queried capabilities to the few
 * measured tuning choices Vulkan cannot express through portable features
 * alone. Unknown families keep the standard submission and kernel paths.
 */

const AMD_VENDOR_ID: u32 = 0x1002;
const NVIDIA_VENDOR_ID: u32 = 0x10de;

const STANDARD_PREFILL_ROWS: usize = 256;
const MATRIX2_PREFILL_ROWS: usize = 512;
// Conservative submission bounds measured on the qualified AMD/RADV family.
const BOUNDED_PREFILL_ROWS: usize = 128;
const DEEP_LONG_PREFILL_ROWS: usize = 64;

#[derive(Clone, Copy)]
pub(crate) struct Profile {
    integer_q4_batch: bool,
    fixed_wave32: bool,
    bounded_submissions: bool,
}

impl Profile {
    pub(crate) const fn from(vendor_id: u32, dp4a: bool, wave32_control: bool) -> Self {
        let measured_family = vendor_id == AMD_VENDOR_ID;
        Self {
            integer_q4_batch: measured_family && dp4a,
            fixed_wave32: measured_family && wave32_control,
            bounded_submissions: measured_family,
        }
    }

    pub(crate) const fn integer_q4_batch(self) -> bool {
        self.integer_q4_batch
    }

    pub(crate) const fn fixed_wave32(self) -> bool {
        self.fixed_wave32
    }

    pub(crate) const fn prefill_rows(
        self,
        block_count: usize,
        context: usize,
        matrix2_wide: bool,
    ) -> usize {
        if self.bounded_submissions && block_count >= 32 && context >= 16_384 {
            DEEP_LONG_PREFILL_ROWS
        } else if self.bounded_submissions {
            BOUNDED_PREFILL_ROWS
        } else if matrix2_wide {
            MATRIX2_PREFILL_ROWS
        } else {
            STANDARD_PREFILL_ROWS
        }
    }

    #[cfg(any(test, feature = "vulkan-hybrid"))]
    pub(crate) const fn hybrid_prefill_rows(
        self,
        matrix2_wide: bool,
        default_rows: usize,
    ) -> usize {
        if matrix2_wide && !self.bounded_submissions {
            MATRIX2_PREFILL_ROWS
        } else {
            default_rows
        }
    }
}

pub(crate) const fn is_nvidia(vendor_id: u32) -> bool {
    vendor_id == NVIDIA_VENDOR_ID
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_devices_keep_portable_tuning() {
        let profile = Profile::from(0x8086, true, true);
        assert!(!profile.integer_q4_batch());
        assert!(!profile.fixed_wave32());
        assert_eq!(profile.prefill_rows(40, 32_768, false), 256);
    }

    #[test]
    fn measured_family_requires_each_optional_capability() {
        let portable = Profile::from(AMD_VENDOR_ID, false, false);
        assert!(!portable.integer_q4_batch());
        assert!(!portable.fixed_wave32());
        assert_eq!(portable.prefill_rows(26, 28_160, true), 128);

        let qualified = Profile::from(AMD_VENDOR_ID, true, true);
        assert!(qualified.integer_q4_batch());
        assert!(qualified.fixed_wave32());
        assert_eq!(qualified.prefill_rows(31, 16_384, true), 128);
        assert_eq!(qualified.prefill_rows(32, 16_384, true), 64);
        assert_eq!(qualified.prefill_rows(32, 16_383, true), 128);
    }

    #[test]
    fn matrix2_capability_owns_wide_rows() {
        let profile = Profile::from(NVIDIA_VENDOR_ID, false, false);
        assert_eq!(profile.prefill_rows(40, 32_768, false), 256);
        assert_eq!(profile.prefill_rows(40, 32_768, true), 512);
        assert_eq!(profile.hybrid_prefill_rows(false, 32), 32);
        assert_eq!(profile.hybrid_prefill_rows(true, 32), 512);
        assert_eq!(
            Profile::from(AMD_VENDOR_ID, true, true).hybrid_prefill_rows(true, 32),
            32
        );
    }
}
