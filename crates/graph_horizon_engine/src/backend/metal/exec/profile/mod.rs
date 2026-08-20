/*
 * Metal stage-boundary profiler
 * Owns one timestamp sample buffer, groups adjacent same-category dispatches
 * into sampled compute passes, and aggregates synchronous command results.
 */

mod category;
mod report;
mod summary;

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use color_eyre::eyre::{Result, eyre};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePassDescriptor,
    MTLCounterResultTimestamp, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSamplingPoint, MTLCounterSet, MTLDevice, MTLStorageMode,
};

use self::category::{Category, Phase};
use self::summary::{Mark, Totals};
use crate::backend::metal::pipeline::Kernel;

const SAMPLE_COUNT: usize = 4096;

pub(crate) struct Profile {
    samples: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    totals: Mutex<[Totals; Phase::COUNT]>,
}

// SAFETY: the sample buffer is reused only by the engine's synchronous command
// lifecycle; aggregate mutation is serialized by `totals`.
unsafe impl Send for Profile {}
// SAFETY: immutable sample-buffer access cannot overlap submission, and all
// shared aggregate access is protected by the mutex.
unsafe impl Sync for Profile {}

pub(crate) struct ProfileCommand {
    profile: Arc<Profile>,
    marks: Vec<Mark>,
    next: usize,
    matmul_slot: usize,
    phase: Option<Phase>,
    active: Option<Active>,
}

struct Active {
    category: Category,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

impl Profile {
    pub(crate) fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Arc<Self>> {
        if !device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary) {
            return Err(eyre!("metal profile: stage counters unavailable"));
        }
        let sets = device
            .counterSets()
            .ok_or_else(|| eyre!("metal profile: timestamp set unavailable"))?;
        // SAFETY: Metal exports this process-lifetime immutable NSString constant.
        let timestamp_name = unsafe { objc2_metal::MTLCommonCounterSetTimestamp }.to_string();
        let timestamp = (0..sets.count())
            .map(|index| sets.objectAtIndex(index))
            .find(|set| set.name().to_string() == timestamp_name)
            .ok_or_else(|| eyre!("metal profile: timestamp set unavailable"))?;
        let descriptor = MTLCounterSampleBufferDescriptor::new();
        descriptor.setCounterSet(Some(&timestamp));
        descriptor.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: the fixed capacity bounds every sample index reserved below.
        unsafe { descriptor.setSampleCount(SAMPLE_COUNT) };
        let samples = device
            .newCounterSampleBufferWithDescriptor_error(&descriptor)
            .map_err(|_| eyre!("metal profile: sample buffer unavailable"))?;
        Ok(Arc::new(Self {
            samples,
            totals: Mutex::new(std::array::from_fn(|_| Totals::default())),
        }))
    }

    pub(crate) fn command(self: &Arc<Self>) -> ProfileCommand {
        ProfileCommand {
            profile: self.clone(),
            marks: Vec::new(),
            next: 0,
            matmul_slot: 0,
            phase: None,
            active: None,
        }
    }

    pub(crate) fn finish(&self) {
        report::write(&self.totals.lock().expect("metal profile lock"));
    }
}

impl ProfileCommand {
    pub(crate) fn encoder(
        &mut self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        kernel: Kernel,
        constants: &[u8],
    ) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
        let (category, phase) = category::classify(kernel, constants, &mut self.matmul_slot);
        if let Some(phase) = phase {
            if self.phase.is_some_and(|current| current != phase) {
                return Err(eyre!("metal profile: mixed command phases"));
            }
            self.phase = Some(phase);
        }
        if self
            .active
            .as_ref()
            .is_some_and(|active| active.category == category)
        {
            self.marks
                .last_mut()
                .expect("active Metal profile mark")
                .dispatches += 1;
            return Ok(self.active.as_ref().unwrap().encoder.clone());
        }
        self.finish_encoder();
        let end = self
            .next
            .checked_add(2)
            .filter(|&end| end <= SAMPLE_COUNT)
            .ok_or_else(|| eyre!("metal profile: sample capacity exceeded"))?;
        self.marks.push(Mark {
            category,
            start: self.next,
            end: self.next + 1,
            dispatches: 1,
        });

        let descriptor = MTLComputePassDescriptor::computePassDescriptor();
        // SAFETY: attachment zero exists and both indices are inside the fixed sample buffer.
        let attachment = unsafe {
            descriptor
                .sampleBufferAttachments()
                .objectAtIndexedSubscript(0)
        };
        attachment.setSampleBuffer(Some(&self.profile.samples));
        // SAFETY: both reserved indices are below SAMPLE_COUNT.
        unsafe {
            attachment.setStartOfEncoderSampleIndex(self.next);
            attachment.setEndOfEncoderSampleIndex(self.next + 1);
        }
        self.next = end;
        let encoder = command
            .computeCommandEncoderWithDescriptor(&descriptor)
            .ok_or_else(|| eyre!("metal profile: compute pass failed"))?;
        self.active = Some(Active {
            category,
            encoder: encoder.clone(),
        });
        Ok(encoder)
    }

    pub(crate) fn finish_encoder(&mut self) {
        if let Some(active) = self.active.take() {
            active.encoder.endEncoding();
        }
    }

    pub(crate) fn resolve(&mut self, gpu_ms: f64, cpu_ms: [f64; 3]) -> Result<()> {
        let Some(phase) = self.phase else {
            return Ok(());
        };
        if !gpu_ms.is_finite() || gpu_ms < 0.0 {
            return Err(eyre!("metal profile: invalid command timestamp"));
        }
        let bytes = self
            .next
            .checked_mul(std::mem::size_of::<MTLCounterResultTimestamp>())
            .ok_or_else(|| eyre!("metal profile: sample capacity exceeded"))?;
        // SAFETY: every requested index was reserved and submission completed.
        let data = unsafe {
            self.profile.samples.resolveCounterRange(NSRange {
                location: 0,
                length: self.next,
            })
        }
        .ok_or_else(|| eyre!("metal profile: timestamp resolve failed"))?;
        if data.length() != bytes {
            return Err(eyre!("metal profile: invalid timestamp payload"));
        }
        let mut values = vec![MTLCounterResultTimestamp { timestamp: 0 }; self.next];
        let pointer = NonNull::new(values.as_mut_ptr().cast::<c_void>())
            .ok_or_else(|| eyre!("metal profile: invalid timestamp payload"))?;
        // SAFETY: `values` owns exactly `bytes` writable bytes.
        unsafe { data.getBytes_length(pointer, bytes) };

        let mut totals = self.profile.totals.lock().expect("metal profile lock");
        totals[phase as usize].add_command(gpu_ms, cpu_ms, &self.marks, &values)
    }
}
