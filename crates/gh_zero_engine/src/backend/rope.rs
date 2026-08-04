/*
 * gh_zero_engine — shared YaRN RoPE parameters
 * Owns the shared Ministral YaRN parameters and query post-scale. CPU builds
 * also use its checked scalar pair math as the reference kernel; Vulkan passes
 * the same parameters directly to shaders.
*/

#[cfg(any(
    feature = "cpu",
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    test
))]
use color_eyre::eyre::{Result, bail};

pub(crate) struct Yarn {
    pub rope_dim: usize,
    pub original_context: usize,
    pub freq_base: f32,
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    #[cfg(any(
        feature = "cpu",
        feature = "vulkan-hybrid",
        feature = "metal-hybrid",
        test
    ))]
    pub log_multiplier: f32,
    pub q_temperature_scale: f32,
}

#[derive(Clone, Copy)]
pub(crate) enum RopeRole {
    Query,
    Key,
}

#[cfg(any(
    feature = "cpu",
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    test
))]
pub(crate) struct Pair {
    pub cos: f32,
    pub sin: f32,
}

impl Yarn {
    #[cfg(any(
        feature = "cpu",
        feature = "vulkan-hybrid",
        feature = "metal-hybrid",
        test
    ))]
    pub(crate) fn validate(&self) -> Result<()> {
        if self.rope_dim == 0 || !self.rope_dim.is_multiple_of(2) {
            bail!("rope: rope_dim must be positive and even");
        }
        if self.original_context == 0 {
            bail!("rope: original context must be positive");
        }
        for (name, value) in [
            ("freq_base", self.freq_base),
            ("factor", self.factor),
            ("beta_fast", self.beta_fast),
            ("beta_slow", self.beta_slow),
            ("log_multiplier", self.log_multiplier),
            ("q_temperature_scale", self.q_temperature_scale),
        ] {
            if !value.is_finite() || value <= 0.0 {
                bail!("rope: {name} must be finite and positive");
            }
        }
        Ok(())
    }

    #[cfg(any(
        feature = "cpu",
        feature = "vulkan-hybrid",
        feature = "metal-hybrid",
        test
    ))]
    pub(crate) fn pair(&self, role: RopeRole, pair: usize, position: usize) -> Result<Pair> {
        self.validate()?;
        if pair >= self.rope_dim / 2 {
            bail!("rope: pair index beyond rope_dim");
        }

        let freq_scale = 1.0 / self.factor;
        let theta_extrap = position as f32
            * self
                .freq_base
                .powf(-(2.0 * pair as f32) / self.rope_dim as f32);
        let theta_interp = freq_scale * theta_extrap;
        let corr = self.correction_dims();
        let ramp = yarn_ramp(corr.0, corr.1, pair);
        let theta = theta_interp * (1.0 - ramp) + theta_extrap * ramp;
        let scale = self.role_scale(role);

        Ok(Pair {
            cos: theta.cos() * scale,
            sin: theta.sin() * scale,
        })
    }

    #[cfg(any(
        feature = "cpu",
        feature = "vulkan-hybrid",
        feature = "metal-hybrid",
        test
    ))]
    fn role_scale(&self, role: RopeRole) -> f32 {
        match role {
            RopeRole::Query => 1.0,
            RopeRole::Key => 1.0,
        }
    }

    pub(crate) fn post_scale(&self, role: RopeRole, position: usize) -> f32 {
        match role {
            RopeRole::Query => {
                let bucket = (position / self.original_context) as f32;
                (bucket + 1.0).ln() * self.q_temperature_scale + 1.0
            }
            RopeRole::Key => 1.0,
        }
    }

    #[cfg(any(
        feature = "cpu",
        feature = "vulkan-hybrid",
        feature = "metal-hybrid",
        test
    ))]
    fn correction_dims(&self) -> (f32, f32) {
        let start = corr_dim(
            self.rope_dim,
            self.original_context,
            self.beta_fast,
            self.freq_base,
        )
        .floor()
        .max(0.0);
        let end = corr_dim(
            self.rope_dim,
            self.original_context,
            self.beta_slow,
            self.freq_base,
        )
        .ceil()
        .min((self.rope_dim - 1) as f32);
        (start, end)
    }
}

#[cfg(any(
    feature = "cpu",
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    test
))]
fn corr_dim(rope_dim: usize, original_context: usize, rotations: f32, base: f32) -> f32 {
    let denom = rotations * 2.0 * std::f32::consts::PI;
    rope_dim as f32 * (original_context as f32 / denom).ln() / (2.0 * base.ln())
}

#[cfg(any(
    feature = "cpu",
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    test
))]
fn yarn_ramp(low: f32, high: f32, pair: usize) -> f32 {
    let y = (pair as f32 - low) / (high - low).max(0.001);
    1.0 - y.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn close(a: f32, b: f32) {
        assert!((a - b).abs() <= 1e-6, "{a} != {b}");
    }

    fn expected_key_angle(y: &Yarn, pair: usize, pos: usize) -> f32 {
        let freq_scale = 1.0 / y.factor;
        let theta_extrap = pos as f32 * y.freq_base.powf(-(2.0 * pair as f32) / y.rope_dim as f32);
        let theta_interp = freq_scale * theta_extrap;
        let corr = y.correction_dims();
        let ramp = yarn_ramp(corr.0, corr.1, pair);
        theta_interp * (1.0 - ramp) + theta_extrap * ramp
    }

    #[test]
    fn key_scale_stays_neutral_across_positions() {
        for pos in [0, 128, 512] {
            let p = yarn().pair(RopeRole::Key, 1, pos).unwrap();
            close((p.cos * p.cos + p.sin * p.sin).sqrt(), 1.0);
        }
    }

    #[test]
    fn query_rope_keeps_unit_magnitude_before_temperature_scale() {
        let y = yarn();
        let p = y.pair(RopeRole::Query, 1, 256).unwrap();
        close((p.cos * p.cos + p.sin * p.sin).sqrt(), 1.0);
    }

    #[test]
    fn query_post_scale_uses_original_context_buckets() {
        let y = yarn();
        close(y.post_scale(RopeRole::Query, 0), 1.0);
        close(y.post_scale(RopeRole::Query, 127), 1.0);
        close(y.post_scale(RopeRole::Query, 128), 2.0f32.ln() * 1.25 + 1.0);
        close(y.post_scale(RopeRole::Key, 512), 1.0);
    }

    #[test]
    fn ramp_mixes_interpolation_and_extrapolation_by_pair() {
        let y = yarn();
        let low = y.pair(RopeRole::Key, 0, 64).unwrap();
        let high = y.pair(RopeRole::Key, 3, 64).unwrap();
        let low_expected = expected_key_angle(&y, 0, 64);
        let high_expected = expected_key_angle(&y, 3, 64);
        close(
            low.sin.atan2(low.cos),
            low_expected.sin().atan2(low_expected.cos()),
        );
        close(
            high.sin.atan2(high.cos),
            high_expected.sin().atan2(high_expected.cos()),
        );
    }

    #[test]
    fn rejects_invalid_parameters_and_pair_index() {
        let mut y = yarn();
        y.factor = 0.0;
        assert!(y.pair(RopeRole::Query, 0, 0).is_err());
        let y = yarn();
        assert!(y.pair(RopeRole::Key, 4, 0).is_err());
    }
}
