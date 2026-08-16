/*
 * graph_horizon_engine — Vulkan F16 prefill-attention routing policy
 * Selects one attention specialization from immutable pipeline availability,
 * runtime shape, and the retained diagnostic switch. It performs no hardware
 * query, command recording, allocation, or shader execution, so route decisions
 * remain unit-testable without a physical GPU.
 */

use std::sync::OnceLock;

const MATRIX2_HEAD_DIM: u32 = 128;
const NVIDIA_Q64_ROWS: u32 = 64;
const QUALIFIED_GQA_RATIO: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Route {
    NvidiaQ64,
    Matrix2Q32,
    CoopQk,
    Tiled,
    Wide,
    Portable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mode {
    Auto,
    Generic,
    Phase19,
}

#[derive(Clone, Copy)]
pub(super) struct Shape {
    pub head_dim: u32,
    pub kv_heads: u32,
    pub q_heads: u32,
    pub rows: u32,
}

#[derive(Clone, Copy, Default)]
pub(super) struct Pipelines {
    pub nvidia_q64: bool,
    pub matrix2_q32: bool,
    pub coop_qk: bool,
    pub tiled: bool,
    pub wide: bool,
}

pub(super) fn matrix2_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("GRAPH_HORIZON_PREFILL_MATRIX2")
                .ok()
                .as_deref(),
            None | Some("1" | "true" | "yes")
        )
    })
}

pub(super) fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| {
        match std::env::var("GRAPH_HORIZON_PREFILL_ATTENTION_ROUTE")
            .ok()
            .as_deref()
        {
            None | Some("auto") => Mode::Auto,
            Some("phase19") => Mode::Phase19,
            // A malformed diagnostic request fails toward the portable path.
            Some("generic") | Some(_) => Mode::Generic,
        }
    })
}

pub(super) fn nvidia_q64_eligible(shape: Shape, pipeline: bool) -> bool {
    pipeline
        && shape.head_dim == MATRIX2_HEAD_DIM
        && shape.rows != 0
        && shape.rows.is_multiple_of(NVIDIA_Q64_ROWS)
        && shape.kv_heads != 0
        && shape.q_heads.is_multiple_of(shape.kv_heads)
        && shape.q_heads / shape.kv_heads == QUALIFIED_GQA_RATIO
}

pub(super) fn select(
    shape: Shape,
    pipelines: Pipelines,
    matrix2: bool,
    mode: Mode,
) -> (Route, u32) {
    if mode == Mode::Generic {
        return (Route::Portable, shape.rows);
    }
    if nvidia_q64_eligible(shape, pipelines.nvidia_q64) && (matrix2 || mode == Mode::Phase19) {
        (Route::NvidiaQ64, shape.rows / NVIDIA_Q64_ROWS)
    } else if mode == Mode::Phase19 {
        // Forced routing never bypasses safety; unsupported tuples fall back.
        (Route::Portable, shape.rows)
    } else if matrix2
        && shape.head_dim == MATRIX2_HEAD_DIM
        && shape.rows != 0
        && shape.rows.is_multiple_of(32)
        && pipelines.matrix2_q32
    {
        (Route::Matrix2Q32, shape.rows / 32)
    } else if shape.head_dim == MATRIX2_HEAD_DIM && pipelines.coop_qk {
        (Route::CoopQk, shape.rows.div_ceil(16))
    } else if shape.head_dim == MATRIX2_HEAD_DIM && pipelines.tiled {
        (Route::Tiled, shape.rows.div_ceil(8))
    } else if pipelines.wide {
        (Route::Wide, shape.rows)
    } else {
        (Route::Portable, shape.rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(rows: u32) -> Shape {
        Shape {
            head_dim: 128,
            kv_heads: 8,
            q_heads: 32,
            rows,
        }
    }

    fn pipelines() -> Pipelines {
        Pipelines {
            nvidia_q64: true,
            matrix2_q32: true,
            coop_qk: true,
            tiled: true,
            wide: true,
        }
    }

    #[test]
    fn q64_uses_only_the_qualified_shape_not_absolute_head_counts() {
        assert!(nvidia_q64_eligible(shape(64), true));
        assert!(nvidia_q64_eligible(
            Shape {
                q_heads: 4,
                kv_heads: 1,
                ..shape(64)
            },
            true
        ));
        assert!(!nvidia_q64_eligible(
            Shape {
                q_heads: 8,
                kv_heads: 1,
                ..shape(64)
            },
            true
        ));
        assert!(!nvidia_q64_eligible(
            Shape {
                head_dim: 64,
                ..shape(64)
            },
            true
        ));
        assert!(!nvidia_q64_eligible(shape(64), false));
    }

    #[test]
    fn q64_boundaries_and_short_complete_tile_are_explicit() {
        for rows in [63, 65, 127, 129, 255, 257] {
            assert_ne!(
                select(shape(rows), pipelines(), true, Mode::Auto).0,
                Route::NvidiaQ64
            );
        }
        for rows in [64, 128, 256] {
            assert_eq!(
                select(shape(rows), pipelines(), true, Mode::Auto).0,
                Route::NvidiaQ64
            );
        }
    }

    #[test]
    fn missing_capability_or_forced_baseline_falls_back() {
        let mut available = pipelines();
        available.nvidia_q64 = false;
        assert_eq!(
            select(shape(64), available, true, Mode::Auto),
            (Route::Matrix2Q32, 2)
        );
        assert_eq!(
            select(shape(64), pipelines(), false, Mode::Auto),
            (Route::CoopQk, 4)
        );

        let unavailable = Pipelines::default();
        assert_eq!(
            select(shape(64), unavailable, true, Mode::Auto),
            (Route::Portable, 64)
        );
    }

    #[test]
    fn unsupported_shape_never_enters_q64() {
        let wrong_gqa = Shape {
            q_heads: 8,
            kv_heads: 1,
            ..shape(64)
        };
        assert_eq!(
            select(wrong_gqa, pipelines(), true, Mode::Auto),
            (Route::Matrix2Q32, 2)
        );
        let wrong_dimension = Shape {
            head_dim: 64,
            ..shape(64)
        };
        assert_eq!(
            select(wrong_dimension, pipelines(), true, Mode::Auto),
            (Route::Wide, 64)
        );
    }

    #[test]
    fn diagnostic_modes_force_exact_safe_endpoints() {
        assert_eq!(
            select(shape(64), pipelines(), true, Mode::Generic),
            (Route::Portable, 64)
        );
        assert_eq!(
            select(shape(64), pipelines(), false, Mode::Phase19),
            (Route::NvidiaQ64, 1)
        );
        assert_eq!(
            select(shape(65), pipelines(), true, Mode::Phase19),
            (Route::Portable, 65)
        );
    }
}
