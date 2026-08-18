/*
 * graph_horizon_engine — Vulkan F16 prefill-attention routing policy
 * Selects one attention specialization from immutable pipeline availability and
 * runtime shape. It performs no hardware query, command recording, allocation,
 * or shader execution, so route decisions remain unit-testable without a GPU.
 */

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

pub(super) fn nvidia_q64_eligible(shape: Shape, pipeline: bool) -> bool {
    pipeline
        && shape.head_dim == MATRIX2_HEAD_DIM
        && shape.rows != 0
        && shape.rows.is_multiple_of(NVIDIA_Q64_ROWS)
        && shape.kv_heads != 0
        && shape.q_heads.is_multiple_of(shape.kv_heads)
        && shape.q_heads / shape.kv_heads == QUALIFIED_GQA_RATIO
}

pub(super) fn select(shape: Shape, pipelines: Pipelines) -> (Route, u32) {
    if nvidia_q64_eligible(shape, pipelines.nvidia_q64) {
        (Route::NvidiaQ64, shape.rows / NVIDIA_Q64_ROWS)
    } else if shape.head_dim == MATRIX2_HEAD_DIM
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
            assert_ne!(select(shape(rows), pipelines()).0, Route::NvidiaQ64);
        }
        for rows in [64, 128, 256] {
            assert_eq!(select(shape(rows), pipelines()).0, Route::NvidiaQ64);
        }
    }

    #[test]
    fn missing_capability_or_forced_baseline_falls_back() {
        let mut available = pipelines();
        available.nvidia_q64 = false;
        assert_eq!(select(shape(64), available), (Route::Matrix2Q32, 2));
        available.matrix2_q32 = false;
        assert_eq!(select(shape(64), available), (Route::CoopQk, 4));

        let unavailable = Pipelines::default();
        assert_eq!(select(shape(64), unavailable), (Route::Portable, 64));
    }

    #[test]
    fn unsupported_shape_never_enters_q64() {
        let wrong_gqa = Shape {
            q_heads: 8,
            kv_heads: 1,
            ..shape(64)
        };
        assert_eq!(select(wrong_gqa, pipelines()), (Route::Matrix2Q32, 2));
        let wrong_dimension = Shape {
            head_dim: 64,
            ..shape(64)
        };
        assert_eq!(select(wrong_dimension, pipelines()), (Route::Wide, 64));
    }
}
