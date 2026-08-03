/*
 * gh_zero_engine — unified-memory placement seam
 * Reserves the pure unified topology boundary for Metal. M0 never selects this
 * path; probing and allocation are excluded and M2 supplies its arithmetic.
 */

use color_eyre::eyre::{Result, eyre};

use super::PlacementInput;
use crate::backend::hybrid::HybridPlan;
use crate::backend::hybrid::weights::model::WeightBytes;

pub(super) fn select(_weights: &WeightBytes, _input: PlacementInput) -> Result<HybridPlan> {
    Err(eyre!("unified hybrid placement is unavailable"))
}
