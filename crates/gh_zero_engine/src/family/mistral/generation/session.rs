/*
 * gh_zero_engine — Ministral runtime-session adapter
 * Constructs the statically selected neutral session from validated model
 * dimensions. It owns no backend mapping, graph traversal, or sampling policy.
 */

use color_eyre::eyre::Result;

use super::super::RuntimeModel;
use super::super::graph::MistralGraph;
use crate::backend::selection::{self, SelectedSession};

pub(super) fn new(model: &RuntimeModel) -> Result<SelectedSession<'_, MistralGraph>> {
    selection::session(
        &model.backend,
        &model.config,
        model.shape(),
        model.context,
        model.scheme,
    )
}
