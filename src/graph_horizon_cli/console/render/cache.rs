/*
 * Graph Horizon CLI Modules - Console - Render - Cache
 * Single responsibility: cache wrapped conversation lines by terminal width
 * and content revision for viewport drawing.
 */

use ratatui::prelude::*;

use super::RenderContent;
use super::conversation;

// Pre-built line cache keyed on (terminal_width, content_revision).
#[derive(Default)]
pub(crate) struct RenderCache {
    width: u16,
    revision: u64,
    lines: Vec<Line<'static>>,
}

impl RenderCache {
    // Rebuilds the line cache when either the terminal width or the content revision changes.
    pub(super) fn update(&mut self, width: u16, revision: u64, content: &RenderContent<'_>) {
        if self.width == width && self.revision == revision {
            return;
        }

        self.width = width;
        self.revision = revision;
        self.lines = conversation::build_lines(width, revision, content);
    }

    // Returns the maximum scroll offset so that the last line is visible at the bottom.
    pub(super) fn max_scroll(&self, height: u16) -> u16 {
        let max_scroll = self.lines.len().saturating_sub(usize::from(height));
        u16::try_from(max_scroll).unwrap_or(u16::MAX)
    }

    // Returns the slice of lines that fit in the viewport starting at the given scroll offset.
    pub(super) fn visible_lines(&self, scroll: u16, height: u16) -> Vec<Line<'static>> {
        let start = usize::from(scroll);
        let end = start
            .saturating_add(usize::from(height))
            .min(self.lines.len());
        self.lines[start..end].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_horizon_cli::runtime::ContextUsage;

    #[test]
    fn max_scroll_tracks_lines_beyond_viewport() {
        let mut cache = RenderCache::default();
        cache.update(
            6,
            1,
            &RenderContent::input(
                &[],
                "abcdefghi",
                "",
                0,
                &[],
                0,
                ContextUsage {
                    estimated_messages: 0,
                    context_limit: 4096,
                },
                None,
                None,
            ),
        );

        assert_eq!(cache.max_scroll(2), 1);
    }
}
