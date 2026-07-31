/*
 * GH Zero CLI Modules - Console - Render - Draw
 * Draws one terminal frame: refreshes the line cache, renders the visible slice, syncs scroll.
 * Composes the status bar from the status-label helpers; owns no persistent state.
 */

use ratatui::DefaultTerminal;
use ratatui::{prelude::*, widgets::Paragraph};

use super::super::scroll::{ViewportState, sync_scroll_state};
use super::status::{count_label, speed_label};
use super::{RenderCache, RenderContent, TokenStatus};
use crate::gh_zero_cli::style;
use color_eyre::eyre::Result;

// Status-bar text shown bottom-right before the token count: the off label while
// pruning is disabled, the "∞" infinite-chat marker while it is active. No
// trailing gap: the bar joins every field with a single space (see draw_viewport),
// so spacing stays uniform across the whole row. Kept as editable constants.
const PRUNING_OFF_LABEL: &str = "pruning off";
const INFINITE_LABEL: &str = "∞";

// Draws one terminal frame: rebuilds the line cache, renders the visible slice at the current
// scroll offset, and synchronizes the viewport with the rendered document.
pub(crate) fn draw_viewport(
    terminal: &mut DefaultTerminal,
    document: &mut RenderCache,
    content_revision: u64,
    content: &RenderContent<'_>,
    viewport: &mut ViewportState,
) -> Result<()> {
    terminal.draw(|f| {
        let area = f.area();

        // Reserve the bottom row for the status bar. saturating_sub keeps the
        // chat area height valid even when the area is only 0 or 1 row tall, in
        // which case there is simply no room for the status line.
        let status_height = u16::from(area.height > 1);
        let chat = Rect {
            height: area.height.saturating_sub(status_height),
            ..area
        };

        document.update(chat.width, content_revision, content);
        let max_scroll = document.max_scroll(chat.height);
        let scroll = viewport.manual_scroll.unwrap_or(max_scroll).min(max_scroll);
        let visible_lines = document.visible_lines(scroll, chat.height);

        f.render_widget(Paragraph::new(Text::from(visible_lines)), chat);

        if status_height == 1 {
            let status = Rect {
                y: chat.y + chat.height,
                height: 1,
                ..area
            };
            // The bottom-right bar, left to right: speed rates, the last turn's
            // token breakdown, the pruning label, and the conversation total. Each
            // is a plain-grey field; empty fields drop out (speed while connecting,
            // counts before the first reply), and the rest are joined by a single
            // space so every gap on the row — glyph-to-number included — is the same
            // width. One span keeps the whole bar one uniform run of text.
            let label = match content.status {
                TokenStatus::Off => PRUNING_OFF_LABEL,
                TokenStatus::Active => INFINITE_LABEL,
            };
            let bar = [
                speed_label(content.throughput),
                count_label(content.output_tokens),
                label.to_string(),
                format!("~{:.1}k tok", content.token_count as f64 / 1000.0),
            ]
            .into_iter()
            .filter(|field| !field.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
            f.render_widget(
                Paragraph::new(Line::from(style::format(style::Palette::Hint, bar)))
                    .alignment(Alignment::Right),
                status,
            );

            // Bottom-left: the raw context window as a fixed pruning reference,
            // exact (no "~", unlike the estimated count on the right). Rendered as
            // a separate left-aligned Paragraph over the same row; with a sane
            // terminal width the two never overlap. A None or zero limit (pruning
            // off) leaves the left side empty.
            if let Some(limit) = content.context_limit.filter(|&l| l > 0) {
                let ctx = format!("ctx {:.1}k", limit as f64 / 1000.0);
                f.render_widget(
                    Paragraph::new(Line::from(style::format(style::Palette::Hint, ctx)))
                        .alignment(Alignment::Left),
                    status,
                );
            }
        }

        sync_scroll_state(viewport, max_scroll);
    })?;

    Ok(())
}
