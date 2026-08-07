/*
 * Graph Horizon CLI Modules - Console - Render - Draw
 * Single responsibility: draw the conversation viewport and one right-aligned
 * context/duration or capacity-error status row.
 */

use ratatui::DefaultTerminal;
use ratatui::{prelude::*, widgets::Paragraph};

use super::super::scroll::{ViewportState, sync_scroll_state};
use super::status::{capacity_error, context_status};
use super::{RenderCache, RenderContent};
use crate::graph_horizon_cli::style;
use color_eyre::eyre::Result;

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
            let (palette, bar) = match content.capacity_error {
                Some(error) => (style::Palette::Error, capacity_error(error)),
                None => (
                    style::Palette::Hint,
                    context_status(content.usage, content.duration),
                ),
            };
            f.render_widget(
                Paragraph::new(Line::from(style::format(palette, bar))).alignment(Alignment::Right),
                status,
            );
        }

        sync_scroll_state(viewport, max_scroll);
    })?;

    Ok(())
}
