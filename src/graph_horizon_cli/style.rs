/*
 * Graph Horizon CLI semantic palette
 * Owns the mapping from terminal content roles to true-color styles. Rendering
 * layout, terminal state, and runtime theme selection remain outside this file.
 */

use ratatui::prelude::*;
use std::borrow::Cow;

// Bright Mistral roles aligned with the existing Web UI semantic palette.
const COLOR_INPUT: Color = Color::Rgb(255, 82, 41);
const COLOR_RESPONSE: Color = Color::Rgb(68, 186, 130);
const COLOR_SECONDARY: Color = Color::Rgb(85, 179, 251);
const COLOR_HINT: Color = Color::Rgb(255, 175, 1);
const COLOR_ERROR: Color = Color::Rgb(229, 19, 0);

// The visual roles the interface can paint. Each maps to one palette entry; the
// enum keeps the call sites self-describing while the styling lives in one place.
#[derive(Clone, Copy)]
pub(crate) enum Palette {
    Input,
    Secondary,
    Hint,
    Error,
    Response,
}

// Styles `text` for the given visual role. Single source of the palette so a
// colour or attribute change touches exactly one arm.
pub(crate) fn format<'a>(palette: Palette, text: impl Into<Cow<'a, str>>) -> Span<'a> {
    let style = match palette {
        Palette::Input => Style::new().fg(COLOR_INPUT),
        Palette::Secondary => Style::new().fg(COLOR_SECONDARY).italic(),
        Palette::Hint => Style::new().fg(COLOR_HINT),
        Palette::Error => Style::new().fg(COLOR_ERROR),
        Palette::Response => Style::new().fg(COLOR_RESPONSE),
    };
    Span::styled(text, style)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mistral_palette_maps_roles_to_exact_styles() {
        for (role, color, italic) in [
            (Palette::Input, Color::Rgb(255, 82, 41), false),
            (Palette::Response, Color::Rgb(68, 186, 130), false),
            (Palette::Secondary, Color::Rgb(85, 179, 251), true),
            (Palette::Hint, Color::Rgb(255, 175, 1), false),
            (Palette::Error, Color::Rgb(229, 19, 0), false),
        ] {
            let span = format(role, "text");
            let expected = if italic {
                Style::new().fg(color).italic()
            } else {
                Style::new().fg(color)
            };
            assert_eq!(span.style, expected);
        }
    }
}
