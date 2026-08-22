/*
 * Graph Horizon CLI semantic palette
 * Owns the mapping from terminal content roles to true-color styles. Rendering
 * layout, terminal state, and runtime theme selection remain outside this file.
 */

use ratatui::prelude::*;
use std::borrow::Cow;

// Bright Azure roles aligned with the logo and Web UI semantic palette. The
// dark logo navy is intentionally excluded because terminals may be dark.
const COLOR_INPUT: Color = Color::Rgb(0, 150, 220);
const COLOR_RESPONSE: Color = Color::Rgb(65, 190, 245);
const COLOR_SECONDARY: Color = Color::Rgb(65, 190, 245);
const COLOR_HINT: Color = Color::Rgb(0, 150, 220);
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
    fn azure_palette_maps_roles_to_exact_styles() {
        for (role, color, italic) in [
            (Palette::Input, Color::Rgb(0, 150, 220), false),
            (Palette::Response, Color::Rgb(65, 190, 245), false),
            (Palette::Secondary, Color::Rgb(65, 190, 245), true),
            (Palette::Hint, Color::Rgb(0, 150, 220), false),
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
