/*
 * markdown.ts — the single Markdown pipeline: parse (marked) → highlight
 * (highlight.js, common subset) → sanitize (DOMPurify). Its output is the
 * only HTML ever injected via {@html} (through Markdown.svelte). The
 * pipeline must stay tolerant of partial/streaming input: it never throws;
 * on failure it returns the source HTML-escaped inside a <p>.
 */
import { Marked } from 'marked';
import { markedHighlight } from 'marked-highlight';
import hljs from 'highlight.js/lib/common';
import DOMPurify from 'dompurify';

const marked = new Marked(
  markedHighlight({
    highlight(code: string, lang: string): string {
      // Only languages registered in the common subset are highlighted;
      // everything else renders as plain code (no auto-detection).
      if (lang && hljs.getLanguage(lang)) {
        return hljs.highlight(code, { language: lang }).value;
      }
      return code;
    }
  }),
  { gfm: true, breaks: true }
);

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function renderMarkdown(source: string): string {
  if (source === '') {
    return '';
  }
  try {
    const html = marked.parse(source, { async: false }) as string;
    // Untrusted model output: DOMPurify (default config) strips scripts
    // and disallowed tags before the HTML can reach the DOM.
    return DOMPurify.sanitize(html);
  } catch {
    // Never propagate parser/highlighter exceptions to the UI.
    return `<p>${escapeHtml(source)}</p>`;
  }
}
