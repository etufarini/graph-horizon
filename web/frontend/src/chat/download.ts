/*
 * download.ts — isolates the Blob/anchor browser-download side effect and
 * owns the exported chat file naming (graph-horizon-chat-<timestamp>.json),
 * mirroring how client.ts isolates fetch. The function trusts its caller: no
 * validation of the JSON text.
 */
export function downloadChatFile(jsonText: string): void {
  const now = new Date();
  const pad = (value: number) => String(value).padStart(2, '0');
  // Local time, zero-padded: graph-horizon-chat-YYYYMMDD-HHMMSS.json. Same-second
  // collisions are left to the browser's save deduplication.
  const timestamp =
    `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}` +
    `-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;

  downloadText(jsonText, `graph-horizon-chat-${timestamp}.json`, 'application/json');
}

export function downloadMarkdownFile(name: string, content: string): void {
  downloadText(content, name, 'text/markdown;charset=utf-8');
}

function downloadText(content: string, name: string, type: string): void {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  // A detached anchor click suffices in all supported browsers; the anchor
  // is never appended to the document.
  anchor.click();
  URL.revokeObjectURL(url);
}
