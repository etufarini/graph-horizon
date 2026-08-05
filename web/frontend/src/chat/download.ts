/*
 * download.ts — isolates the Blob/anchor browser-download side effect and
 * owns the exported chat file naming (graph-orizon-chat-<timestamp>.json),
 * mirroring how client.ts isolates fetch and systemPrompt.ts localStorage.
 * The function trusts its caller: no validation of the JSON text.
 */
export function downloadChatFile(jsonText: string): void {
  const now = new Date();
  const pad = (value: number) => String(value).padStart(2, '0');
  // Local time, zero-padded: graph-orizon-chat-YYYYMMDD-HHMMSS.json. Same-second
  // collisions are left to the browser's save deduplication.
  const timestamp =
    `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}` +
    `-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;

  const url = URL.createObjectURL(new Blob([jsonText], { type: 'application/json' }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `graph-orizon-chat-${timestamp}.json`;
  // A detached anchor click suffices in all supported browsers; the anchor
  // is never appended to the document.
  anchor.click();
  URL.revokeObjectURL(url);
}
