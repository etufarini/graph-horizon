/*
 * systemPrompt.ts — isolates the localStorage side effects for the system
 * prompt under the key 'gh-zero.system-prompt'. Both functions are
 * exception-safe by contract: storage may be unavailable (private browsing,
 * quota, disabled), so every access is wrapped and failures are absorbed.
 */
const STORAGE_KEY = 'gh-zero.system-prompt';

export function loadSystemPrompt(): string {
  try {
    // The try wraps the localStorage property access itself: in some
    // environments even touching window.localStorage throws.
    return window.localStorage.getItem(STORAGE_KEY) ?? '';
  } catch {
    return '';
  }
}

export function saveSystemPrompt(text: string): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, text);
  } catch {
    // Quota or disabled storage: the value stays in memory for the session.
  }
}
