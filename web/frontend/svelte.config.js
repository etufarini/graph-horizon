/*
 * Svelte compiler configuration: use Vite preprocessing for the component
 * TypeScript and styles owned by this frontend.
 */

import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

export default {
  preprocess: vitePreprocess()
};
