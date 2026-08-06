/*
 * Frontend build configuration: compile the Svelte application into the static
 * directory embedded by the Rust web surface.
 */

import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: 'dist',
    emptyOutDir: true
  }
});
