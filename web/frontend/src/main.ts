/*
 * Frontend entry point: mount the sole application component into the static
 * document root and fail clearly when the hosting template is malformed.
 */

import { mount } from 'svelte';
import App from './App.svelte';

const target = document.getElementById('app');

if (!target) {
  throw new Error('app root missing');
}

mount(App, { target });
