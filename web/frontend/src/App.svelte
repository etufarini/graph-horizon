<!--
  Application shell: load the global visual layers and host the single chat
  surface. Conversation state and transport remain inside Chat.
-->
<script lang="ts">
  import Chat from './components/Chat.svelte';
  import './theme/tokens.scss';
  import './theme/global.scss';
  import './theme/markdown.scss';
</script>

<main class="app-shell">
  <Chat />
</main>
