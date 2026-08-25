<!--
Chat header: present only the product and immutable runtime identity. Workspace
panel controls belong to the panels themselves so this header remains a narrow
orientation boundary.
-->
<script lang="ts">
  // @ts-expect-error Vite resolves this local asset and fails the build if it is missing.
  import logoUrl from '../../../../../assets/graph-horizon-logo.svg';
  import Runtime from '../Runtime.svelte';
  import type { RuntimeInfo } from '../../chat/types';

  export let runtimeInfo: RuntimeInfo | null = null;
</script>

<header>
  <div class="identity">
    <img src={logoUrl} alt="" aria-hidden="true" />
    <h1><span>Graph Horizon</span><small>Local inference</small></h1>
  </div>
  {#if runtimeInfo}<Runtime info={runtimeInfo} />{/if}
</header>

<style lang="scss">
  header { min-width: 0; display: flex; flex-wrap: wrap; align-items: center; gap: var(--gn-space-xs) var(--gn-space-lg); border-bottom: var(--gn-border-width) solid var(--gn-border); padding: 0 var(--gn-header-end-reserve) var(--gn-space-sm) var(--gn-header-start-reserve); transition: padding var(--gn-motion-fast) ease; }
  .identity { min-width: 0; display: flex; align-items: center; gap: var(--gn-space-sm); }
  header > :global(.runtime) { min-width: 0; flex: 1 1 360px; justify-content: flex-end; }
  img { width: var(--gn-brand-logo-size); height: var(--gn-brand-logo-size); flex: 0 0 auto; object-fit: contain; }
  h1 { min-width: 0; margin: 0; display: flex; align-items: baseline; gap: var(--gn-space-sm); color: var(--gn-text-primary); font-size: var(--gn-text-md); line-height: var(--gn-line-height-tight); }
  h1 span { font-family: var(--gn-font-mono); font-weight: 750; letter-spacing: 0.08em; text-transform: uppercase; }
  h1 small { color: var(--gn-text-muted); font-family: var(--gn-font-mono); font-size: var(--gn-text-xs); font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; }
  @media (max-width: 640px) {
    h1 small { display: none; }
    header > :global(.runtime) { flex-basis: 100%; justify-content: flex-start; }
  }
</style>
