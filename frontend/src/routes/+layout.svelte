<script lang="ts">
  import '../app.css';
  import { base } from '$app/paths';
  let { children } = $props();
</script>

<div class="app-container">
  <header class="app-header">
    <h1>Islamabad AQI Predictor</h1>
    <nav>
      <a href="{base}/">Dashboard</a>
    </nav>
  </header>

  <main>
    {@render children()}
  </main>
</div>

<style>
  .app-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
  }
  .app-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding-bottom: 2rem;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 2rem;
  }
  .app-header h1 {
    font-size: 1.5rem;
    margin: 0;
    color: var(--accent);
  }
</style>
