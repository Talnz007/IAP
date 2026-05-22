<script lang="ts">
  import { onMount } from 'svelte';

  let currentAqi = $state<number | null>(null);
  let predictedAqi = $state<number | null>(null);
  let status = $state('Loading...');

  onMount(async () => {
    try {
      // In production, configure API URL via env vars
      const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      
      // Basic health check
      const res = await fetch(`${apiBase}/`);
      if (res.ok) {
        status = 'System Online';
        
        // Let's call the actual predict endpoint
        const predictRes = await fetch(`${apiBase}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            pm25: 45.2, 
            pm10: 60.1, 
            temperature: 25.0, 
            humidity: 40.0,
            no2: 20.0,
            so2: 10.0,
            co: 0.5,
            o3: 30.0
          })
        });
        
        if (predictRes.ok) {
            const data = await predictRes.json();
            predictedAqi = data.aqi_prediction;
        }
      } else {
        status = 'Backend Unreachable';
      }
    } catch (e) {
      status = 'System Offline';
    }
  });
</script>

<svelte:head>
  <title>AQI Dashboard - Islamabad</title>
</svelte:head>

<div class="dashboard">
  <section class="status-panel">
    <div class="status-indicator">
      <span class="dot" class:online={status === 'System Online'}></span>
      <span class="status-text">{status}</span>
    </div>
  </section>

  <div class="metrics-grid">
    <div class="card metric-card">
      <div class="data-value">{currentAqi ?? '--'}</div>
      <div class="data-label">Current AQI (Live)</div>
    </div>
    
    <div class="card metric-card">
      <div class="data-value">{predictedAqi !== null ? predictedAqi.toFixed(1) : '--'}</div>
      <div class="data-label">Predicted AQI (Next 24h)</div>
    </div>
  </div>
</div>

<style>
  .dashboard {
    display: flex;
    flex-direction: column;
    gap: 2rem;
  }
  
  .status-panel {
    display: flex;
    justify-content: flex-end;
  }
  
  .status-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: var(--font-mono);
    font-size: 0.875rem;
    color: var(--text-secondary);
  }
  
  .dot {
    width: 8px;
    height: 8px;
    background-color: var(--text-muted);
  }
  
  .dot.online {
    background-color: var(--accent);
  }
  
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
  }
</style>
