<script lang="ts">
  import { onMount } from 'svelte';
  import { fade, fly } from 'svelte/transition';
  import { spring } from 'svelte/motion';
  import { MapLibre, Marker, Popup } from 'svelte-maplibre';
  import { Wind, Droplets, Thermometer, AlertTriangle, Info, CloudFog, TrendingUp, Activity } from 'lucide-svelte';
  import Particles from '@tsparticles/svelte';
  import { loadSlim } from '@tsparticles/slim';

  // ─── State ───────────────────────────────────────────────────────────────────
  let mounted = $state(false);
  let apiOnline = $state(false);
  let loading = $state(true);

  type PredictionResult = {
    city: string;
    current_aqi: number | null;
    predicted_aqi_24h: number;
    aqi_category: string;
    health_advisory: string;
    model_used: string;
  };
  
  type AccuracyResult = {
    accuracy_score: number | null;
    history: { date: string; predicted: number; actual: number | null }[];
  };

  type ExplainResult = {
    top_drivers: { feature: string; importance: number; current_value: number }[];
  };

  let prediction = $state<PredictionResult | null>(null);
  let accuracy = $state<AccuracyResult | null>(null);
  let explanation = $state<ExplainResult | null>(null);
  let error = $state<string | null>(null);

  const CITY = 'Islamabad';
  const LAT = 33.6844;
  const LNG = 73.0479;
  const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

  // Spring-loaded values for smooth counting animation
  const animatedAqi = spring(0, { stiffness: 0.1, damping: 0.6 });
  const animatedScore = spring(0, { stiffness: 0.1, damping: 0.6 });

  // ─── Helpers ─────────────────────────────────────────────────────────────────
  function aqiColor(aqi: number | null): string {
    if (aqi === null) return '#7A7670';
    if (aqi <= 50)  return '#4ade80';
    if (aqi <= 100) return '#facc15';
    if (aqi <= 150) return '#fb923c';
    if (aqi <= 200) return '#f87171';
    if (aqi <= 300) return '#c084fc';
    return '#f43f5e';
  }

  function aqiLabel(aqi: number | null): string {
    if (aqi === null) return 'Unknown';
    if (aqi <= 50)  return 'Good';
    if (aqi <= 100) return 'Moderate';
    if (aqi <= 150) return 'Unhealthy for Sensitive Groups';
    if (aqi <= 200) return 'Unhealthy';
    if (aqi <= 300) return 'Very Unhealthy';
    return 'Hazardous';
  }

  function formatFeatureName(name: string): string {
    const map: Record<string, string> = {
      'temp': 'Temperature',
      'feels_like': 'Feels Like Temp',
      'humidity': 'Humidity',
      'pressure': 'Atmospheric Pressure',
      'wind_speed': 'Wind Speed',
      'pm2_5': 'PM2.5 Particles',
      'pm10': 'PM10 Particles',
      'aqi': 'Historical AQI',
      'clouds': 'Cloud Cover',
      'visibility': 'Visibility'
    };
    return map[name] || name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  function getFeatureIcon(name: string) {
    if (name.includes('wind')) return Wind;
    if (name.includes('temp') || name.includes('feels')) return Thermometer;
    if (name.includes('humid')) return Droplets;
    if (name.includes('pm') || name.includes('aqi')) return CloudFog;
    return Activity;
  }

  // ─── API ─────────────────────────────────────────────────────────────────────
  const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  async function fetchData() {
    loading = true;
    error = null;
    try {
      const [predRes, accRes, expRes] = await Promise.all([
        fetch(`${apiBase}/predict/${CITY}`),
        fetch(`${apiBase}/accuracy/${CITY}`),
        fetch(`${apiBase}/explain/${CITY}`)
      ]);

      if (!predRes.ok) throw new Error(`API error ${predRes.status}`);
      
      prediction = await predRes.json();
      animatedAqi.set(prediction?.current_aqi ?? 0);

      if (accRes.ok) {
        accuracy = await accRes.json();
        if (accuracy?.accuracy_score) {
          animatedScore.set(accuracy.accuracy_score);
        }
      }
      
      if (expRes.ok) {
        explanation = await expRes.json();
      }

    } catch (e: any) {
      error = e.message ?? 'Unknown error';
    } finally {
      loading = false;
    }
  }

  onMount(async () => {
    mounted = true;
    try {
      const health = await fetch(`${apiBase}/health`);
      apiOnline = health.ok;
    } catch {
      apiOnline = false;
    }
    fetchData();
  });

  // ─── Particles Config ────────────────────────────────────────────────────────
  let particlesInit = async (engine: any) => {
    await loadSlim(engine);
  };

  const particlesOptions = $derived({
    fpsLimit: 60,
    particles: {
      color: { value: aqiColor(prediction?.current_aqi ?? null) },
      move: {
        direction: "right",
        enable: true,
        random: true,
        speed: (prediction?.current_aqi ?? 50) > 150 ? 0.3 : 0.8, // Smog moves slow, clear air moves fast
        straight: false,
      },
      number: {
        density: { enable: true, area: 800 },
        value: Math.min((prediction?.current_aqi ?? 50) * 0.8, 300), // More particles for worse AQI
      },
      opacity: {
        value: { min: 0.1, max: 0.4 },
      },
      shape: { type: "circle" },
      size: { value: { min: 1, max: 4 } },
    },
    detectRetina: true,
  });

</script>

<svelte:head>
  <title>AQI Observatory — Islamabad</title>
</svelte:head>

{#if mounted}
  <!-- Atmospheric Background -->
  <div class="particles-bg" in:fade={{ duration: 2000 }}>
    <Particles id="tsparticles" options={particlesOptions} particlesInit={particlesInit} />
  </div>

  <div class="app-wrapper">
    <!-- ── Header ─────────────────────────────────────────────────────────────── -->
    <header in:fade={{ duration: 900, delay: 100 }} class="observatory-header">
      <div class="header-inner">
        <div class="wordmark">
          <Activity size={24} color="var(--accent)" />
          <span>AQI Observatory</span>
        </div>
        <div class="status-pill" class:online={apiOnline}>
          <span class="dot"></span>
          <span>{apiOnline ? 'Observatory Online' : 'Backend Offline'}</span>
        </div>
      </div>

      <div class="hero-text">
        <h1>Islamabad Atmosphere</h1>
        <p class="subtitle">
          Real-time monitoring, MLOps performance tracking, and explainable AI 
          for particulate matter intelligence.
        </p>
      </div>
    </header>

    <!-- ── Main grid ──────────────────────────────────────────────────────────── -->
    <main class="main-grid">

      <!-- LEFT COLUMN -->
      <div class="left-col">
        
        <!-- Live Map & Primary Metric -->
        <section in:fly={{ y: 20, duration: 800, delay: 200 }} class="map-section card">
          <div class="map-header">
            <h2><CloudFog size={16}/> Live Location</h2>
            {#if prediction}
              <span class="aqi-pill" style="background:{aqiColor(prediction.current_aqi)}22; color:{aqiColor(prediction.current_aqi)}; border-color:{aqiColor(prediction.current_aqi)}55">
                AQI {$animatedAqi.toFixed(0)} · {aqiLabel(prediction.current_aqi)}
              </span>
            {/if}
          </div>
          <div class="map-container">
            <MapLibre
              style={MAP_STYLE}
              center={[LNG, LAT]}
              zoom={10.5}
              class="map"
              attributionControl={false}
            >
              <Marker lngLat={[LNG, LAT]}>
                <div
                  class="map-marker pulse-marker"
                  style="--aqi-color: {aqiColor(prediction?.current_aqi ?? null)}"
                >
                  <span class="marker-label">
                    {$animatedAqi.toFixed(0)}
                  </span>
                </div>
              </Marker>
            </MapLibre>
          </div>
          
          {#if prediction}
            <div class="advisory-box" style="border-left-color: {aqiColor(prediction.current_aqi)}">
              <Info size={20} color={aqiColor(prediction.current_aqi)} />
              <p>{prediction.health_advisory}</p>
            </div>
          {/if}
        </section>

        <!-- Explainable AI (Drivers) -->
        {#if explanation && explanation.top_drivers.length > 0}
          <section in:fly={{ y: 20, duration: 800, delay: 400 }} class="xai-section card">
            <div class="card-header">
              <h2><Activity size={18}/> Air Quality Drivers</h2>
              <span class="header-tag">Explainable AI</span>
            </div>
            <p class="section-desc">Machine learning feature importance driving the current prediction.</p>
            
            <div class="drivers-list">
              {#each explanation.top_drivers.slice(0, 5) as driver}
                {@const Icon = getFeatureIcon(driver.feature)}
                <div class="driver-item">
                  <div class="driver-info">
                    <Icon size={16} color="var(--text-muted)" />
                    <span class="driver-name">{formatFeatureName(driver.feature)}</span>
                    <span class="driver-val">{driver.current_value.toFixed(1)}</span>
                  </div>
                  <div class="importance-bar-bg">
                    <div class="importance-bar-fill" style="width: {Math.min(driver.importance * 100 * 5, 100)}%; background: var(--accent);"></div>
                  </div>
                </div>
              {/each}
            </div>
          </section>
        {/if}
      </div>

      <!-- RIGHT COLUMN (MLOps) -->
      <div class="right-col">
        
        {#if loading}
          <div in:fade class="card loading-card">
            <div class="loading-spinner"></div>
            <p>Analyzing atmospheric data…</p>
          </div>
        {:else if error}
          <div in:fade class="card error-card">
            <AlertTriangle size={32} color="#f87171" />
            <p>{error}</p>
            <button class="retry-btn" onclick={() => fetchData()}>Retry</button>
          </div>
        {:else if prediction}
          
          <!-- 24h Forecast -->
          <div in:fly={{ y: 16, duration: 700, delay: 300 }} class="card metric-card">
            <div class="card-header">
              <h2><TrendingUp size={18}/> 24h Prediction</h2>
            </div>
            <div class="data-value aqi-value" style="color: {aqiColor(prediction.predicted_aqi_24h)}">
              {prediction.predicted_aqi_24h.toFixed(1)}
            </div>
            <span class="aqi-category-badge"
              style="background:{aqiColor(prediction.predicted_aqi_24h)}22; color:{aqiColor(prediction.predicted_aqi_24h)}; border-color:{aqiColor(prediction.predicted_aqi_24h)}44">
              {aqiLabel(prediction.predicted_aqi_24h)}
            </span>
          </div>

          <!-- MLOps Accuracy Tracking -->
          {#if accuracy && accuracy.history.length > 0}
            <div in:fly={{ y: 16, duration: 700, delay: 450 }} class="card chart-card">
              <div class="card-header">
                <h2>Model Accuracy</h2>
                <div class="accuracy-score">
                  {$animatedScore.toFixed(1)}%
                </div>
              </div>
              <p class="section-desc">Predicted vs Actual AQI over the last 7 days.</p>
              
              <!-- History List (Fallback for simple UI) -->
              <div class="history-list">
                {#each accuracy.history.slice(-5).reverse() as day}
                  <div class="history-row">
                    <span class="hist-date">{day.date.slice(5)}</span>
                    <div class="hist-bars">
                      <div class="hist-metric">
                        <span class="hist-label">Pred</span>
                        <span class="hist-val" style="color: var(--accent)">{day.predicted.toFixed(0)}</span>
                      </div>
                      <div class="hist-metric">
                        <span class="hist-label">Act</span>
                        <span class="hist-val" style="color: {day.actual ? aqiColor(day.actual) : 'var(--text-muted)'}">
                          {day.actual ? day.actual.toFixed(0) : '—'}
                        </span>
                      </div>
                    </div>
                  </div>
                {/each}
              </div>
            </div>
          {/if}

        {/if}
      </div>
    </main>

    <footer in:fade={{ duration: 600, delay: 800 }} class="observatory-footer">
      <p>10Pearls Capstone · Data via OpenAQ · Predictions via Random Forest ML</p>
    </footer>
  </div>
{/if}

<style>
  /* ── Layout & Theme ──────────────────────────────────────────────────────── */
  :global(body) {
    background-color: var(--bg-base, #111315);
    color: var(--text-primary, #E6E4D9);
    margin: 0;
    font-family: 'Inter', system-ui, sans-serif;
  }

  .particles-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 0;
    pointer-events: none;
  }

  .app-wrapper {
    position: relative;
    z-index: 1;
    min-height: 100vh;
    background: radial-gradient(circle at 50% 0%, rgba(28, 32, 36, 0.4) 0%, transparent 60%);
  }

  .observatory-header {
    max-width: 1200px;
    margin: 0 auto;
    padding: 3rem 2rem 1.5rem;
  }

  .header-inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2.5rem;
  }

  .wordmark {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    color: var(--accent);
    letter-spacing: 0.04em;
  }

  .hero-text {
    text-align: left;
    max-width: 800px;
  }

  .hero-text h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2rem, 4vw, 3.5rem);
    margin-bottom: 0.5rem;
    font-weight: 500;
  }

  .subtitle {
    color: var(--text-secondary);
    font-size: 1.05rem;
    line-height: 1.6;
    margin: 0;
  }

  .status-pill {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8rem;
    color: var(--text-muted);
    padding: 0.4rem 0.9rem;
    background: rgba(28, 32, 36, 0.8);
    backdrop-filter: blur(8px);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-muted);
    transition: background 1s ease, box-shadow 1s ease;
  }

  .status-pill.online .dot {
    background: #4ade80;
    box-shadow: 0 0 8px rgba(74, 222, 128, 0.5);
  }

  /* ── Grid ────────────────────────────────────────────────────────────────── */
  .main-grid {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1.5rem 2rem 3rem;
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 1.5rem;
    align-items: start;
  }

  @media (max-width: 900px) {
    .main-grid { grid-template-columns: 1fr; }
  }

  .left-col, .right-col {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  /* ── Cards Shared ────────────────────────────────────────────────────────── */
  .card {
    background: rgba(28, 32, 36, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-subtle);
    padding: 1.5rem;
    border-radius: 8px;
  }

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.5rem;
  }

  .card-header h2 {
    font-size: 0.85rem;
    color: var(--text-secondary);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .header-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent);
    border: 1px solid var(--border-subtle);
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
  }

  .section-desc {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 0;
    margin-bottom: 1.5rem;
  }

  /* ── Map ─────────────────────────────────────────────────────────────────── */
  .map-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
  }

  .map-header h2 {
    font-size: 0.85rem;
    color: var(--text-secondary);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .map-container {
    height: 380px;
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid var(--border-subtle);
    margin-bottom: 1rem;
  }

  :global(.map) { width: 100%; height: 100%; }

  .map-marker {
    width: 44px;
    height: 44px;
    border-radius: 50% 50% 50% 0;
    transform: rotate(-45deg);
    background: var(--bg-surface, #1C2024);
    border: 2px solid var(--aqi-color);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 24px color-mix(in srgb, var(--aqi-color) 40%, transparent), 0 4px 12px rgba(0,0,0,0.5);
  }

  .marker-label {
    transform: rotate(45deg);
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
  }

  .advisory-box {
    display: flex;
    gap: 1rem;
    padding: 1rem;
    background: rgba(0,0,0,0.2);
    border-left: 3px solid;
    border-radius: 4px;
    margin-top: 1rem;
  }

  .advisory-box p {
    margin: 0;
    font-size: 0.9rem;
    line-height: 1.5;
    color: var(--text-primary);
  }

  .aqi-pill {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.3rem 0.8rem;
    border-radius: 12px;
    border: 1px solid;
    letter-spacing: 0.03em;
  }

  /* ── Explainable AI (XAI) ────────────────────────────────────────────────── */
  .drivers-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .driver-item {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }

  .driver-info {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
  }
  
  .driver-name {
    flex: 1;
    color: var(--text-primary);
  }

  .driver-val {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-secondary);
    font-size: 0.8rem;
  }

  .importance-bar-bg {
    height: 4px;
    background: var(--border-subtle);
    border-radius: 2px;
    overflow: hidden;
  }

  .importance-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
  }

  /* ── Metrics ─────────────────────────────────────────────────────────────── */
  .aqi-value {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    line-height: 1;
    margin: 1rem 0 0.5rem;
  }

  .aqi-category-badge {
    display: inline-block;
    font-size: 0.75rem;
    letter-spacing: 0.04em;
    padding: 0.25rem 0.65rem;
    border-radius: 4px;
    border: 1px solid;
    width: fit-content;
    font-weight: 600;
  }

  /* ── MLOps Accuracy ──────────────────────────────────────────────────────── */
  .accuracy-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2rem;
    color: #4ade80;
    font-weight: 600;
  }

  .history-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .history-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-subtle);
  }
  
  .history-row:last-child {
    border-bottom: none;
    padding-bottom: 0;
  }

  .hist-date {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
  }

  .hist-bars {
    display: flex;
    gap: 1.5rem;
  }

  .hist-metric {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
  }

  .hist-label {
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .hist-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 600;
  }

  /* ── Loading & Error ─────────────────────────────────────────────────────── */
  .loading-card, .error-card {
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 4rem 2rem;
  }

  .loading-spinner {
    width: 28px;
    height: 28px;
    border: 2px solid var(--border-subtle);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 1rem;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  .retry-btn {
    background: rgba(255,255,255,0.1);
    border: 1px solid var(--border-strong);
    color: white;
    padding: 0.4rem 1.1rem;
    border-radius: 4px;
    cursor: pointer;
    margin-top: 1rem;
  }

  /* ── Footer ──────────────────────────────────────────────────────────────── */
  .observatory-footer {
    text-align: center;
    padding: 2rem;
    border-top: 1px solid var(--border-subtle);
    font-size: 0.8rem;
    color: var(--text-muted);
    max-width: 1200px;
    margin: 0 auto;
  }
</style>
