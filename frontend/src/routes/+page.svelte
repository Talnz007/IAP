<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { fade, fly } from 'svelte/transition';
  import { MapLibre, Marker, FillExtrusionLayer } from 'svelte-maplibre';
  import { Wind, Droplets, Thermometer, AlertTriangle, Info, CloudFog, TrendingUp, Activity, LayoutGrid, LayoutList, Gauge, Zap } from 'lucide-svelte';
  
  import AQIGauge from '$lib/components/AQIGauge.svelte';

  // ─── State ───────────────────────────────────────────────────────────────────
  let mounted = $state(false);
  let apiOnline = $state(false);
  let loading = $state(true);

  type PredictionResult = {
    city: string;
    prediction_time: string;
    current_aqi: number | null;
    predicted_aqi_24h: number;
    aqi_category: string;
    health_advisory: string;
    model_used: string;
    temperature: number | null;
    humidity: number | null;
    pm2_5: number | null;
    pm10: number | null;
    co: number | null;
    o3: number | null;
    no2: number | null;
    so2: number | null;
    wind_speed: number | null;
    clouds: number | null;
  };
  
  let modelsInfo = $state<any>(null);
  let selectedModel = $state<string>('random_forest');
  let viewMode = $state<'single' | 'compare'>('single');
  let comparePredictions = $state<Record<string, PredictionResult | null>>({});

  let prediction = $state<PredictionResult | null>(null);
  let explanation = $state<any | null>(null);
  let accuracy = $state<any | null>(null);
  let error = $state<string | null>(null);

  const CITY = 'Islamabad';
  const LAT = 33.6844;
  const LNG = 73.0479;
  const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

  // ─── Helpers ─────────────────────────────────────────────────────────────────
  function aqiColor(aqi: number | null): string {
    if (aqi === null) return '#7A7670';
    if (aqi <= 50)  return '#4ade80';
    if (aqi <= 100) return '#facc15';
    if (aqi <= 150) return '#fb923c';
    if (aqi <= 200) return '#f87171';
    if (aqi <= 300) return '#c084fc';
    return '#881337';
  }

  function formatFeatureName(name: string): string {
    const map: Record<string, string> = {
      'temp': 'Temperature', 'feels_like': 'Feels Like', 'humidity': 'Humidity',
      'pressure': 'Pressure', 'wind_speed': 'Wind Speed', 'pm2_5': 'PM2.5',
      'pm10': 'PM10', 'aqi': 'AQI', 'clouds': 'Cloud Cover', 'visibility': 'Visibility'
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

  async function fetchModels() {
    try {
      const res = await fetch(`${apiBase}/models`);
      if (res.ok) {
        modelsInfo = await res.json();
        if (modelsInfo && !modelsInfo[selectedModel]) {
          selectedModel = Object.keys(modelsInfo)[0] || 'random_forest';
        }
      }
    } catch (e) {
      console.warn("Failed to fetch models", e);
    }
  }

  async function fetchSinglePrediction(model: string) {
    loading = true;
    error = null;
    try {
      const [predRes, expRes, accRes] = await Promise.all([
        fetch(`${apiBase}/predict/${CITY}?model_name=${model}`),
        fetch(`${apiBase}/explain/${CITY}?model_name=${model}`),
        fetch(`${apiBase}/accuracy/${CITY}?model_name=${model}`)
      ]);

      if (!predRes.ok) throw new Error(`API error ${predRes.status}`);
      prediction = await predRes.json();
      
      if (expRes.ok) explanation = await expRes.json();
      if (accRes.ok) accuracy = await accRes.json();
    } catch (e: any) {
      error = e.message ?? 'Unknown error';
    } finally {
      loading = false;
    }
  }

  let compareAccuracy = $state<Record<string, any>>({});

  async function fetchComparePredictions() {
    if (!modelsInfo) return;
    loading = true;
    const modelNames = Object.keys(modelsInfo);
    for (const model of modelNames) {
      try {
        const [predRes, accRes] = await Promise.all([
          fetch(`${apiBase}/predict/${CITY}?model_name=${model}`),
          fetch(`${apiBase}/accuracy/${CITY}?model_name=${model}`)
        ]);
        
        if (predRes.ok) {
          comparePredictions[model] = await predRes.json();
        }
        if (accRes.ok) {
          compareAccuracy[model] = await accRes.json();
        }
      } catch(e) {
        console.error(e);
      }
    }
    loading = false;
  }

  function switchViewMode(mode: 'single' | 'compare') {
    viewMode = mode;
    if (mode === 'single') fetchSinglePrediction(selectedModel);
    else fetchComparePredictions();
  }

  function handleModelChange() {
    if (viewMode === 'single') fetchSinglePrediction(selectedModel);
  }

  onMount(async () => {
    mounted = true;
    try {
      const health = await fetch(`${apiBase}/health`);
      apiOnline = health.ok;
      if (apiOnline) {
        await fetchModels();
        await fetchSinglePrediction(selectedModel);
      }
    } catch {
      apiOnline = false;
    }
  });

  // ─── Map & Canvas Simulation ────────────────────────────────────────────────
  let mapInstance: any = $state(null);
  let canvasOverlay: HTMLCanvasElement | null = $state(null);
  let animationFrameId: number;
  let particles: any[] = [];
  const NUM_PARTICLES = 250;

  function initParticles() {
    particles = [];
    for (let i = 0; i < NUM_PARTICLES; i++) {
      particles.push({
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        size: Math.random() * 2.5 + 1,
        vx: (Math.random() - 0.5) * 1.5,
        vy: (Math.random() - 0.5) * 1.5
      });
    }
  }

  function renderCanvas() {
    if (!canvasOverlay || !mapInstance) return;
    const ctx = canvasOverlay.getContext('2d');
    if (!ctx) return;
    
    const rect = canvasOverlay.parentElement?.getBoundingClientRect();
    if (rect && (canvasOverlay.width !== rect.width || canvasOverlay.height !== rect.height)) {
      canvasOverlay.width = rect.width;
      canvasOverlay.height = rect.height;
    }

    ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);

    const pm25 = prediction?.pm2_5 ?? 50;
    const windSpeed = prediction?.wind_speed ?? 2;
    const cloudCover = prediction?.clouds ?? 0;
    
    const densityRatio = Math.min(pm25 / 150, 1.0);
    const particleOpacity = 0.15 + densityRatio * 0.5;
    
    // Particulate Haze
    if (densityRatio > 0.2) {
       const gradient = ctx.createRadialGradient(
         canvasOverlay.width/2, canvasOverlay.height/2, 0, 
         canvasOverlay.width/2, canvasOverlay.height/2, canvasOverlay.width
       );
       const hazeColor = pm25 > 200 ? 'rgba(136, 19, 55' : 'rgba(192, 132, 252'; 
       gradient.addColorStop(0, `${hazeColor}, ${densityRatio * 0.25})`);
       gradient.addColorStop(1, 'rgba(0,0,0,0)');
       ctx.fillStyle = gradient;
       ctx.fillRect(0,0,canvasOverlay.width, canvasOverlay.height);
    }

    // Dynamic Clouds
    if (cloudCover > 10) {
      const time = Date.now() * 0.00005 * Math.max(windSpeed, 1);
      ctx.save();
      for(let i=0; i<4; i++) {
         const cx = ((time * 150 * (i+1)) % (canvasOverlay.width + 600)) - 300;
         const cy = (Math.sin(time * 2 + i) * 100) + (canvasOverlay.height * 0.25 * i);
         
         const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 300);
         grad.addColorStop(0, `rgba(255,255,255, ${0.08 * (cloudCover/100)})`);
         grad.addColorStop(1, 'rgba(255,255,255,0)');
         ctx.fillStyle = grad;
         ctx.beginPath();
         ctx.arc(cx, cy, 300, 0, Math.PI * 2);
         ctx.fill();
      }
      ctx.restore();
    }

    // Particles
    ctx.fillStyle = `rgba(184, 169, 154, ${particleOpacity})`;
    const speedMult = (pm25 > 150) ? 0.3 : 1.2; // heavy smog is stagnant
    
    for (let p of particles) {
      p.x += p.vx * speedMult;
      p.y += p.vy * speedMult;
      
      if (p.x < 0) p.x = canvasOverlay.width;
      if (p.x > canvasOverlay.width) p.x = 0;
      if (p.y < 0) p.y = canvasOverlay.height;
      if (p.y > canvasOverlay.height) p.y = 0;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function startAnimation() {
    if (!particles.length) initParticles();
    renderCanvas();
    animationFrameId = requestAnimationFrame(startAnimation);
  }

  function handleMapLoad(e: any) {
    const map = e?.detail?.map || e?.map || e;
    if (!map || !map.getStyle) return;
    mapInstance = map;
    
    map.on('move', renderCanvas);
    startAnimation();
  }

  onDestroy(() => {
    if (animationFrameId) cancelAnimationFrame(animationFrameId);
  });
</script>

<svelte:head>
  <title>Urban Intel | Islamabad AQI</title>
</svelte:head>

{#if mounted}
  <div class="app-wrapper">
    <!-- ── Header ─────────────────────────────────────────────────────────────── -->
    <header in:fade={{ duration: 900 }} class="observatory-header">
      <div class="header-inner">
        <div class="wordmark">
          <Activity size={24} color="var(--accent)" />
          <span>Urban Intel</span>
        </div>
        
        <div class="header-controls">
          <div class="view-toggles">
            <button class:active={viewMode === 'single'} onclick={() => switchViewMode('single')} title="Single Model View">
              <LayoutGrid size={16} /> Single
            </button>
            <button class:active={viewMode === 'compare'} onclick={() => switchViewMode('compare')} title="Multi-Model Comparison">
              <LayoutList size={16} /> Compare
            </button>
          </div>

          <div class="status-pill" class:online={apiOnline}>
            <span class="dot"></span>
            <span>{apiOnline ? 'Live Data Feed' : 'Offline'}</span>
          </div>
        </div>
      </div>

      <div class="hero-text">
        <h1>Islamabad Atmosphere</h1>
        <p class="subtitle">
          Next-generation environmental intelligence. Real-time geospatial tracking and high-fidelity multi-model AQI predictions.
        </p>
      </div>
    </header>

    <main class="main-grid">
      <!-- LEFT COLUMN -->
      <div class="left-col">
        <!-- 3D Map Simulation -->
        <section in:fly={{ y: 20, duration: 800, delay: 200 }} class="map-section card">
          <div class="map-header">
            <h2><CloudFog size={16}/> Geospatial Simulation</h2>
            {#if viewMode === 'single' && modelsInfo}
              <select bind:value={selectedModel} onchange={handleModelChange} class="model-select">
                {#each Object.keys(modelsInfo) as m}
                  <option value={m}>{m.replace('_', ' ').toUpperCase()}</option>
                {/each}
              </select>
            {/if}
          </div>
          
          <div class="map-container">
            <MapLibre
              style={MAP_STYLE}
              center={[LNG, LAT]}
              zoom={13.5}
              pitch={65}
              bearing={-20}
              class="map"
              attributionControl={false}
              onload={handleMapLoad}
            >
              <FillExtrusionLayer
                id="3d-buildings"
                source="carto"
                sourceLayer="building"
                minzoom={13}
                paint={{
                  'fill-extrusion-color': '#2C2B29',
                  'fill-extrusion-height': [
                    'coalesce', 
                    ['get', 'height'], 
                    ['get', 'render_height'], 
                    25 
                  ],
                  'fill-extrusion-base': [
                    'coalesce',
                    ['get', 'min_height'],
                    ['get', 'render_min_height'],
                    0
                  ],
                  'fill-extrusion-opacity': 0.8
                }}
                beforeLayerType="symbol"
              />

              {#if prediction}
                <Marker lngLat={[LNG, LAT]}>
                  <div class="map-marker pulse-marker" style="--aqi-color: {aqiColor(prediction.current_aqi)}">
                    <span class="marker-label">{Math.round(prediction.current_aqi || 0)}</span>
                  </div>
                </Marker>
              {/if}
            </MapLibre>
            
            <!-- Custom Canvas Overlay for Weather/Particles -->
            <canvas bind:this={canvasOverlay} class="simulation-canvas"></canvas>
          </div>
        </section>

        <!-- Secondary Environmental Metrics -->
        {#if viewMode === 'single' && prediction && !loading}
          <section in:fly={{ y: 20, duration: 800, delay: 300 }} class="metrics-grid">
            <div class="metric-box">
              <div class="metric-title"><Droplets size={14}/> PM2.5</div>
              <div class="metric-val">{prediction.pm2_5?.toFixed(1) || '--'} <span class="unit">μg/m³</span></div>
            </div>
            <div class="metric-box">
              <div class="metric-title"><Droplets size={14}/> PM10</div>
              <div class="metric-val">{prediction.pm10?.toFixed(1) || '--'} <span class="unit">μg/m³</span></div>
            </div>
            <div class="metric-box">
              <div class="metric-title"><CloudFog size={14}/> CO</div>
              <div class="metric-val">{prediction.co?.toFixed(1) || '--'} <span class="unit">μg/m³</span></div>
            </div>
            <div class="metric-box">
              <div class="metric-title"><CloudFog size={14}/> NO2</div>
              <div class="metric-val">{prediction.no2?.toFixed(1) || '--'} <span class="unit">μg/m³</span></div>
            </div>
            <div class="metric-box">
              <div class="metric-title"><CloudFog size={14}/> SO2</div>
              <div class="metric-val">{prediction.so2?.toFixed(1) || '--'} <span class="unit">μg/m³</span></div>
            </div>
            <div class="metric-box">
              <div class="metric-title"><CloudFog size={14}/> O3</div>
              <div class="metric-val">{prediction.o3?.toFixed(1) || '--'} <span class="unit">μg/m³</span></div>
            </div>
            <div class="metric-box">
              <div class="metric-title"><Thermometer size={14}/> Temp</div>
              <div class="metric-val">{prediction.temperature?.toFixed(1) || '--'} <span class="unit">°C</span></div>
            </div>
            <div class="metric-box">
              <div class="metric-title"><Wind size={14}/> Wind</div>
              <div class="metric-val">{prediction.wind_speed?.toFixed(1) || '--'} <span class="unit">m/s</span></div>
            </div>
          </section>
        {/if}
      </div>

      <!-- RIGHT COLUMN -->
      <div class="right-col">
        {#if loading}
          <div in:fade class="card loading-card">
            <div class="loading-spinner"></div>
            <p>Evaluating atmospheric models…</p>
          </div>
        {:else if error}
          <div in:fade class="card error-card">
            <AlertTriangle size={32} color="#f87171" />
            <p>{error}</p>
          </div>
        {:else if viewMode === 'single' && prediction}
          <!-- Radial Gauges -->
          <div class="gauges-container" in:fly={{ y: 16, duration: 700, delay: 300 }}>
            <AQIGauge aqi={prediction.current_aqi} label="NOW" title="Live Observation" />
            <AQIGauge aqi={prediction.predicted_aqi_24h} label="PREDICTED" title="24h Forecast" />
          </div>

          {#if prediction.health_advisory}
            <div class="advisory-box" style="border-left-color: {aqiColor(prediction.current_aqi)}">
              <Info size={20} color={aqiColor(prediction.current_aqi)} />
              <p>{prediction.health_advisory}</p>
            </div>
          {/if}

          <!-- Explainable AI (Drivers) -->
          {#if explanation && explanation.top_drivers}
            <section in:fly={{ y: 20, duration: 800, delay: 400 }} class="xai-section card">
              <div class="card-header">
                <h2><Activity size={18}/> Feature Importance</h2>
                <span class="header-tag">XAI</span>
              </div>
              <div class="drivers-list">
                {#each explanation.top_drivers.slice(0, 4) as driver}
                  {@const Icon = getFeatureIcon(driver.feature)}
                  <div class="driver-item">
                    <div class="driver-info">
                      <Icon size={14} color="var(--text-muted)" />
                      <span class="driver-name">{formatFeatureName(driver.feature)}</span>
                      <span class="driver-val">{driver.current_value.toFixed(1)}</span>
                    </div>
                    <div class="importance-bar-bg">
                      <div class="importance-bar-fill" style="width: {Math.min(driver.importance * 100, 100)}%; background: var(--accent);"></div>
                    </div>
                  </div>
                {/each}
              </div>
            </section>
          {/if}

          <!-- Model Accuracy -->
          {#if accuracy && accuracy.accuracy_score !== null}
            <section in:fly={{ y: 20, duration: 800, delay: 500 }} class="accuracy-section card">
              <div class="card-header">
                <h2><TrendingUp size={18}/> Model Performance</h2>
                <span class="header-tag">7-Day</span>
              </div>
              <div class="accuracy-content">
                <div class="acc-score">
                  <span class="acc-val">{accuracy.accuracy_score.toFixed(1)}%</span>
                  <span class="acc-label">Avg. Accuracy</span>
                </div>
                <div class="history-list">
                  {#each accuracy.history as h}
                  <div class="history-item">
                    <span class="date">{new Date(h.date).toLocaleDateString(undefined, {month: 'short', day: 'numeric'})}</span>
                    <span class="val pred" title="Predicted">P: {h.predicted.toFixed(0)}</span>
                    <span class="val act" title="Actual">A: {h.actual ? h.actual.toFixed(0) : '--'}</span>
                  </div>
                  {/each}
                </div>
              </div>
            </section>
          {/if}
          
        {:else if viewMode === 'compare'}
          <!-- Multi-Model Comparison -->
          <div class="compare-grid" in:fade>
            {#each Object.entries(modelsInfo) as [mName, mData]: any}
              {@const p = comparePredictions[mName]}
              <div class="compare-card card" class:highlight={mName === 'random_forest'}>
                <div class="compare-header">
                  <h3>{mName.replace('_', ' ').toUpperCase()}</h3>
                  {#if mName === 'random_forest'}<span class="badge">Ensemble</span>{/if}
                </div>
                
                <div class="compare-metrics">
                  <div class="c-metric">
                    <span>Validation R²</span>
                    <strong>{mData.r2 ? mData.r2.toFixed(3) : '--'}</strong>
                  </div>
                  <div class="c-metric">
                    <span>7-Day Accuracy</span>
                    <strong>{compareAccuracy[mName]?.accuracy_score ? `${compareAccuracy[mName].accuracy_score.toFixed(1)}%` : '--'}</strong>
                  </div>
                </div>

                <div class="compare-prediction">
                  <span>24h Forecast</span>
                  {#if p}
                    <div class="c-aqi" style="color: {aqiColor(p.predicted_aqi_24h)}">
                      {p.predicted_aqi_24h.toFixed(1)}
                    </div>
                  {:else}
                    <div class="c-aqi">--</div>
                  {/if}
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </main>

    <footer in:fade={{ duration: 600, delay: 800 }} class="observatory-footer">
      <p>Urban Intel GeoAI Pipeline · High-Fidelity Svelte 5 & FastAPI Framework</p>
    </footer>
  </div>
{/if}

<style>
  :global(body) {
    background-color: var(--bg-base, #111315);
    color: var(--text-primary, #E6E4D9);
    margin: 0;
    font-family: 'Inter', system-ui, sans-serif;
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

  .header-controls {
    display: flex;
    align-items: center;
    gap: 1.5rem;
  }

  .view-toggles {
    display: flex;
    background: rgba(0,0,0,0.3);
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    overflow: hidden;
  }

  .view-toggles button {
    background: transparent;
    border: none;
    color: var(--text-muted);
    padding: 0.5rem 1rem;
    font-size: 0.8rem;
    font-family: var(--font-sans);
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    transition: all 0.3s;
  }

  .view-toggles button.active {
    background: var(--border-strong);
    color: var(--text-primary);
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
    max-width: 800px;
  }

  .main-grid {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1.5rem 2rem 3rem;
    display: grid;
    grid-template-columns: 1fr 400px;
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

  .card {
    background: rgba(28, 32, 36, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-subtle);
    padding: 1.5rem;
    border-radius: 8px;
  }

  /* Map Container */
  .map-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
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

  .model-select {
    background: var(--bg-base);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    padding: 0.3rem 0.5rem;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    outline: none;
  }

  .map-container {
    height: 480px;
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid var(--border-subtle);
    position: relative;
  }

  :global(.map) { width: 100%; height: 100%; }

  .simulation-canvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none; /* Let map events pass through */
    z-index: 2;
  }

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

  /* Secondary Metrics Grid */
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
  }

  @media (max-width: 600px) {
    .metrics-grid { grid-template-columns: repeat(2, 1fr); }
  }

  .metric-box {
    background: rgba(28, 32, 36, 0.4);
    border: 1px solid var(--border-subtle);
    padding: 1rem;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
  }

  .metric-title {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
  }

  .metric-val {
    font-family: var(--font-mono);
    font-size: 1.2rem;
    color: var(--text-primary);
  }

  .metric-val .unit {
    font-size: 0.7rem;
    color: var(--text-muted);
  }

  /* Gauges & XAI */
  .gauges-container {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .advisory-box {
    display: flex;
    gap: 1rem;
    padding: 1rem;
    background: rgba(0,0,0,0.2);
    border-left: 3px solid;
    border-radius: 4px;
    margin-top: 1.5rem;
  }

  .advisory-box p {
    margin: 0;
    font-size: 0.9rem;
    line-height: 1.5;
    color: var(--text-primary);
  }

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
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

  /* Accuracy Section */
  .accuracy-content {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .acc-score {
    display: flex;
    flex-direction: column;
    align-items: center;
    background: rgba(0,0,0,0.2);
    padding: 1rem;
    border-radius: 6px;
    border: 1px solid var(--border-subtle);
  }

  .acc-val {
    font-family: var(--font-serif);
    font-size: 2.5rem;
    color: #4ade80;
    line-height: 1;
  }

  .acc-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
  }

  .history-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .history-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    padding: 0.4rem 0.5rem;
    background: rgba(255,255,255,0.03);
    border-radius: 4px;
  }

  .history-item .date {
    color: var(--text-muted);
    flex: 1;
  }

  .history-item .val {
    width: 60px;
    text-align: right;
  }

  .history-item .pred {
    color: var(--text-primary);
  }

  .history-item .act {
    color: var(--accent);
  }

  /* Compare Mode */
  .compare-grid {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .compare-card {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding: 1.25rem;
  }

  .compare-card.highlight {
    border-color: var(--accent);
    background: rgba(184, 169, 154, 0.05);
  }

  .compare-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 0.5rem;
  }

  .compare-header h3 {
    margin: 0;
    font-size: 0.9rem;
    font-family: var(--font-mono);
    letter-spacing: 0.05em;
  }

  .badge {
    font-size: 0.65rem;
    background: var(--accent);
    color: var(--bg-base);
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    font-weight: 700;
  }

  .compare-metrics {
    display: flex;
    gap: 2rem;
  }

  .c-metric {
    display: flex;
    flex-direction: column;
  }

  .c-metric span {
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
  }

  .c-metric strong {
    font-family: var(--font-mono);
    font-size: 0.9rem;
    color: var(--text-secondary);
  }

  .compare-prediction {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 0.5rem;
    background: rgba(0,0,0,0.2);
    padding: 0.75rem;
    border-radius: 6px;
  }

  .compare-prediction span {
    font-size: 0.8rem;
    color: var(--text-primary);
  }

  .c-aqi {
    font-family: var(--font-serif);
    font-size: 1.8rem;
    font-weight: 600;
    line-height: 1;
  }

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
    margin: 0 auto 1rem;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

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
