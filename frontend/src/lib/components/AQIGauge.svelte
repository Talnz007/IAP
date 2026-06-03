<script lang="ts">
  import { spring } from 'svelte/motion';
  import { onMount } from 'svelte';

  let {
    aqi = null,
    label = "NOW",
    title = "Current AQI"
  }: {
    aqi?: number | null;
    label?: string;
    title?: string;
  } = $props();

  const MAX_AQI = 500;
  
  // Spring to animate the value
  const animatedAqi = spring(0, { stiffness: 0.05, damping: 0.5 });
  
  $effect(() => {
    if (aqi !== null) {
      animatedAqi.set(aqi);
    }
  });

  function getAqiColor(val: number | null): string {
    if (val === null) return '#7A7670';
    if (val <= 50)  return '#4ade80'; // Good
    if (val <= 100) return '#facc15'; // Moderate
    if (val <= 150) return '#fb923c'; // Unhealthy for Sensitive
    if (val <= 200) return '#f87171'; // Unhealthy
    if (val <= 300) return '#c084fc'; // Very Unhealthy
    return '#881337'; // Hazardous
  }

  function getAqiCategory(val: number | null): string {
    if (val === null) return 'Unknown';
    if (val <= 50)  return 'Good';
    if (val <= 100) return 'Moderate';
    if (val <= 150) return 'Unhealthy for Sensitive Groups';
    if (val <= 200) return 'Unhealthy';
    if (val <= 300) return 'Very Unhealthy';
    return 'Hazardous';
  }

  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  // 270 degree arc = 0.75 of circumference
  const arcLength = 0.75 * circumference;
  const gapLength = circumference - arcLength;

  // The dasharray consists of the visible arc, then the gap
  const trackDashArray = `${arcLength} ${gapLength}`;
  
  let offset = $derived(arcLength - (Math.min($animatedAqi, MAX_AQI) / MAX_AQI) * arcLength);
  let currentColor = $derived(getAqiColor($animatedAqi));
  let category = $derived(getAqiCategory(aqi));
</script>

<div class="gauge-container">
  <div class="gauge-header">
    <span class="gauge-title">{title}</span>
    <span class="gauge-label">{label}</span>
  </div>

  <div class="gauge-wrapper">
    <svg viewBox="0 0 100 100" class="gauge-svg">
      <!-- Defs for gradients or filters can go here -->
      <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="3" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>

      <!-- Track Background -->
      <circle
        cx="50"
        cy="50"
        r="{radius}"
        fill="none"
        stroke="var(--border-subtle)"
        stroke-width="8"
        stroke-linecap="round"
        stroke-dasharray="{trackDashArray}"
        stroke-dashoffset="{gapLength / 2}"
        transform="rotate(135 50 50)"
      />

      <!-- Active Track -->
      <circle
        cx="50"
        cy="50"
        r="{radius}"
        fill="none"
        stroke="{currentColor}"
        stroke-width="8"
        stroke-linecap="round"
        stroke-dasharray="{trackDashArray}"
        stroke-dashoffset="{offset + gapLength / 2}"
        transform="rotate(135 50 50)"
        filter="url(#glow)"
        style="transition: stroke 0.4s ease;"
      />
    </svg>

    <div class="gauge-center">
      <div class="gauge-value" style="color: {currentColor}">
        {aqi !== null ? Math.round($animatedAqi) : '--'}
      </div>
      <div class="gauge-unit">AQI</div>
    </div>
  </div>

  <div class="gauge-footer" style="color: {currentColor}; background: {currentColor}15; border-color: {currentColor}40">
    {category}
  </div>
</div>

<style>
  .gauge-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
    padding: 1.5rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
  }

  .gauge-header {
    display: flex;
    justify-content: space-between;
    width: 100%;
    margin-bottom: 1rem;
    align-items: center;
  }

  .gauge-title {
    font-family: var(--font-serif);
    font-size: 1.1rem;
    color: var(--text-primary);
  }

  .gauge-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--accent);
    border: 1px solid var(--border-subtle);
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    letter-spacing: 0.05em;
  }

  .gauge-wrapper {
    position: relative;
    width: 160px;
    height: 160px;
    margin-bottom: 0.5rem;
  }

  .gauge-svg {
    width: 100%;
    height: 100%;
    transform-origin: center;
  }

  .gauge-center {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin-top: 5px;
  }

  .gauge-value {
    font-family: var(--font-serif);
    font-size: 3.2rem;
    font-weight: 500;
    line-height: 1;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    transition: color 0.4s ease;
  }

  .gauge-unit {
    font-family: var(--font-sans);
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    margin-top: -2px;
  }

  .gauge-footer {
    margin-top: auto;
    font-family: var(--font-sans);
    font-size: 0.85rem;
    font-weight: 600;
    text-align: center;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    border: 1px solid transparent;
    transition: all 0.4s ease;
    width: 100%;
  }
</style>
