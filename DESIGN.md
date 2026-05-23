# Design System

## Theme
Dark Mode (default and only theme, optimized for the "Dark Academia" aesthetic).

## Color Strategy
**Restrained**: Deep, tinted neutral backgrounds with a single, highly deliberate saturated accent color (Muted Bronze/Gold) carrying the visual weight. No pure black `#000` or pure white `#fff` used anywhere.

## Palette
- **Background Base:** Deep Slate `#111315` (or OKLCH equivalent)
- **Background Elevated:** Charcoal `#1C2024` for cards and panels
- **Background Highlight:** `#252A2E` for subtle hover states
- **Accent Primary:** Muted Bronze/Antique Gold `#B8A99A`
- **Text Primary:** Soft Off-White `#E6E4D9`
- **Text Secondary:** Muted Gray-Brown `#A39F98`
- **Borders/Dividers:** `#2D3236`

## Typography
- **Headings (Display, H1-H3):** *Playfair Display* (Elegant, scholarly serif)
- **Body & Data UI:** *Inter* or *Outfit* (Clean, legible sans-serif for numbers and UI elements)
- **Hierarchy:** High contrast between headings (larger, serif) and data points (structured, sans-serif). Max width for reading blocks: 65-75ch.

## Layout & Components
- **Spacing:** Generous, rhythmic padding. Avoid completely flat spacing.
- **Cards:** Used sparingly, only to group distinct data points (e.g., Current vs Predicted AQI). No nested cards. Borders should be subtle.
- **Status Indicators:** Elegant dots or text, without side-stripe colored borders or harsh alert colors unless absolutely necessary (if alerts are needed, mute their chroma to fit the academia tone).

## Motion & Micro-interactions
- **Easing:** Exponential ease-outs (e.g., `cubic-bezier(0.16, 1, 0.3, 1)`).
- **Transitions:** Smooth fades and subtle vertical drifts for data appearance. No bounce or elastic effects.
- **Hover:** Very subtle background lightening and shadow depth adjustments; no layout shifts.
- **Exclusion:** No CSS layout property animations.
