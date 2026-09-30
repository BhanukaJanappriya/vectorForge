/**
 * Pan/zoom math shared by all comparison panes (so they stay synchronized).
 * Offsets are fractions of the viewport size, which makes the same view valid for
 * panes of different pixel sizes.
 */
export interface View {
  scale: number;
  /** Translation as a fraction of the viewport width (<= 0). */
  x: number;
  /** Translation as a fraction of the viewport height (<= 0). */
  y: number;
}

export const MIN_SCALE = 1;
export const MAX_SCALE = 16;
export const IDENTITY: View = { scale: 1, x: 0, y: 0 };

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/** Keeps the content covering the viewport (no empty margins when zoomed in). */
export function clampView(view: View): View {
  const scale = clamp(view.scale, MIN_SCALE, MAX_SCALE);
  const min = 1 - scale;
  return { scale, x: clamp(view.x, min, 0), y: clamp(view.y, min, 0) };
}

/** Zooms by `factor`, keeping the point (px, py) (fractions of the viewport) fixed on screen. */
export function zoomAt(view: View, factor: number, px = 0.5, py = 0.5): View {
  const scale = clamp(view.scale * factor, MIN_SCALE, MAX_SCALE);
  const ratio = scale / view.scale;
  return clampView({ scale, x: px - (px - view.x) * ratio, y: py - (py - view.y) * ratio });
}

/** Pans by (dx, dy) fractions of the viewport. */
export function panBy(view: View, dx: number, dy: number): View {
  return clampView({ ...view, x: view.x + dx, y: view.y + dy });
}

/** CSS transform for the content layer (transform-origin must be 0 0). */
export function toTransform(view: View): string {
  return `translate(${view.x * 100}%, ${view.y * 100}%) scale(${view.scale})`;
}
