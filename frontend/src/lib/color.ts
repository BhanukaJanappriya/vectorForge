/** Palette helpers. The API requires palette colors as lowercase `#rrggbb`. */

const HEX_RE = /^#[0-9a-f]{6}$/;

/** Normalizes user input (`#ABC`, `abc123`, ` #AbC123 `) to `#rrggbb`, or returns null if invalid. */
export function normalizeHex(input: string): string | null {
  let value = input.trim().toLowerCase();
  if (!value.startsWith('#')) value = `#${value}`;
  if (/^#[0-9a-f]{3}$/.test(value)) {
    value = `#${value.slice(1).split('').map((c) => c + c).join('')}`;
  }
  return HEX_RE.test(value) ? value : null;
}

export function isValidHex(value: string): boolean {
  return HEX_RE.test(value);
}

/** De-duplicates while preserving order (mirrors the server's palette_override validator). */
export function dedupePalette(colors: readonly string[]): string[] {
  return Array.from(new Set(colors));
}

/** Relative luminance (WCAG) used to pick readable text over a swatch. */
export function luminance(hex: string): number {
  const channel = (i: number) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(1) + 0.7152 * channel(3) + 0.0722 * channel(5);
}

/** Upper bound of `Settings.palette_override` in the spec (maxItems). */
export const MAX_PALETTE = 64;

/** Builds the `palette_override` list from the edited entries (lowercase #rrggbb, de-duplicated). */
export function buildPaletteOverride(entries: readonly { hex: string }[]): string[] {
  return dedupePalette(entries.map((e) => e.hex)).slice(0, MAX_PALETTE);
}
