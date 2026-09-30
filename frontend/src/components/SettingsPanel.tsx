import { useId } from 'react';
import { DETAIL_LEVELS, LINE_MODES, OUTPUT_FORMATS, PROCESSING_MODES } from '../api/enums';
import type { OutputFormat, Settings } from '../api/types';
import { MODE_LABELS } from '../lib/format';

/** Color count used when the user switches "auto" off without having chosen a value. */
export const DEFAULT_MANUAL_COLORS = 8;
export const MIN_COLORS = 2;
export const MAX_COLORS = 64;

const DETAIL_LABELS = { low: 'Low', medium: 'Medium', high: 'High' } as const;
const LINE_MODE_LABELS = { outline: 'Outline', centerline: 'Centerline' } as const;
const FORMAT_LABELS: Record<OutputFormat, string> = {
  svg: 'SVG (always)',
  ai: 'AI (Illustrator)',
  eps: 'EPS',
  png: 'PNG preview',
};

interface SettingsPanelProps {
  value: Settings;
  onChange: (next: Settings) => void;
  disabled?: boolean;
  /** Number of colors in the active palette override, if the current job uses one. */
  paletteOverrideCount?: number | null;
}

/** Form for every user-controllable `Settings` field except palette_override (edited in the palette panel). */
export function SettingsPanel({ value, onChange, disabled = false, paletteOverrideCount = null }: SettingsPanelProps) {
  const id = useId();
  const set = <K extends keyof Settings>(key: K, v: Settings[K]) => onChange({ ...value, [key]: v });
  const autoColors = value.max_colors === null;
  const formats = new Set<OutputFormat>(value.output_formats ?? ['svg']);

  const toggleFormat = (format: OutputFormat, on: boolean) => {
    const next = new Set(formats);
    if (on) next.add(format);
    else next.delete(format);
    next.add('svg');
    set(
      'output_formats',
      OUTPUT_FORMATS.filter((f) => next.has(f)),
    );
  };

  return (
    <section aria-labelledby={`${id}-heading`} className="panel">
      <h2 id={`${id}-heading`} className="panel-title">
        2. Settings
      </h2>
      <fieldset disabled={disabled} className="space-y-5">
        <legend className="sr-only">Conversion settings</legend>

        <div>
          <label htmlFor={`${id}-mode`} className="field-label">
            Mode
          </label>
          <select
            id={`${id}-mode`}
            className="field-input"
            value={value.mode}
            onChange={(e) => set('mode', e.currentTarget.value as Settings['mode'])}
          >
            {PROCESSING_MODES.map((m) => (
              <option key={m} value={m}>
                {MODE_LABELS[m]}
              </option>
            ))}
          </select>
        </div>

        <fieldset>
          <legend className="field-label">Colors</legend>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              className="h-4 w-4 accent-sky-700"
              checked={autoColors}
              onChange={(e) => set('max_colors', e.currentTarget.checked ? null : DEFAULT_MANUAL_COLORS)}
            />
            Auto-detect number of colors
          </label>
          <div className="mt-2">
            <label htmlFor={`${id}-colors`} className="flex justify-between text-sm text-slate-700">
              <span>Max colors</span>
              <output htmlFor={`${id}-colors`} className="font-medium tabular-nums">
                {autoColors ? 'Auto' : value.max_colors}
              </output>
            </label>
            <input
              id={`${id}-colors`}
              type="range"
              className="w-full accent-sky-700"
              min={MIN_COLORS}
              max={MAX_COLORS}
              step={1}
              disabled={autoColors || disabled}
              value={value.max_colors ?? DEFAULT_MANUAL_COLORS}
              onChange={(e) => set('max_colors', Number(e.currentTarget.value))}
            />
          </div>
          {paletteOverrideCount !== null && (
            <p className="mt-1 text-xs text-amber-800">
              A custom palette ({paletteOverrideCount} colors) is active, so max colors is ignored on re-run.
            </p>
          )}
        </fieldset>

        <fieldset>
          <legend className="field-label">Detail level</legend>
          <div className="segmented" role="presentation">
            {DETAIL_LEVELS.map((d) => (
              <label key={d} className="segmented-item">
                <input
                  type="radio"
                  name={`${id}-detail`}
                  value={d}
                  className="sr-only"
                  checked={value.detail_level === d}
                  onChange={() => set('detail_level', d)}
                />
                <span>{DETAIL_LABELS[d]}</span>
              </label>
            ))}
          </div>
        </fieldset>

        <fieldset>
          <legend className="field-label">Line mode</legend>
          <div className="segmented" role="presentation">
            {LINE_MODES.map((m) => (
              <label key={m} className="segmented-item">
                <input
                  type="radio"
                  name={`${id}-line`}
                  value={m}
                  className="sr-only"
                  checked={value.line_mode === m}
                  onChange={() => set('line_mode', m)}
                />
                <span>{LINE_MODE_LABELS[m]}</span>
              </label>
            ))}
          </div>
        </fieldset>

        <div>
          <label htmlFor={`${id}-smoothing`} className="flex justify-between text-sm font-medium text-slate-800">
            <span>Smoothing</span>
            <output htmlFor={`${id}-smoothing`} className="tabular-nums">
              {value.smoothing}
            </output>
          </label>
          <input
            id={`${id}-smoothing`}
            type="range"
            className="w-full accent-sky-700"
            min={0}
            max={100}
            step={1}
            value={value.smoothing}
            aria-describedby={`${id}-smoothing-hint`}
            onChange={(e) => set('smoothing', Number(e.currentTarget.value))}
          />
          <p id={`${id}-smoothing-hint`} className="flex justify-between text-xs text-slate-600">
            <span>Polygonal</span>
            <span>Smooth curves</span>
          </p>
        </div>

        <label className="flex items-start gap-2 text-sm">
          <input
            type="checkbox"
            className="mt-0.5 h-4 w-4 accent-sky-700"
            checked={value.remove_background}
            onChange={(e) => set('remove_background', e.currentTarget.checked)}
          />
          <span>
            Remove background
            <span className="block text-xs text-slate-600">Makes the detected background transparent.</span>
          </span>
        </label>

        <fieldset>
          <legend className="field-label">Output formats</legend>
          <div className="grid grid-cols-2 gap-2">
            {OUTPUT_FORMATS.map((f) => (
              <label key={f} className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="h-4 w-4 accent-sky-700"
                  checked={f === 'svg' || formats.has(f)}
                  disabled={f === 'svg' || disabled}
                  onChange={(e) => toggleFormat(f, e.currentTarget.checked)}
                />
                {FORMAT_LABELS[f]}
              </label>
            ))}
          </div>
        </fieldset>
      </fieldset>
    </section>
  );
}
