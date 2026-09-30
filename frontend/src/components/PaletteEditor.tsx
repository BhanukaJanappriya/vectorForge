import { useId, useMemo, useState } from 'react';
import { MAX_PALETTE, buildPaletteOverride, luminance, normalizeHex } from '../lib/color';


interface Entry {
  key: number;
  hex: string;
  draft: string;
}

interface PaletteEditorProps {
  /** `result.palette_hex` of the current job. */
  palette: readonly string[];
  /** True if the current job was produced with a palette_override. */
  overrideActive: boolean;
  busy?: boolean;
  /** Re-run with `settings.palette_override = palette`. */
  onRerun: (palette: string[]) => void;
  /** Re-run with `settings.palette_override = null` (automatic palette). */
  onResetAuto: () => void;
}

function toEntries(palette: readonly string[]): Entry[] {
  return palette.map((hex, i) => ({ key: i, hex, draft: hex }));
}

/** Shows the result palette; colors can be edited, merged or removed, then re-run. */
export function PaletteEditor({ palette, overrideActive, busy = false, onRerun, onResetAuto }: PaletteEditorProps) {
  const id = useId();
  const [entries, setEntries] = useState<Entry[]>(() => toEntries(palette));
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [nextKey, setNextKey] = useState(palette.length);
  const [notice, setNotice] = useState('');

  const override = useMemo(() => buildPaletteOverride(entries), [entries]);
  const invalid = entries.some((e) => normalizeHex(e.draft) === null);
  const changed = override.join() !== palette.join();
  const canRerun = changed && !invalid && !busy && override.length >= 1;

  const update = (key: number, patch: Partial<Entry>) =>
    setEntries((list) => list.map((e) => (e.key === key ? { ...e, ...patch } : e)));

  const setDraft = (key: number, draft: string) => {
    const hex = normalizeHex(draft);
    update(key, hex ? { draft, hex } : { draft });
  };

  const commitDraft = (key: number) => {
    setEntries((list) =>
      list.map((e) => {
        if (e.key !== key) return e;
        const hex = normalizeHex(e.draft);
        return hex ? { ...e, hex, draft: hex } : e;
      }),
    );
  };

  const toggle = (key: number, on: boolean) =>
    setSelected((s) => {
      const next = new Set(s);
      if (on) next.add(key);
      else next.delete(key);
      return next;
    });

  const merge = () => {
    const chosen = entries.filter((e) => selected.has(e.key));
    const target = chosen[0];
    if (!target || chosen.length < 2) return;
    // Merged colors collapse into the first selected color; the others are removed.
    setEntries((list) => list.filter((e) => e.key === target.key || !selected.has(e.key)));
    setSelected(new Set());
    setNotice(`Merged ${chosen.length} colors into ${target.hex}.`);
  };

  const remove = (key: number) => {
    const entry = entries.find((e) => e.key === key);
    setEntries((list) => list.filter((e) => e.key !== key));
    toggle(key, false);
    if (entry) setNotice(`Removed ${entry.hex}; its pixels will use the nearest remaining color.`);
  };

  const addColor = () => {
    setEntries((list) => [...list, { key: nextKey, hex: '#000000', draft: '#000000' }]);
    setNextKey((k) => k + 1);
    setNotice('Added a color.');
  };

  const undo = () => {
    setEntries(toEntries(palette));
    setSelected(new Set());
    setNextKey(palette.length);
    setNotice('Palette changes discarded.');
  };

  return (
    <section aria-labelledby={`${id}-heading`} className="panel" data-testid="palette">
      <h2 id={`${id}-heading`} className="panel-title">
        Palette ({entries.length} {entries.length === 1 ? 'color' : 'colors'})
      </h2>
      <p className="mb-3 text-sm text-slate-600">
        Edit a color, or select two or more and merge them, then re-run the conversion with your palette.
      </p>
      <ul className="space-y-2">
        {entries.map((entry, index) => {
          const n = index + 1;
          const entryInvalid = normalizeHex(entry.draft) === null;
          return (
            <li key={entry.key} className="flex flex-wrap items-center gap-2 rounded-md border border-slate-200 bg-white p-2" data-testid="palette-entry">
              <input
                type="checkbox"
                className="h-4 w-4 accent-sky-700"
                aria-label={`Select color ${n} (${entry.hex}) for merging`}
                checked={selected.has(entry.key)}
                onChange={(e) => toggle(entry.key, e.currentTarget.checked)}
                disabled={busy}
              />
              <input
                type="color"
                className="h-9 w-11 cursor-pointer rounded border border-slate-300 bg-white p-0.5"
                aria-label={`Color ${n} picker`}
                value={entry.hex}
                disabled={busy}
                onChange={(e) => update(entry.key, { hex: e.currentTarget.value.toLowerCase(), draft: e.currentTarget.value.toLowerCase() })}
              />
              <div className="min-w-0 flex-1">
                <label htmlFor={`${id}-hex-${entry.key}`} className="sr-only">
                  Color {n} hex value
                </label>
                <input
                  id={`${id}-hex-${entry.key}`}
                  type="text"
                  inputMode="text"
                  spellCheck={false}
                  autoComplete="off"
                  className="field-input w-full font-mono text-sm"
                  value={entry.draft}
                  disabled={busy}
                  aria-invalid={entryInvalid}
                  aria-describedby={entryInvalid ? `${id}-err-${entry.key}` : undefined}
                  onChange={(e) => setDraft(entry.key, e.currentTarget.value)}
                  onBlur={() => commitDraft(entry.key)}
                />
                {entryInvalid && (
                  <p id={`${id}-err-${entry.key}`} className="mt-1 text-xs text-red-700">
                    Use a hex color like #1a2b3c.
                  </p>
                )}
              </div>
              <span
                className="hidden rounded px-2 py-1 font-mono text-xs sm:inline-block"
                style={{ backgroundColor: entry.hex, color: luminance(entry.hex) > 0.4 ? '#0f172a' : '#ffffff' }}
                aria-hidden="true"
              >
                Aa
              </span>
              <button
                type="button"
                className="btn btn-secondary px-2"
                onClick={() => remove(entry.key)}
                disabled={busy || entries.length <= 1}
                aria-label={`Remove color ${n} (${entry.hex})`}
              >
                Remove
              </button>
            </li>
          );
        })}
      </ul>
      <p className="sr-only" aria-live="polite">
        {notice}
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        <button type="button" className="btn btn-secondary" onClick={merge} disabled={busy || selected.size < 2}>
          Merge selected{selected.size >= 2 ? ` (${selected.size})` : ''}
        </button>
        <button type="button" className="btn btn-secondary" onClick={addColor} disabled={busy || entries.length >= MAX_PALETTE}>
          Add color
        </button>
        <button type="button" className="btn btn-secondary" onClick={undo} disabled={busy || (!changed && !invalid)}>
          Undo changes
        </button>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        <button type="button" className="btn btn-primary" onClick={() => onRerun(override)} disabled={!canRerun}>
          Re-run with edited palette
        </button>
        {overrideActive && (
          <button type="button" className="btn btn-secondary" onClick={onResetAuto} disabled={busy}>
            Use automatic palette
          </button>
        )}
      </div>
      {invalid && <p className="mt-2 text-xs text-red-700">Fix the invalid hex values before re-running.</p>}
      {override.length < entries.length && (
        <p className="mt-2 text-xs text-slate-600">Duplicate colors are combined into one layer.</p>
      )}
    </section>
  );
}
