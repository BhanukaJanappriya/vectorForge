import {
  useEffect,
  useId,
  useRef,
  useState,
  type Dispatch,
  type KeyboardEvent,
  type PointerEvent,
  type ReactNode,
  type SetStateAction,
} from 'react';
import { IDENTITY, MAX_SCALE, MIN_SCALE, panBy, toTransform, zoomAt, type View } from '../lib/viewport';

type Mode = 'slider' | 'side';

interface ComparisonViewerProps {
  originalUrl: string;
  /** URL of output.svg. Rendered with <img> only: the SVG markup is never inlined into the page. */
  vectorUrl: string;
  width: number;
  height: number;
}

const ZOOM_STEP = 1.25;
const PAN_STEP = 0.1;

interface ZoomPaneProps {
  view: View;
  setView: Dispatch<SetStateAction<View>>;
  label: string;
  /** Width / height of the image. */
  aspect: number;
  children: ReactNode;
  testId: string;
}

/** A focusable viewport: wheel/drag/keyboard update the shared view. */
function ZoomPane({ view, setView, label, aspect, children, testId }: ZoomPaneProps) {
  const ref = useRef<HTMLDivElement>(null);
  const drag = useRef<{ id: number; x: number; y: number } | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = el.getBoundingClientRect();
      const px = (event.clientX - rect.left) / rect.width;
      const py = (event.clientY - rect.top) / rect.height;
      const factor = event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
      setView((v) => zoomAt(v, factor, px, py));
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [setView]);

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    drag.current = { id: event.pointerId, x: event.clientX, y: event.clientY };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };
  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d || d.id !== event.pointerId) return;
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const dx = (event.clientX - d.x) / rect.width;
    const dy = (event.clientY - d.y) / rect.height;
    drag.current = { ...d, x: event.clientX, y: event.clientY };
    setView((v) => panBy(v, dx, dy));
  };
  const endDrag = () => {
    drag.current = null;
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const actions: Record<string, (v: View) => View> = {
      ArrowLeft: (v) => panBy(v, PAN_STEP, 0),
      ArrowRight: (v) => panBy(v, -PAN_STEP, 0),
      ArrowUp: (v) => panBy(v, 0, PAN_STEP),
      ArrowDown: (v) => panBy(v, 0, -PAN_STEP),
      '+': (v) => zoomAt(v, ZOOM_STEP),
      '=': (v) => zoomAt(v, ZOOM_STEP),
      '-': (v) => zoomAt(v, 1 / ZOOM_STEP),
      '0': () => IDENTITY,
    };
    const action = actions[event.key];
    if (!action) return;
    event.preventDefault();
    setView(action);
  };

  /* A 2-D pan/zoom surface has no native element. It is focusable and fully keyboard operable
     (arrows pan, +/- zoom, 0 resets); the same actions are also available as buttons. */
  /* eslint-disable jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex */
  return (
    <div
      ref={ref}
      role="application"
      aria-roledescription="zoomable image"
      aria-label={`${label}. Arrow keys pan, plus and minus zoom, 0 resets.`}
      tabIndex={0}
      data-testid={testId}
      onKeyDown={onKeyDown}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      className={`checkerboard relative mx-auto overflow-hidden rounded-lg border border-slate-300 focus-visible:outline-3 focus-visible:outline-offset-2 focus-visible:outline-sky-600 ${
        view.scale > 1 ? 'cursor-grab touch-none active:cursor-grabbing' : ''
      }`}
      // Width is capped so the pane keeps the image's aspect ratio within 70vh of height;
      // the slider percentage then maps exactly onto image columns.
      style={{ aspectRatio: String(aspect), width: `min(100%, calc(70vh * ${aspect}))` }}
    >
      {children}
    </div>
  );
  /* eslint-enable jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex */
}

function Layer({ view, src, alt, pixelated }: { view: View; src: string; alt: string; pixelated?: boolean }) {
  const [failed, setFailed] = useState(false);
  return (
    <div className="absolute inset-0 origin-top-left" style={{ transform: toTransform(view) }}>
      {failed ? (
        <p className="flex h-full items-center justify-center p-4 text-sm text-slate-700">{alt} could not be loaded.</p>
      ) : (
        <img
          src={src}
          alt={alt}
          draggable={false}
          onError={() => setFailed(true)}
          className="pointer-events-none absolute inset-0 h-full w-full object-contain select-none"
          style={pixelated && view.scale > 1 ? { imageRendering: 'pixelated' } : undefined}
        />
      )}
    </div>
  );
}

function Tag({ children, side }: { children: ReactNode; side: 'left' | 'right' }) {
  return (
    <span
      className={`pointer-events-none absolute top-2 ${side === 'left' ? 'left-2' : 'right-2'} rounded bg-slate-900/80 px-2 py-0.5 text-xs font-medium text-white`}
      aria-hidden="true"
    >
      {children}
    </span>
  );
}

/** Original vs. vector: slider overlay or side-by-side, with one shared zoom/pan state. */
export function ComparisonViewer({ originalUrl, vectorUrl, width, height }: ComparisonViewerProps) {
  const id = useId();
  const [mode, setMode] = useState<Mode>('slider');
  const [split, setSplit] = useState(50);
  const [view, setView] = useState<View>(IDENTITY);
  const aspect = Math.max(1, width) / Math.max(1, height);

  return (
    <section aria-labelledby={`${id}-heading`} className="panel" data-testid="comparison">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 id={`${id}-heading`} className="panel-title mb-0">
          Original vs. vector
        </h2>
        <div className="segmented" role="group" aria-label="Comparison layout">
          {(['slider', 'side'] as const).map((m) => (
            <button
              key={m}
              type="button"
              aria-pressed={mode === m}
              className="segmented-button"
              onClick={() => setMode(m)}
            >
              {m === 'slider' ? 'Slider' : 'Side by side'}
            </button>
          ))}
        </div>
      </div>

      {mode === 'slider' ? (
        <>
          <ZoomPane view={view} setView={setView} label="Original and vector overlay" aspect={aspect} testId="pane-overlay">
            <Layer view={view} src={originalUrl} alt="Original image" pixelated />
            <div className="absolute inset-0" style={{ clipPath: `inset(0 0 0 ${split}%)` }} data-testid="vector-clip">
              <Layer view={view} src={vectorUrl} alt="Vector result (SVG)" />
            </div>
            <div
              className="pointer-events-none absolute inset-y-0 w-0.5 -translate-x-1/2 bg-sky-500 shadow-[0_0_0_1px_rgba(255,255,255,0.8)]"
              style={{ left: `${split}%` }}
              aria-hidden="true"
            />
            <Tag side="left">Original</Tag>
            <Tag side="right">Vector</Tag>
          </ZoomPane>
          <label htmlFor={`${id}-split`} className="mt-3 block text-sm text-slate-700">
            Divider position (original on the left, vector on the right)
          </label>
          <input
            id={`${id}-split`}
            type="range"
            min={0}
            max={100}
            step={1}
            value={split}
            aria-valuetext={`${split}% original`}
            onChange={(e) => setSplit(Number(e.currentTarget.value))}
            className="w-full accent-sky-700"
          />
        </>
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <figure className="min-w-0">
            <ZoomPane view={view} setView={setView} label="Original image" aspect={aspect} testId="pane-original">
              <Layer view={view} src={originalUrl} alt="Original image" pixelated />
            </ZoomPane>
            <figcaption className="mt-1 text-center text-sm text-slate-700">Original</figcaption>
          </figure>
          <figure className="min-w-0">
            <ZoomPane view={view} setView={setView} label="Vector result" aspect={aspect} testId="pane-vector">
              <Layer view={view} src={vectorUrl} alt="Vector result (SVG)" />
            </ZoomPane>
            <figcaption className="mt-1 text-center text-sm text-slate-700">Vector (SVG)</figcaption>
          </figure>
        </div>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2" role="group" aria-label="Zoom controls">
        <button
          type="button"
          className="btn btn-secondary"
          onClick={() => setView((v) => zoomAt(v, 1 / ZOOM_STEP))}
          disabled={view.scale <= MIN_SCALE}
          aria-label="Zoom out"
        >
          −
        </button>
        <span className="min-w-14 text-center text-sm tabular-nums" aria-live="polite" data-testid="zoom-level">
          {Math.round(view.scale * 100)}%
        </span>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={() => setView((v) => zoomAt(v, ZOOM_STEP))}
          disabled={view.scale >= MAX_SCALE}
          aria-label="Zoom in"
        >
          +
        </button>
        <button type="button" className="btn btn-secondary" onClick={() => setView(IDENTITY)} disabled={view.scale === 1 && view.x === 0 && view.y === 0}>
          Reset view
        </button>
        <p className="w-full text-xs text-slate-600 sm:w-auto">Scroll or use the buttons to zoom; drag to pan. Both panes stay in sync.</p>
      </div>
    </section>
  );
}
