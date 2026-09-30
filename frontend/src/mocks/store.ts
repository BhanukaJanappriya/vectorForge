/**
 * In-memory job store behind the MSW mock API. Produces spec-valid `JobResponse` payloads
 * that walk through the `PipelineStage` values, then succeed with a hand-written SVG
 * (colored by the job's palette) and reuse the uploaded image as the preview.
 *
 * Deterministic triggers for UI testing:
 * - a filename containing "fail" makes the job fail during `vectorize` (stage_failed);
 * - bytes that are not a PNG/JPEG header are rejected with 422 invalid_image.
 */
import { DETAIL_LEVELS, LINE_MODES, OUTPUT_FORMATS, PIPELINE_STAGES, PROCESSING_MODES } from '../api/enums';
import type {
  ConfigResponse,
  ErrorDetail,
  FileKind,
  ImageClassLabel,
  JobFile,
  JobResponse,
  JobResult,
  MetricCheck,
  OutputFormat,
  PipelineStage,
  QualityReport,
  Settings,
} from '../api/types';
import { deltaE } from './colorScience';
import type { SniffedImage } from './image';

export const MOCK_MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
export const MOCK_JOB_TTL_SECONDS = 3600;
/** Palette of samples/01_logo_4color.png (from its ground-truth JSON). */
export const MOCK_SOURCE_PALETTE = ['#ffffff', '#dc322f', '#268bd2', '#fac81e'];

export const MOCK_DEFAULT_SETTINGS: Settings = {
  mode: 'auto',
  max_colors: null,
  detail_level: 'medium',
  line_mode: 'outline',
  smoothing: 50,
  remove_background: false,
  palette_override: null,
  output_formats: ['svg', 'ai', 'eps', 'png'],
};

export const MOCK_CONFIG: ConfigResponse = {
  defaults: MOCK_DEFAULT_SETTINGS,
  limits: {
    max_upload_bytes: MOCK_MAX_UPLOAD_BYTES,
    job_ttl_seconds: MOCK_JOB_TTL_SECONDS,
    max_pixels: 36_000_000,
    accepted_media_types: ['image/png', 'image/jpeg'],
  },
};

export class SettingsError extends Error {}

const HEX_RE = /^#[0-9a-f]{6}$/;
const SETTINGS_KEYS = new Set(Object.keys(MOCK_DEFAULT_SETTINGS));

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function includes<T extends string>(list: readonly T[], value: unknown): value is T {
  return typeof value === 'string' && (list as readonly string[]).includes(value);
}

function isInt(value: unknown, min: number, max: number): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= min && value <= max;
}

/**
 * Validates and normalizes a Settings payload the way the Pydantic model does
 * (defaults for omitted fields, extra fields forbidden, SVG always first, palette de-duplicated).
 */
export function parseSettings(raw: unknown): Settings {
  if (raw === undefined || raw === null) return { ...MOCK_DEFAULT_SETTINGS };
  if (!isRecord(raw)) throw new SettingsError('settings must be a JSON object');
  for (const key of Object.keys(raw)) {
    if (!SETTINGS_KEYS.has(key)) throw new SettingsError(`settings.${key}: extra fields not permitted`);
  }
  const s = { ...MOCK_DEFAULT_SETTINGS, ...raw } as Record<string, unknown>;
  if (!includes(PROCESSING_MODES, s.mode)) throw new SettingsError('settings.mode: invalid value');
  if (s.max_colors !== null && !isInt(s.max_colors, 2, 64)) {
    throw new SettingsError('settings.max_colors: must be null or an integer from 2 to 64');
  }
  if (!includes(DETAIL_LEVELS, s.detail_level)) throw new SettingsError('settings.detail_level: invalid value');
  if (!includes(LINE_MODES, s.line_mode)) throw new SettingsError('settings.line_mode: invalid value');
  if (!isInt(s.smoothing, 0, 100)) throw new SettingsError('settings.smoothing: must be an integer from 0 to 100');
  if (typeof s.remove_background !== 'boolean') throw new SettingsError('settings.remove_background: must be boolean');
  let palette: string[] | null = null;
  if (s.palette_override !== null) {
    const p = s.palette_override;
    if (!Array.isArray(p) || p.length < 1 || p.length > 64) {
      throw new SettingsError('settings.palette_override: must be null or 1-64 colors');
    }
    for (const c of p) {
      if (typeof c !== 'string' || !HEX_RE.test(c)) {
        throw new SettingsError(`settings.palette_override: color must be lowercase '#rrggbb', got ${JSON.stringify(c)}`);
      }
    }
    palette = Array.from(new Set(p as string[]));
  }
  const formats = s.output_formats;
  if (!Array.isArray(formats) || !formats.every((f) => includes(OUTPUT_FORMATS, f))) {
    throw new SettingsError('settings.output_formats: invalid value');
  }
  const wanted = new Set<OutputFormat>([...formats, 'svg']);
  return {
    mode: s.mode,
    max_colors: s.max_colors,
    detail_level: s.detail_level,
    line_mode: s.line_mode,
    smoothing: s.smoothing,
    remove_background: s.remove_background,
    palette_override: palette,
    output_formats: OUTPUT_FORMATS.filter((f) => wanted.has(f)),
  };
}

interface Upload {
  bytes: Uint8Array;
  image: SniffedImage;
  filename: string;
}

interface MockJob {
  id: string;
  upload: Upload;
  settings: Settings;
  sourceJobId: string | null;
  createdAt: number;
  updatedAt: number;
  stageIndex: number;
  stages: PipelineStage[];
  imageClass: ImageClassLabel;
  fails: boolean;
  svg: string | null;
  deleted: boolean;
}

export interface MockStoreOptions {
  /** Simulated time per pipeline stage in ms (each poll also advances at least one stage). */
  stageMs?: number;
  now?: () => number;
}

function newId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  const hex = Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-4${hex.slice(13, 16)}-a${hex.slice(17, 20)}-${hex.slice(20)}`;
}

function iso(ms: number): string {
  return new Date(ms).toISOString();
}

function classify(settings: Settings): ImageClassLabel {
  return settings.mode === 'auto' ? 'flat_color' : settings.mode;
}

function stagesFor(imageClass: ImageClassLabel, settings: Settings): PipelineStage[] {
  const needsLines = imageClass !== 'flat_color' || settings.line_mode === 'centerline';
  return PIPELINE_STAGES.filter((s) => s !== 'extract_lines' || needsLines);
}

/** The palette the mock "pipeline" produces for these settings. */
export function resultPalette(settings: Settings): string[] {
  if (settings.palette_override) return settings.palette_override;
  const n = settings.max_colors ?? MOCK_SOURCE_PALETTE.length;
  return MOCK_SOURCE_PALETTE.slice(0, Math.max(1, Math.min(n, MOCK_SOURCE_PALETTE.length)));
}

function nearest(color: string, palette: readonly string[]): { color: string; dE: number } {
  let best = { color: palette[0] ?? color, dE: Number.POSITIVE_INFINITY };
  for (const p of palette) {
    const d = deltaE(color, p);
    if (d < best.dE) best = { color: p, dE: d };
  }
  return best;
}

const COMPARE: Record<MetricCheck['comparator'], (v: number, t: number) => boolean> = {
  '<': (v, t) => v < t,
  '<=': (v, t) => v <= t,
  '>=': (v, t) => v >= t,
  '>': (v, t) => v > t,
};

function check(name: MetricCheck['name'], value: number, comparator: MetricCheck['comparator'], threshold: number) {
  return { name, value, threshold, comparator, passed: COMPARE[comparator](value, threshold) } satisfies MetricCheck;
}

/** Fakes a QualityReport whose ΔE values reflect how far the palette is from the source colors. */
export function buildQuality(palette: readonly string[], imageClass: ImageClassLabel, svgBytes: number, w: number, h: number): QualityReport {
  const errors = MOCK_SOURCE_PALETTE.map((c) => nearest(c, palette).dE);
  const maxDeltaE = Math.round(Math.max(0.6, ...errors) * 100) / 100;
  const meanDeltaE = Math.round(Math.max(0.4, errors.reduce((a, b) => a + b, 0) / errors.length) * 100) / 100;
  const ssim = Math.round(Math.max(0.5, 0.972 - 0.002 * Math.max(0, maxDeltaE - 1)) * 1000) / 1000;
  const gapRatio = 0.00012;
  const time = 0.84;
  const budget = Math.max(2, (10 * (w * h)) / 4_000_000);
  const checks = [
    check('ssim', ssim, '>=', imageClass === 'flat_color' ? 0.9 : 0.85),
    check('mean_delta_e', meanDeltaE, '<', 2),
    check('max_delta_e', maxDeltaE, '<', 3),
    check('gap_ratio', gapRatio, '<=', 0.0005),
    check('processing_time_s', time, '<=', Math.round(budget * 100) / 100),
    check('svg_valid', 1, '>=', 1),
  ];
  return {
    ssim,
    mean_delta_e: meanDeltaE,
    max_delta_e: maxDeltaE,
    gap_ratio: gapRatio,
    alpha_iou: null,
    node_count: 40 + palette.length * 23,
    file_size_bytes: svgBytes,
    processing_time_s: time,
    checks,
    passed: checks.every((c) => c.passed),
  };
}

/** Hand-written, valid SVG with one named group per palette color (ids follow `layer_name()`). */
export function buildSvg(palette: readonly string[], width: number, height: number): string {
  const s = (v: number) => Math.round(v * 100) / 100;
  const shapes = [
    (): string => `<rect x="0" y="0" width="${width}" height="${height}"/>`,
    (): string => `<circle cx="${s(width * 0.35)}" cy="${s(height * 0.4)}" r="${s(Math.min(width, height) * 0.22)}"/>`,
    (): string => `<rect x="${s(width * 0.52)}" y="${s(height * 0.18)}" width="${s(width * 0.3)}" height="${s(height * 0.3)}" rx="${s(width * 0.02)}"/>`,
    (): string => `<path d="M ${s(width * 0.3)} ${s(height * 0.85)} L ${s(width * 0.5)} ${s(height * 0.55)} L ${s(width * 0.7)} ${s(height * 0.85)} Z"/>`,
  ];
  const layers = palette.map((color, i) => {
    const upper = color.slice(1).toUpperCase();
    const shape = shapes[i]?.() ??
      `<circle cx="${s(width * (0.1 + ((i - 4) % 8) * 0.11))}" cy="${s(height * 0.93)}" r="${s(Math.min(width, height) * 0.04)}"/>`;
    return `  <g id="color_${i + 1}_${upper}" inkscape:label="color_${i + 1}_#${upper}" inkscape:groupmode="layer" fill="${color}">${shape}</g>`;
  });
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    ...layers,
    '</svg>',
    '',
  ].join('\n');
}

const MOCK_AI = '%PDF-1.4\n% VectorForge mock .ai (PDF-compatible)\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages /Kids [] /Count 0 >> endobj\ntrailer << /Root 1 0 R >>\n%%EOF\n';
const MOCK_EPS = '%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 100 100\n%%Title: VectorForge mock\n%%EndComments\nshowpage\n%%EOF\n';

const FILE_MEDIA_TYPES: Record<FileKind, string> = {
  original: 'image/png',
  svg: 'image/svg+xml',
  ai: 'application/illustrator',
  eps: 'application/postscript',
  png: 'image/png',
};

export interface MockFileBody {
  body: Uint8Array | string;
  contentType: string;
}

export class MockJobStore {
  private readonly jobs = new Map<string, MockJob>();
  private readonly stageMs: number;
  private readonly now: () => number;

  constructor(options: MockStoreOptions = {}) {
    this.stageMs = options.stageMs ?? 350;
    this.now = options.now ?? (() => Date.now());
  }

  create(upload: Upload, settings: Settings, sourceJobId: string | null = null): JobResponse {
    const now = this.now();
    const imageClass = classify(settings);
    const job: MockJob = {
      id: newId(),
      upload,
      settings,
      sourceJobId,
      createdAt: now,
      updatedAt: now,
      stageIndex: 0,
      stages: stagesFor(imageClass, settings),
      imageClass,
      fails: /fail/i.test(upload.filename),
      svg: null,
      deleted: false,
    };
    this.jobs.set(job.id, job);
    return this.toResponse(job);
  }

  rerun(jobId: string, settings: Settings): JobResponse | null {
    const source = this.find(jobId);
    if (!source) return null;
    return this.create(source.upload, settings, source.id);
  }

  /** Returns the job, advancing it by elapsed time (and by at least one stage per poll). */
  poll(jobId: string): JobResponse | null {
    const job = this.find(jobId);
    if (!job) return null;
    if (!this.isTerminal(job)) {
      const byTime = this.stageMs > 0 ? Math.floor((this.now() - job.createdAt) / this.stageMs) : Number.POSITIVE_INFINITY;
      const next = Math.max(job.stageIndex + 1, byTime);
      const failIndex = job.fails ? job.stages.indexOf('vectorize') : -1;
      job.stageIndex = Math.min(next, failIndex >= 0 ? failIndex : job.stages.length - 1);
      job.updatedAt = this.now();
    }
    return this.toResponse(job);
  }

  delete(jobId: string): boolean {
    const job = this.find(jobId);
    if (!job) return false;
    job.deleted = true;
    return true;
  }

  file(jobId: string, kind: FileKind): MockFileBody | null {
    const job = this.find(jobId);
    if (!job) return null;
    if (kind === 'original') return { body: job.upload.bytes, contentType: job.upload.image.mediaType };
    if (!this.succeeded(job)) return null;
    const produced = this.producedKinds(job);
    if (!produced.includes(kind)) return null;
    switch (kind) {
      case 'svg':
        return { body: this.svgOf(job), contentType: FILE_MEDIA_TYPES.svg };
      case 'ai':
        return { body: MOCK_AI, contentType: FILE_MEDIA_TYPES.ai };
      case 'eps':
        return { body: MOCK_EPS, contentType: FILE_MEDIA_TYPES.eps };
      case 'png':
        // The mock has no rasterizer: the uploaded image stands in for preview.png.
        return { body: job.upload.bytes, contentType: job.upload.image.mediaType };
    }
  }

  private find(jobId: string): MockJob | null {
    const job = this.jobs.get(jobId);
    return job && !job.deleted ? job : null;
  }

  private isTerminal(job: MockJob): boolean {
    return this.succeeded(job) || this.failed(job);
  }

  private succeeded(job: MockJob): boolean {
    return !job.fails && job.stageIndex === job.stages.length - 1;
  }

  private failed(job: MockJob): boolean {
    return job.fails && job.stages[job.stageIndex] === 'vectorize';
  }

  private producedKinds(job: MockJob): FileKind[] {
    return ['original', ...(job.settings.output_formats ?? ['svg'])];
  }

  private svgOf(job: MockJob): string {
    job.svg ??= buildSvg(resultPalette(job.settings), job.upload.image.width, job.upload.image.height);
    return job.svg;
  }

  private byteLength(body: Uint8Array | string): number {
    return typeof body === 'string' ? new TextEncoder().encode(body).length : body.length;
  }

  private buildResult(job: MockJob): JobResult {
    const { width, height } = job.upload.image;
    const palette = resultPalette(job.settings);
    const svgBytes = this.byteLength(this.svgOf(job));
    const files: JobFile[] = this.producedKinds(job).map((kind) => {
      const file = this.file(job.id, kind);
      return {
        kind,
        url: `/api/v1/jobs/${job.id}/files/${kind}`,
        media_type: kind === 'original' ? job.upload.image.mediaType : FILE_MEDIA_TYPES[kind],
        size_bytes: file ? this.byteLength(file.body) : 0,
      };
    });
    const warnings: string[] = [];
    if (job.settings.palette_override && job.settings.max_colors !== null) {
      warnings.push('max_colors is ignored because palette_override is set.');
    }
    return {
      width,
      height,
      image_class: {
        label: job.imageClass,
        confidence: job.settings.mode === 'auto' ? 0.93 : 1,
        forced: job.settings.mode !== 'auto',
        features: { unique_colors: 4, edge_density: 0.031 },
      },
      layer_count: palette.length,
      palette_hex: [...palette],
      quality: buildQuality(palette, job.imageClass, svgBytes, width, height),
      files,
      warnings,
    };
  }

  private toResponse(job: MockJob): JobResponse {
    const stage = job.stages[job.stageIndex] ?? 'upload';
    const succeeded = this.succeeded(job);
    const failed = this.failed(job);
    const error: ErrorDetail | null = failed
      ? {
          code: 'stage_failed',
          message: 'Vectorization failed: the traced paths could not be closed (mock failure for filenames containing "fail").',
          stage: 'vectorize',
        }
      : null;
    const status = succeeded ? 'succeeded' : failed ? 'failed' : job.stageIndex === 0 ? 'queued' : 'running';
    const terminalAt = succeeded || failed ? job.updatedAt : this.now();
    return {
      job_id: job.id,
      status,
      stage,
      progress: succeeded ? 1 : Math.round((job.stageIndex / (job.stages.length - 1)) * 1000) / 1000,
      created_at: iso(job.createdAt),
      updated_at: iso(job.updatedAt),
      expires_at: iso(terminalAt + MOCK_JOB_TTL_SECONDS * 1000),
      filename: job.upload.filename,
      settings: job.settings,
      source_job_id: job.sourceJobId,
      result: succeeded ? this.buildResult(job) : null,
      error,
    };
  }
}
