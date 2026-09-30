import { describe, expect, it } from 'vitest';
import { PIPELINE_STAGES } from '../api/enums';
import type { MetricCheck } from '../api/types';
import { sniffImage } from './image';
import { MOCK_DEFAULT_SETTINGS, MOCK_SOURCE_PALETTE, MockJobStore, SettingsError, buildQuality, buildSvg, parseSettings, resultPalette } from './store';

const PNG_HEADER = new Uint8Array([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13, 0x49, 0x48, 0x44, 0x52, 0, 0, 1, 0, 0, 0, 0, 200, 8, 6, 0, 0, 0,
]);

function upload(filename = 'logo.png') {
  const image = sniffImage(PNG_HEADER);
  if (!image) throw new Error('fixture must decode');
  return { bytes: PNG_HEADER, image, filename };
}

const COMPARE: Record<MetricCheck['comparator'], (v: number, t: number) => boolean> = {
  '<': (v, t) => v < t,
  '<=': (v, t) => v <= t,
  '>=': (v, t) => v >= t,
  '>': (v, t) => v > t,
};

describe('sniffImage', () => {
  it('reads PNG and JPEG sizes and rejects junk', () => {
    expect(sniffImage(PNG_HEADER)).toEqual({ mediaType: 'image/png', width: 256, height: 200 });
    const jpeg = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0, 4, 0, 0, 0xff, 0xc0, 0, 17, 8, 0, 48, 0, 64, 3, 1, 0x22, 0]);
    expect(sniffImage(jpeg)).toEqual({ mediaType: 'image/jpeg', width: 64, height: 48 });
    expect(sniffImage(new TextEncoder().encode('hello world, not an image'))).toBeNull();
  });
});

describe('parseSettings', () => {
  it('fills defaults, puts SVG first and de-duplicates the palette (like the Pydantic model)', () => {
    const s = parseSettings({ output_formats: ['png', 'eps'], palette_override: ['#ffffff', '#ffffff', '#000000'] });
    expect(s.output_formats).toEqual(['svg', 'eps', 'png']);
    expect(s.palette_override).toEqual(['#ffffff', '#000000']);
    expect(s.mode).toBe('auto');
    expect(parseSettings(undefined)).toEqual(MOCK_DEFAULT_SETTINGS);
  });

  it.each([
    [{ max_colors: 1 }],
    [{ max_colors: 65 }],
    [{ smoothing: 101 }],
    [{ mode: 'photo' }],
    [{ palette_override: [] }],
    [{ palette_override: ['#FFFFFF'] }],
    [{ output_formats: ['pdf'] }],
    [{ unknown: true }],
    ['not an object'],
  ])('rejects %j', (raw) => {
    expect(() => parseSettings(raw)).toThrow(SettingsError);
  });
});

describe('MockJobStore', () => {
  it('walks every stage in order (lines only when needed) and succeeds with a valid result', () => {
    let now = 0;
    const store = new MockJobStore({ stageMs: 1_000_000, now: () => now });
    const created = store.create(upload(), parseSettings({ mode: 'line_art' }));
    expect(created).toMatchObject({ status: 'queued', stage: 'upload', progress: 0, result: null, error: null });
    const stages = [created.stage];
    for (let i = 0; i < 20; i++) {
      now += 1;
      const job = store.poll(created.job_id);
      if (!job) throw new Error('job vanished');
      stages.push(job.stage);
      if (job.status === 'succeeded') {
        expect(job.result?.files.map((f) => f.kind)).toEqual(['original', 'svg', 'ai', 'eps', 'png']);
        expect(job.result?.image_class).toMatchObject({ label: 'line_art', forced: true });
        break;
      }
      expect(job.status).toBe('running');
    }
    expect(stages).toEqual([...PIPELINE_STAGES]);
  });

  it('skips extract_lines for flat color with outline mode', () => {
    const store = new MockJobStore({ stageMs: 1_000_000 });
    const job = store.create(upload(), parseSettings({}));
    const seen = [job.stage];
    for (let i = 0; i < 20; i++) {
      const next = store.poll(job.job_id);
      if (!next) break;
      seen.push(next.stage);
      if (next.status === 'succeeded') break;
    }
    expect(seen).not.toContain('extract_lines');
    expect(seen.at(-1)).toBe('done');
  });

  it('fails jobs whose filename contains "fail" at vectorize', () => {
    const store = new MockJobStore({ stageMs: 1_000_000 });
    const job = store.create(upload('fail.png'), parseSettings({}));
    let last = job;
    for (let i = 0; i < 20 && last.status !== 'failed'; i++) last = store.poll(job.job_id) ?? last;
    expect(last).toMatchObject({ status: 'failed', stage: 'vectorize', result: null, error: { code: 'stage_failed' } });
    expect(store.file(job.job_id, 'svg')).toBeNull();
  });

  it('re-runs keep the upload and link source_job_id; delete removes the job', () => {
    const store = new MockJobStore();
    const job = store.create(upload(), parseSettings({}));
    const rerun = store.rerun(job.job_id, parseSettings({ palette_override: ['#000000'] }));
    expect(rerun?.source_job_id).toBe(job.job_id);
    expect(rerun?.filename).toBe('logo.png');
    expect(store.rerun('nope', MOCK_DEFAULT_SETTINGS)).toBeNull();
    expect(store.delete(job.job_id)).toBe(true);
    expect(store.poll(job.job_id)).toBeNull();
  });

  it('only serves requested formats', () => {
    const store = new MockJobStore({ stageMs: 0 });
    const job = store.create(upload(), parseSettings({ output_formats: ['svg'] }));
    const done = store.poll(job.job_id);
    expect(done?.status).toBe('succeeded');
    expect(done?.result?.files.map((f) => f.kind)).toEqual(['original', 'svg']);
    expect(store.file(job.job_id, 'eps')).toBeNull();
    expect(store.file(job.job_id, 'original')?.contentType).toBe('image/png');
  });
});

describe('buildQuality', () => {
  it('produces checks consistent with their comparators and passes for the source palette', () => {
    const q = buildQuality(MOCK_SOURCE_PALETTE, 'flat_color', 800, 512, 512);
    for (const c of q.checks) expect(c.passed).toBe(COMPARE[c.comparator](c.value, c.threshold));
    expect(q.passed).toBe(true);
  });

  it('fails the ΔE checks when colors are merged away', () => {
    const q = buildQuality(['#ffffff', '#dc322f'], 'flat_color', 800, 512, 512);
    expect(q.passed).toBe(false);
    expect(q.checks.find((c) => c.name === 'max_delta_e')?.passed).toBe(false);
  });

  it('uses max_colors when there is no override', () => {
    expect(resultPalette(parseSettings({ max_colors: 2 }))).toEqual(MOCK_SOURCE_PALETTE.slice(0, 2));
  });
});

describe('buildSvg', () => {
  it('is well-formed SVG with one layer per color named like layer_name()', () => {
    const svg = buildSvg(['#ffffff', '#dc322f', '#268bd2', '#fac81e', '#000000'], 512, 256);
    const doc = new DOMParser().parseFromString(svg, 'image/svg+xml');
    expect(doc.getElementsByTagName('parsererror')).toHaveLength(0);
    const groups = Array.from(doc.getElementsByTagName('g'));
    expect(groups.map((g) => g.getAttribute('id'))).toEqual([
      'color_1_FFFFFF',
      'color_2_DC322F',
      'color_3_268BD2',
      'color_4_FAC81E',
      'color_5_000000',
    ]);
    expect(doc.documentElement.getAttribute('viewBox')).toBe('0 0 512 256');
  });
});
