import { describe, expect, it } from 'vitest';
import type { Limits } from '../api/types';
import { buildPaletteOverride, dedupePalette, luminance, normalizeHex } from './color';
import { errorTitle } from './errors';
import { baseName, downloadName, formatBytes, formatMetric } from './format';
import { acceptAttribute, fileMediaType, validateUpload } from './upload';
import { IDENTITY, MAX_SCALE, clampView, panBy, toTransform, zoomAt } from './viewport';

const LIMITS: Limits = {
  max_upload_bytes: 1000,
  job_ttl_seconds: 3600,
  max_pixels: 36_000_000,
  accepted_media_types: ['image/png', 'image/jpeg'],
};

describe('color', () => {
  it('normalizes hex input to lowercase #rrggbb', () => {
    expect(normalizeHex('#ABC')).toBe('#aabbcc');
    expect(normalizeHex(' B22222 ')).toBe('#b22222');
    expect(normalizeHex('#12345')).toBeNull();
    expect(normalizeHex('red')).toBeNull();
  });

  it('builds a de-duplicated palette_override', () => {
    expect(dedupePalette(['#000000', '#ffffff', '#000000'])).toEqual(['#000000', '#ffffff']);
    expect(buildPaletteOverride([{ hex: '#111111' }, { hex: '#111111' }, { hex: '#222222' }])).toEqual(['#111111', '#222222']);
    expect(buildPaletteOverride(Array.from({ length: 70 }, (_, i) => ({ hex: `#0000${i.toString(16).padStart(2, '0')}` })))).toHaveLength(64);
  });

  it('computes luminance', () => {
    expect(luminance('#ffffff')).toBeCloseTo(1);
    expect(luminance('#000000')).toBeCloseTo(0);
  });
});

describe('format', () => {
  it('formats bytes and metrics', () => {
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(2048)).toBe('2.0 KB');
    expect(formatBytes(20 * 1024 * 1024)).toBe('20 MB');
    expect(formatMetric('ssim', 0.97234)).toBe('0.972');
    expect(formatMetric('gap_ratio', 0.00012)).toBe('0.00012');
    expect(formatMetric('processing_time_s', 1.234)).toBe('1.23 s');
    expect(formatMetric('svg_valid', 0)).toBe('no');
  });

  it('names downloads after the upload', () => {
    expect(baseName('logo.final.png')).toBe('logo.final');
    expect(downloadName('logo.png', 'svg')).toBe('logo.svg');
    expect(downloadName('logo.png', 'ai')).toBe('logo.ai');
    expect(downloadName('logo.png', 'png')).toBe('logo_preview.png');
    expect(downloadName('.png', 'eps')).toBe('output.eps');
  });
});

describe('errors', () => {
  it.each([
    [413, 'too_large', 'File too large'],
    [415, 'unsupported_media_type', 'Unsupported file type'],
    [422, 'invalid_image', 'Image could not be read'],
    [422, 'invalid_settings', 'Invalid settings'],
    [422, 'other', 'Request rejected'],
    [404, 'not_found', 'Job not found or expired'],
    [0, 'network_error', 'Connection problem'],
    [500, 'boom', 'Server error'],
    [418, 'teapot', 'Something went wrong'],
  ])('HTTP %i %s -> %s', (status, code, title) => {
    expect(errorTitle(status, { code, message: '', stage: null })).toBe(title);
  });
});

describe('upload validation', () => {
  it('accepts PNG/JPG within limits', () => {
    expect(validateUpload(new File(['x'], 'a.png', { type: 'image/png' }), LIMITS)).toBeNull();
    expect(validateUpload(new File(['x'], 'a.JPG'), LIMITS)).toBeNull();
    expect(fileMediaType(new File(['x'], 'a.jpeg'))).toBe('image/jpeg');
  });

  it('rejects other types, oversize and empty files', () => {
    expect(validateUpload(new File(['x'], 'a.gif', { type: 'image/gif' }), LIMITS)).toContain('not a supported image');
    expect(validateUpload(new File(['x'.repeat(1001)], 'a.png', { type: 'image/png' }), LIMITS)).toContain('maximum upload size');
    expect(validateUpload(new File([], 'a.png', { type: 'image/png' }), LIMITS)).toContain('empty');
  });

  it('builds the accept attribute from limits', () => {
    expect(acceptAttribute(LIMITS)).toBe('image/png,image/jpeg,.png,.jpg,.jpeg');
    expect(acceptAttribute({ ...LIMITS, accepted_media_types: ['image/png'] })).toBe('image/png,.png');
  });
});

describe('viewport', () => {
  it('zooms around a point and keeps content covering the viewport', () => {
    const v = zoomAt(IDENTITY, 2, 0.5, 0.5);
    expect(v).toEqual({ scale: 2, x: -0.5, y: -0.5 });
    expect(zoomAt(v, 0.5)).toEqual(IDENTITY);
    expect(zoomAt(IDENTITY, 1000).scale).toBe(MAX_SCALE);
    expect(zoomAt(IDENTITY, 2, 0, 0)).toEqual({ scale: 2, x: 0, y: 0 });
  });

  it('clamps panning', () => {
    const v = zoomAt(IDENTITY, 2);
    expect(panBy(v, 10, 10)).toEqual({ scale: 2, x: 0, y: 0 });
    expect(panBy(v, -10, -10)).toEqual({ scale: 2, x: -1, y: -1 });
    expect(panBy(IDENTITY, 0.3, 0.3)).toEqual(IDENTITY);
    expect(clampView({ scale: 0.1, x: 1, y: 1 })).toEqual(IDENTITY);
  });

  it('renders a CSS transform', () => {
    expect(toTransform({ scale: 2, x: -0.5, y: -0.25 })).toBe('translate(-50%, -25%) scale(2)');
  });
});
