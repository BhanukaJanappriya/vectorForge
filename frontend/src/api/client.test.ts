import { http, HttpResponse } from 'msw';
import { describe, expect, it } from 'vitest';
import { server } from '../mocks/node';
import { makePng } from '../test/fixtures';
import { ApiError, buildConvertForm, convert, fetchJobFile, getConfig, parseErrorBody, rerunJob } from './client';

describe('parseErrorBody', () => {
  it('reads the spec ErrorResponse', () => {
    expect(parseErrorBody(413, { error: { code: 'too_large', message: 'Too big', stage: 'upload' } })).toEqual({
      code: 'too_large',
      message: 'Too big',
      stage: 'upload',
    });
  });

  it('falls back for FastAPI detail bodies and plain text', () => {
    expect(parseErrorBody(422, { detail: [{ msg: 'x' }] }).code).toBe('invalid_request');
    expect(parseErrorBody(502, undefined, 'Bad gateway').message).toBe('Bad gateway');
    expect(parseErrorBody(500, undefined).message).toContain('HTTP 500');
  });
});

describe('client', () => {
  it('loads config with limits and defaults', async () => {
    const config = await getConfig();
    expect(config.limits.max_upload_bytes).toBe(20 * 1024 * 1024);
    expect(config.defaults.output_formats).toContain('svg');
  });

  it('builds a multipart body with the file and JSON settings', async () => {
    const { defaults } = await getConfig();
    const form = buildConvertForm(makePng('a.png'), defaults);
    expect(form.get('file')).toBeInstanceOf(File);
    const settings = form.get('settings');
    expect(typeof settings).toBe('string');
    expect(JSON.parse(settings as string)).toEqual(defaults);
  });

  it.each([
    [413, 'too_large'],
    [415, 'unsupported_media_type'],
    [422, 'invalid_image'],
  ])('maps HTTP %i to an ApiError', async (status, code) => {
    server.use(
      http.post('/api/v1/convert', () => HttpResponse.json({ error: { code, message: 'nope', stage: 'upload' } }, { status })),
    );
    const { defaults } = await getConfig();
    const err: unknown = await convert(makePng('a.png'), defaults).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status, detail: { code, stage: 'upload' } });
  });

  it('reports network failures as status 0', async () => {
    server.use(http.get('/api/v1/config', () => HttpResponse.error()));
    await expect(getConfig()).rejects.toMatchObject({ status: 0, detail: { code: 'network_error' } });
  });

  it('rerun of an unknown job is a 404 ApiError', async () => {
    const { defaults } = await getConfig();
    await expect(rerunJob('missing', defaults)).rejects.toMatchObject({ status: 404 });
  });

  it('mock rejects invalid settings with 422 invalid_settings', async () => {
    const { defaults } = await getConfig();
    const bad = { ...defaults, palette_override: ['#GGGGGG'] };
    await expect(convert(makePng('a.png'), bad)).rejects.toMatchObject({ status: 422, detail: { code: 'invalid_settings' } });
  });

  it('mock rejects undecodable bytes with 422 invalid_image', async () => {
    const { defaults } = await getConfig();
    const junk = new File(['not an image'], 'x.png', { type: 'image/png' });
    await expect(convert(junk, defaults)).rejects.toMatchObject({ status: 422, detail: { code: 'invalid_image' } });
  });

  it('mock rejects other media types with 415', async () => {
    const { defaults } = await getConfig();
    const gif = new File(['GIF89a'], 'x.gif', { type: 'image/gif' });
    await expect(convert(gif, defaults)).rejects.toMatchObject({ status: 415 });
  });

  it('fetchJobFile surfaces 404 for files that are not ready', async () => {
    await expect(
      fetchJobFile({ kind: 'svg', url: '/api/v1/jobs/nope/files/svg', media_type: 'image/svg+xml', size_bytes: 0 }),
    ).rejects.toMatchObject({ status: 404 });
  });
});
