import { http, HttpResponse } from 'msw';
import { describe, expect, it } from 'vitest';
import { server } from '../mocks/node';
import { makePng } from '../test/fixtures';
import { convert, getConfig } from './client';
import { PIPELINE_STAGES } from './enums';
import { DEFAULT_POLL_OPTIONS, isTerminal, nextDelay, pollJob } from './poll';
import type { JobResponse } from './types';

const FAST = { initialDelayMs: 1, maxDelayMs: 2, factor: 1.5 };

describe('nextDelay', () => {
  it('backs off from 300 ms to a 2 s cap', () => {
    const schedule = [DEFAULT_POLL_OPTIONS.initialDelayMs];
    for (let i = 0; i < 7; i++) schedule.push(nextDelay(schedule.at(-1) ?? 0));
    expect(schedule).toEqual([300, 450, 675, 1013, 1520, 2000, 2000, 2000]);
  });
});

describe('pollJob', () => {
  it('walks the pipeline stages in order until the job succeeds', async () => {
    const { defaults } = await getConfig();
    const job = await convert(makePng('logo.png'), defaults);
    expect(job.status).toBe('queued');
    expect(job.stage).toBe('upload');
    const seen: JobResponse[] = [];
    const final = await pollJob(job.job_id, (j) => seen.push(j), new AbortController().signal, FAST);
    expect(final.status).toBe('succeeded');
    expect(final.stage).toBe('done');
    expect(final.progress).toBe(1);
    expect(isTerminal(final)).toBe(true);
    const indices = seen.map((j) => PIPELINE_STAGES.indexOf(j.stage));
    expect(indices).toEqual([...indices].sort((a, b) => a - b));
    expect(seen.slice(0, -1).every((j) => j.status === 'running' && j.result === null)).toBe(true);
  });

  it('ends in failed with an ErrorDetail for failing jobs', async () => {
    const { defaults } = await getConfig();
    const job = await convert(makePng('please_fail.png'), defaults);
    const final = await pollJob(job.job_id, () => undefined, new AbortController().signal, FAST);
    expect(final.status).toBe('failed');
    expect(final.result).toBeNull();
    expect(final.error).toMatchObject({ code: 'stage_failed', stage: 'vectorize' });
  });

  it('stops when aborted', async () => {
    const controller = new AbortController();
    const promise = pollJob('any', () => undefined, controller.signal, { initialDelayMs: 50, maxDelayMs: 50, factor: 1 });
    controller.abort();
    await expect(promise).rejects.toMatchObject({ name: 'AbortError' });
  });

  it('rejects with ApiError on 404', async () => {
    server.use(
      http.get('/api/v1/jobs/:id', () =>
        HttpResponse.json({ error: { code: 'not_found', message: 'Unknown job.', stage: null } }, { status: 404 }),
      ),
    );
    await expect(pollJob('gone', () => undefined, new AbortController().signal, FAST)).rejects.toMatchObject({
      status: 404,
      detail: { code: 'not_found' },
    });
  });
});
