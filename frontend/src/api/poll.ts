/** Job polling with exponential backoff (300 ms -> 2 s by default). */
import { getJob } from './client';
import type { JobResponse } from './types';

export interface PollOptions {
  /** First delay before polling, in ms. */
  initialDelayMs: number;
  /** Upper bound for the delay, in ms. */
  maxDelayMs: number;
  /** Multiplier applied after each poll. */
  factor: number;
}

export const DEFAULT_POLL_OPTIONS: PollOptions = { initialDelayMs: 300, maxDelayMs: 2000, factor: 1.5 };

/** Delay schedule: initial, initial*factor, ... capped at maxDelayMs. */
export function nextDelay(previousMs: number, options: PollOptions = DEFAULT_POLL_OPTIONS): number {
  return Math.min(options.maxDelayMs, Math.max(options.initialDelayMs, Math.round(previousMs * options.factor)));
}

export function isTerminal(job: JobResponse): boolean {
  return job.status === 'succeeded' || job.status === 'failed';
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException('Aborted', 'AbortError'));
    };
    signal.addEventListener('abort', onAbort, { once: true });
  });
}

/**
 * Polls `GET /api/v1/jobs/{id}` until the job succeeds or fails, calling `onUpdate`
 * with every response. Resolves with the terminal job. Rejects with AbortError when aborted
 * and with ApiError on HTTP errors.
 */
export async function pollJob(
  jobId: string,
  onUpdate: (job: JobResponse) => void,
  signal: AbortSignal,
  options: PollOptions = DEFAULT_POLL_OPTIONS,
): Promise<JobResponse> {
  let delay = options.initialDelayMs;
  for (;;) {
    await sleep(delay, signal);
    const job = await getJob(jobId, signal);
    onUpdate(job);
    if (isTerminal(job)) return job;
    delay = nextDelay(delay, options);
  }
}
