import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError, convert, rerunJob } from '../api/client';
import { DEFAULT_POLL_OPTIONS, isTerminal, pollJob, type PollOptions } from '../api/poll';
import type { JobResponse, Settings } from '../api/types';

export type RunnerState =
  | { phase: 'idle' }
  | { phase: 'submitting'; kind: 'convert' | 'rerun'; previous: JobResponse | null }
  | { phase: 'polling'; job: JobResponse; previous: JobResponse | null }
  | { phase: 'finished'; job: JobResponse };

export interface JobRunner {
  state: RunnerState;
  /** HTTP / network error of the last request (413/415/422/404/...). Cleared on the next start. */
  error: ApiError | null;
  startConvert: (file: File, settings: Settings) => void;
  startRerun: (jobId: string, settings: Settings) => void;
  reset: () => void;
  dismissError: () => void;
}

function toApiError(err: unknown): ApiError {
  if (err instanceof ApiError) return err;
  return new ApiError(0, {
    code: 'unexpected_error',
    message: err instanceof Error ? err.message : 'Unexpected error.',
    stage: null,
  });
}

function isAbort(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

/** Drives convert/rerun -> poll with backoff. Only one job is tracked at a time. */
export function useJobRunner(pollOptions: PollOptions = DEFAULT_POLL_OPTIONS): JobRunner {
  const [state, setState] = useState<RunnerState>({ phase: 'idle' });
  const [error, setError] = useState<ApiError | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const lastFinishedRef = useRef<JobResponse | null>(null);

  useEffect(() => () => controllerRef.current?.abort(), []);

  const run = useCallback(
    (kind: 'convert' | 'rerun', submit: (signal: AbortSignal) => Promise<JobResponse>, keepPrevious: boolean) => {
      controllerRef.current?.abort();
      const controller = new AbortController();
      controllerRef.current = controller;
      const previous = keepPrevious ? lastFinishedRef.current : null;
      if (!keepPrevious) lastFinishedRef.current = null;
      setError(null);
      setState({ phase: 'submitting', kind, previous });

      const finish = (job: JobResponse) => {
        lastFinishedRef.current = job;
        setState({ phase: 'finished', job });
      };

      submit(controller.signal)
        .then(async (job) => {
          if (isTerminal(job)) {
            finish(job);
            return;
          }
          setState({ phase: 'polling', job, previous });
          const final = await pollJob(
            job.job_id,
            (update) => {
              if (!controller.signal.aborted && !isTerminal(update)) setState({ phase: 'polling', job: update, previous });
            },
            controller.signal,
            pollOptions,
          );
          if (!controller.signal.aborted) finish(final);
        })
        .catch((err: unknown) => {
          if (isAbort(err) || controller.signal.aborted) return;
          setError(toApiError(err));
          const last = lastFinishedRef.current;
          setState(last ? { phase: 'finished', job: last } : { phase: 'idle' });
        });
    },
    [pollOptions],
  );

  const startConvert = useCallback(
    (file: File, settings: Settings) => run('convert', (signal) => convert(file, settings, signal), false),
    [run],
  );
  const startRerun = useCallback(
    (jobId: string, settings: Settings) => run('rerun', (signal) => rerunJob(jobId, settings, signal), true),
    [run],
  );
  const reset = useCallback(() => {
    controllerRef.current?.abort();
    lastFinishedRef.current = null;
    setError(null);
    setState({ phase: 'idle' });
  }, []);
  const dismissError = useCallback(() => setError(null), []);

  return { state, error, startConvert, startRerun, reset, dismissError };
}
