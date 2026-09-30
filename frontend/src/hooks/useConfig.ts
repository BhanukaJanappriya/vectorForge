import { useCallback, useEffect, useState } from 'react';
import { ApiError, getConfig } from '../api/client';
import type { ConfigResponse } from '../api/types';

export type ConfigState =
  | { status: 'loading' }
  | { status: 'ready'; config: ConfigResponse }
  | { status: 'error'; error: ApiError };

/** Loads `GET /api/v1/config` (defaults + upload limits) with a retry callback. */
export function useConfig(): [ConfigState, () => void] {
  const [state, setState] = useState<ConfigState>({ status: 'loading' });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    getConfig(controller.signal).then(
      (config) => setState({ status: 'ready', config }),
      (err: unknown) => {
        if (controller.signal.aborted) return;
        setState({
          status: 'error',
          error:
            err instanceof ApiError
              ? err
              : new ApiError(0, { code: 'config_error', message: 'Could not load settings.', stage: null }),
        });
      },
    );
    return () => controller.abort();
  }, [attempt]);

  const retry = useCallback(() => {
    setState({ status: 'loading' });
    setAttempt((n) => n + 1);
  }, []);

  return [state, retry];
}
