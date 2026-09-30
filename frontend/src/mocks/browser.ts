/** Browser MSW worker (only loaded in `--mode mock`). */
import { setupWorker } from 'msw/browser';
import { createHandlers } from './handlers';
import { MockJobStore } from './store';

export const worker = setupWorker(...createHandlers(new MockJobStore()));
