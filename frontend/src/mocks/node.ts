/** Node MSW server for Vitest. Fast stage timing so component tests run quickly. */
import { setupServer } from 'msw/node';
import { createHandlers } from './handlers';
import { MockJobStore } from './store';

export const mockStore = new MockJobStore({ stageMs: 5 });
export const server = setupServer(...createHandlers(mockStore));
