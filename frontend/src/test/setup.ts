import '@testing-library/jest-dom/vitest';
import { File as NodeFile } from 'node:buffer';
import { cleanup } from '@testing-library/react';
import { afterAll, afterEach, beforeAll } from 'vitest';
import { server } from '../mocks/node';
import { fileBytes } from './fixtures';

// jsdom lacks object URLs; the app only needs stable strings.
if (typeof URL.createObjectURL !== 'function') {
  Object.assign(URL, { createObjectURL: () => 'blob:mock', revokeObjectURL: () => undefined });
}

// Vitest converts jsdom FormData for Node's fetch but drops file names (parts become "blob").
// Convert multipart bodies ourselves so the mock API sees real filenames, as a browser would send.
const NodeFormData = (
  await new Response('a=1', { headers: { 'content-type': 'application/x-www-form-urlencoded' } }).formData()
).constructor as typeof FormData;
function installMultipartShim() {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
  if (init?.body instanceof FormData) {
    const converted = new NodeFormData();
    for (const [key, value] of init.body.entries()) {
      if (typeof value === 'string') converted.append(key, value);
      else converted.append(key, new NodeFile([await fileBytes(value)], value.name, { type: value.type }) as unknown as Blob);
    }
    return originalFetch(input, { ...init, body: converted });
  }
  return originalFetch(input, init);
  };
}

beforeAll(() => {
  server.listen({ onUnhandledRequest: 'error' });
  // Installed after MSW so it runs before MSW's interceptor builds the Request.
  installMultipartShim();
});
afterEach(() => {
  cleanup();
  server.resetHandlers();
});
afterAll(() => server.close());
