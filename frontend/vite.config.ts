import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { loadEnv } from 'vite';
import { defineConfig } from 'vitest/config';

/**
 * Modes:
 * - `development` / `production`: talk to the real API. In dev, `/api` is proxied to
 *   `VITE_API_TARGET` (default http://localhost:8000); in Docker nginx proxies `/api` on the same origin.
 * - `mock`: the MSW service worker answers `/api/v1/*` from `src/mocks/` (see `.env.mock`).
 *   Only this mode serves `mock-public/mockServiceWorker.js`, so production builds never ship the mock.
 */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const apiTarget = env.VITE_API_TARGET ?? 'http://localhost:8000';
  return {
    plugins: [react(), tailwindcss()],
    publicDir: mode === 'mock' ? 'mock-public' : 'public',
    server: {
      port: 5173,
      proxy: { '/api': { target: apiTarget, changeOrigin: true } },
    },
    preview: {
      port: 4173,
      proxy: { '/api': { target: apiTarget, changeOrigin: true } },
    },
    test: {
      environment: 'jsdom',
      setupFiles: ['./src/test/setup.ts'],
      include: ['src/**/*.test.{ts,tsx}'],
      css: false,
      testTimeout: 15000,
      coverage: {
        provider: 'v8',
        include: ['src/**/*.{ts,tsx}'],
        exclude: ['src/**/*.test.{ts,tsx}', 'src/test/**', 'src/main.tsx', 'src/mocks/browser.ts', 'src/**/*.d.ts'],
        reporter: ['text-summary', 'text'],
      },
    },
  };
});
