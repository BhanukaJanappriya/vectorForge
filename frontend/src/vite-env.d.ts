/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "true" enables the MSW mock API (set by `.env.mock`). */
  readonly VITE_API_MOCK?: string;
  /** Dev/preview proxy target for /api (default http://localhost:8000). */
  readonly VITE_API_TARGET?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
