import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

/** In `--mode mock` the MSW service worker serves the API; the import is tree-shaken otherwise. */
async function enableMocking(): Promise<void> {
  if (import.meta.env.VITE_API_MOCK !== 'true') return;
  const { worker } = await import('./mocks/browser');
  await worker.start({ onUnhandledRequest: 'bypass', quiet: true });
}

void enableMocking().then(() => {
  const root = document.getElementById('root');
  if (!root) throw new Error('#root element missing');
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
});
