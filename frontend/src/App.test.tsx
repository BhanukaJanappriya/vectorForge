import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { describe, expect, it, vi } from 'vitest';
import App from './App';
import type { RerunRequest } from './api/types';
import { server } from './mocks/node';
import { makePng } from './test/fixtures';

const FAST = { initialDelayMs: 2, maxDelayMs: 10, factor: 1.5 };

async function renderReady() {
  const user = userEvent.setup();
  render(<App pollOptions={FAST} />);
  await screen.findByRole('button', { name: 'Choose image' });
  return user;
}

async function upload(user: ReturnType<typeof userEvent.setup>, file = makePng('logo.png')) {
  await user.upload(screen.getByLabelText('Image file'), file);
}

describe('App', () => {
  it('converts, shows the result, re-runs with an edited palette and downloads', async () => {
    const user = await renderReady();
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
    await upload(user);
    expect(screen.getByTestId('selected-file')).toHaveTextContent('logo.png');

    await user.click(screen.getByRole('button', { name: 'Convert' }));
    expect(await screen.findByTestId('progress')).toBeInTheDocument();
    expect(await screen.findByTestId('result', {}, { timeout: 5000 })).toBeInTheDocument();
    expect(screen.queryByTestId('progress')).not.toBeInTheDocument();
    expect(screen.getByTestId('announcer')).toHaveTextContent('Conversion complete.');
    expect(screen.getByTestId('quality-badge')).toHaveTextContent('All 6 quality checks passed');
    expect(screen.getAllByTestId('palette-entry')).toHaveLength(4);
    expect(screen.getByAltText('Original image').getAttribute('src')).toMatch(/^blob:/);

    let rerunBody: RerunRequest | null = null;
    server.events.on('request:start', ({ request }) => {
      if (request.url.endsWith('/rerun')) {
        void request
          .clone()
          .json()
          .then((b) => {
            rerunBody = b as RerunRequest;
          });
      }
    });

    const hex = screen.getByRole('textbox', { name: 'Color 3 hex value' });
    await user.clear(hex);
    await user.type(hex, '#00ff00');
    await user.click(screen.getByRole('button', { name: 'Re-run with edited palette' }));
    expect(await screen.findByTestId('rerun-note', {}, { timeout: 5000 })).toHaveTextContent('custom palette');
    await waitFor(() => expect(screen.queryByTestId('progress')).not.toBeInTheDocument());
    expect(rerunBody).toMatchObject({ settings: { palette_override: ['#ffffff', '#dc322f', '#00ff00', '#fac81e'] } });
    server.events.removeAllListeners();
    expect(screen.getByRole('textbox', { name: 'Color 3 hex value' })).toHaveValue('#00ff00');
    expect(screen.getByRole('button', { name: 'Use automatic palette' })).toBeInTheDocument();
    expect(screen.getByText(/custom palette \(4 colors\)/)).toBeInTheDocument();

    const createUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:download');
    const clicks: string[] = [];
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      clicks.push(this.download);
    });
    await user.click(screen.getByRole('button', { name: /Download EPS/ }));
    await waitFor(() => expect(clicks).toEqual(['logo.eps']));
    createUrl.mockRestore();
    click.mockRestore();
  });

  it('re-runs with changed settings without re-uploading', async () => {
    const user = await renderReady();
    await upload(user);
    await user.click(screen.getByRole('button', { name: 'Convert' }));
    await screen.findByTestId('result', {}, { timeout: 5000 });
    let converts = 0;
    server.events.on('request:start', ({ request }) => {
      if (request.url.endsWith('/convert')) converts += 1;
    });
    await user.selectOptions(screen.getByLabelText('Mode'), 'mixed');
    await user.click(screen.getByRole('button', { name: 'Re-run with these settings' }));
    await waitFor(() => expect(screen.getByTestId('image-class')).toHaveTextContent('Mixed (forced)'), { timeout: 5000 });
    expect(converts).toBe(0);
    server.events.removeAllListeners();

    await user.click(screen.getByRole('button', { name: 'Start over' }));
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Convert' })).toBeDisabled();
  });

  it.each([
    [413, 'too_large', 'File too large'],
    [415, 'unsupported_media_type', 'Unsupported file type'],
    [422, 'invalid_image', 'Image could not be read'],
  ])('shows HTTP %i from /convert', async (status, code, title) => {
    server.use(
      http.post('/api/v1/convert', () =>
        HttpResponse.json({ error: { code, message: `server says ${code}`, stage: 'upload' } }, { status }),
      ),
    );
    const user = await renderReady();
    await upload(user);
    await user.click(screen.getByRole('button', { name: 'Convert' }));
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(title);
    expect(alert).toHaveTextContent(`server says ${code}`);
    expect(screen.queryByTestId('result')).not.toBeInTheDocument();
    await user.click(within(alert).getByRole('button', { name: 'Dismiss' }));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('shows the real mock 422 for undecodable images', async () => {
    const user = await renderReady();
    await upload(user, new File(['definitely not a png'], 'broken.png', { type: 'image/png' }));
    await user.click(screen.getByRole('button', { name: 'Convert' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Image could not be read');
  });

  it('rejects unsupported files before uploading', async () => {
    const user = userEvent.setup({ applyAccept: false });
    render(<App pollOptions={FAST} />);
    await screen.findByRole('button', { name: 'Choose image' });
    await user.upload(screen.getByLabelText('Image file'), new File(['GIF89a'], 'anim.gif', { type: 'image/gif' }));
    expect(screen.getByRole('alert')).toHaveTextContent('File not accepted');
    expect(screen.getByRole('button', { name: 'Convert' })).toBeDisabled();
  });

  it('shows failed jobs with their stage and error code', async () => {
    const user = await renderReady();
    await upload(user, makePng('this_will_fail.png'));
    await user.click(screen.getByRole('button', { name: 'Convert' }));
    const alert = await screen.findByRole('alert', {}, { timeout: 5000 });
    expect(alert).toHaveTextContent('Conversion failed');
    expect(alert).toHaveTextContent('stage: Tracing vectors');
    expect(alert).toHaveTextContent('code: stage_failed');
    expect(screen.getByTestId('announcer')).toHaveTextContent('Conversion failed.');
  });

  it('shows a retryable error when config cannot be loaded', async () => {
    server.use(
      http.get('/api/v1/config', () =>
        HttpResponse.json({ error: { code: 'unavailable', message: 'down', stage: null } }, { status: 503 }),
      ),
    );
    const user = userEvent.setup();
    render(<App pollOptions={FAST} />);
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('Server error');
    server.resetHandlers();
    await user.click(within(alert).getByRole('button', { name: 'Try again' }));
    expect(await screen.findByRole('button', { name: 'Choose image' })).toBeInTheDocument();
  });

  it('surfaces polling errors (e.g. expired job) and keeps the app usable', async () => {
    server.use(
      http.get('/api/v1/jobs/:id', () =>
        HttpResponse.json({ error: { code: 'not_found', message: 'Unknown job.', stage: null } }, { status: 404 }),
      ),
    );
    const user = await renderReady();
    await upload(user);
    await user.click(screen.getByRole('button', { name: 'Convert' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Job not found or expired');
    expect(screen.getByRole('button', { name: 'Convert' })).toBeEnabled();
  });
});
