import { act, fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '../api/client';
import type { JobResponse, Settings } from '../api/types';
import { MOCK_DEFAULT_SETTINGS, MOCK_SOURCE_PALETTE, buildQuality } from '../mocks/store';
import { ComparisonViewer } from './ComparisonViewer';
import { Downloads } from './Downloads';
import { ApiErrorBanner } from './ErrorBanner';
import { PaletteEditor } from './PaletteEditor';
import { ProgressIndicator } from './ProgressIndicator';
import { QualityBadge } from './QualityBadge';
import { SettingsPanel } from './SettingsPanel';
import { UploadDropzone } from './UploadDropzone';

function ControlledSettings({ onChange }: { onChange: (s: Settings) => void }) {
  const [value, setValue] = useState<Settings>(MOCK_DEFAULT_SETTINGS);
  return (
    <SettingsPanel
      value={value}
      onChange={(s) => {
        setValue(s);
        onChange(s);
      }}
    />
  );
}

function job(overrides: Partial<JobResponse> = {}): JobResponse {
  return {
    job_id: 'j1',
    status: 'running',
    stage: 'quantize',
    progress: 0.375,
    created_at: '2026-09-30T10:00:00Z',
    updated_at: '2026-09-30T10:00:01Z',
    expires_at: '2026-09-30T11:00:01Z',
    filename: 'logo.png',
    settings: MOCK_DEFAULT_SETTINGS,
    source_job_id: null,
    result: null,
    error: null,
    ...overrides,
  };
}

describe('SettingsPanel', () => {
  it('exposes every setting as a labelled control', () => {
    render(<SettingsPanel value={MOCK_DEFAULT_SETTINGS} onChange={() => undefined} />);
    expect(screen.getByLabelText('Mode')).toHaveValue('auto');
    expect(screen.getByRole('checkbox', { name: 'Auto-detect number of colors' })).toBeChecked();
    expect(screen.getByRole('slider', { name: /Max colors/ })).toBeDisabled();
    expect(screen.getByRole('radio', { name: 'Medium' })).toBeChecked();
    expect(screen.getByRole('radio', { name: 'Outline' })).toBeChecked();
    expect(screen.getByRole('slider', { name: /Smoothing/ })).toHaveValue('50');
    expect(screen.getByRole('checkbox', { name: /Remove background/ })).not.toBeChecked();
    const svg = screen.getByRole('checkbox', { name: 'SVG (always)' });
    expect(svg).toBeChecked();
    expect(svg).toBeDisabled();
  });

  it('emits spec-shaped settings', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn<(s: Settings) => void>();
    render(<ControlledSettings onChange={onChange} />);
    await user.selectOptions(screen.getByLabelText('Mode'), 'line_art');
    await user.click(screen.getByRole('checkbox', { name: 'Auto-detect number of colors' }));
    const colors = screen.getByRole('slider', { name: /Max colors/ });
    expect(colors).toBeEnabled();
    fireEvent.change(colors, { target: { value: '64' } });
    await user.click(screen.getByRole('radio', { name: 'High' }));
    await user.click(screen.getByRole('radio', { name: 'Centerline' }));
    fireEvent.change(screen.getByRole('slider', { name: /Smoothing/ }), { target: { value: '0' } });
    await user.click(screen.getByRole('checkbox', { name: /Remove background/ }));
    await user.click(screen.getByRole('checkbox', { name: 'EPS' }));
    await user.click(screen.getByRole('checkbox', { name: 'AI (Illustrator)' }));
    expect(onChange).toHaveBeenLastCalledWith({
      mode: 'line_art',
      max_colors: 64,
      detail_level: 'high',
      line_mode: 'centerline',
      smoothing: 0,
      remove_background: true,
      palette_override: null,
      output_formats: ['svg', 'png'],
    });
    await user.click(screen.getByRole('checkbox', { name: 'Auto-detect number of colors' }));
    expect(onChange.mock.lastCall?.[0].max_colors).toBeNull();
  });

  it('notes that max colors is ignored while a palette override is active', () => {
    render(<SettingsPanel value={MOCK_DEFAULT_SETTINGS} onChange={() => undefined} paletteOverrideCount={3} />);
    expect(screen.getByText(/custom palette \(3 colors\)/)).toBeInTheDocument();
  });
});

describe('PaletteEditor', () => {
  it('edits a color and re-runs with palette_override', async () => {
    const user = userEvent.setup();
    const onRerun = vi.fn();
    render(<PaletteEditor palette={MOCK_SOURCE_PALETTE} overrideActive={false} onRerun={onRerun} onResetAuto={() => undefined} />);
    const rerun = screen.getByRole('button', { name: 'Re-run with edited palette' });
    expect(rerun).toBeDisabled();
    const hex = screen.getByRole('textbox', { name: 'Color 2 hex value' });
    await user.clear(hex);
    await user.type(hex, 'zz');
    expect(hex).toHaveAttribute('aria-invalid', 'true');
    expect(rerun).toBeDisabled();
    await user.clear(hex);
    await user.type(hex, '#B22');
    await user.tab();
    expect(hex).toHaveValue('#bb2222');
    await user.click(rerun);
    expect(onRerun).toHaveBeenCalledWith(['#ffffff', '#bb2222', '#268bd2', '#fac81e']);
  });

  it('merges selected colors into the first selected one', async () => {
    const user = userEvent.setup();
    const onRerun = vi.fn();
    render(<PaletteEditor palette={MOCK_SOURCE_PALETTE} overrideActive onRerun={onRerun} onResetAuto={() => undefined} />);
    const merge = screen.getByRole('button', { name: /Merge selected/ });
    expect(merge).toBeDisabled();
    await user.click(screen.getByRole('checkbox', { name: /Select color 2/ }));
    await user.click(screen.getByRole('checkbox', { name: /Select color 4/ }));
    await user.click(screen.getByRole('button', { name: 'Merge selected (2)' }));
    expect(screen.getAllByTestId('palette-entry')).toHaveLength(3);
    await user.click(screen.getByRole('button', { name: 'Re-run with edited palette' }));
    expect(onRerun).toHaveBeenCalledWith(['#ffffff', '#dc322f', '#268bd2']);
  });

  it('removes, adds and undoes; offers automatic palette when an override is active', async () => {
    const user = userEvent.setup();
    const onResetAuto = vi.fn();
    render(<PaletteEditor palette={['#ffffff', '#000000']} overrideActive onRerun={() => undefined} onResetAuto={onResetAuto} />);
    await user.click(screen.getByRole('button', { name: /Remove color 2/ }));
    expect(screen.getByRole('button', { name: /Remove color 1/ })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Add color' }));
    expect(screen.getAllByTestId('palette-entry')).toHaveLength(2);
    await user.click(screen.getByRole('button', { name: 'Undo changes' }));
    expect(screen.getByRole('textbox', { name: 'Color 2 hex value' })).toHaveValue('#000000');
    await user.click(screen.getByRole('button', { name: 'Use automatic palette' }));
    expect(onResetAuto).toHaveBeenCalled();
  });

  it('updates from the color picker', () => {
    const onRerun = vi.fn();
    render(<PaletteEditor palette={['#ffffff']} overrideActive={false} onRerun={onRerun} onResetAuto={() => undefined} />);
    fireEvent.change(screen.getByLabelText('Color 1 picker'), { target: { value: '#123456' } });
    fireEvent.click(screen.getByRole('button', { name: 'Re-run with edited palette' }));
    expect(onRerun).toHaveBeenCalledWith(['#123456']);
  });
});

describe('QualityBadge', () => {
  it('shows pass with icon and text and every check', () => {
    render(<QualityBadge quality={buildQuality(MOCK_SOURCE_PALETTE, 'flat_color', 900, 512, 512)} />);
    const badge = screen.getByTestId('quality-badge');
    expect(badge).toHaveTextContent('All 6 quality checks passed');
    expect(badge.querySelector('svg[aria-hidden="true"]')).not.toBeNull();
    for (const name of ['SSIM', 'Mean ΔE', 'Max ΔE', 'Gap ratio', 'Processing time', 'Valid SVG']) {
      expect(screen.getByRole('rowheader', { name })).toBeInTheDocument();
    }
    expect(screen.getByTestId('stat-nodes')).toHaveTextContent('132');
    expect(screen.getByText('900 B')).toBeInTheDocument();
  });

  it('shows failures as text, not only color', () => {
    render(<QualityBadge quality={buildQuality(['#ffffff'], 'flat_color', 900, 512, 512)} />);
    expect(screen.getByTestId('quality-badge')).toHaveTextContent(/of 6 quality checks failed/);
    expect(within(screen.getByTestId('check-max_delta_e')).getByText('Fail')).toBeInTheDocument();
    expect(within(screen.getByTestId('check-gap_ratio')).getByText('Pass')).toBeInTheDocument();
  });
});

describe('ProgressIndicator', () => {
  it('shows the stage name and an accessible progress bar', () => {
    render(<ProgressIndicator job={job()} label="Converting" />);
    expect(screen.getByTestId('progress-stage')).toHaveTextContent('Quantizing colors');
    const bar = screen.getByRole('progressbar', { name: 'Conversion progress' });
    expect(bar).toHaveAttribute('aria-valuenow', '38');
    expect(bar).toHaveAttribute('aria-valuetext', '38% - Quantizing colors');
    expect(screen.getByText('Classifying image').parentElement).toHaveTextContent('(complete)');
    expect(screen.getByText('Tracing vectors').parentElement).toHaveTextContent('(pending)');
  });

  it('shows Uploading before a job exists and Queued for queued jobs', () => {
    const { rerender } = render(<ProgressIndicator job={null} label="Uploading" />);
    expect(screen.getByTestId('progress-stage')).toHaveTextContent('Uploading');
    rerender(<ProgressIndicator job={job({ status: 'queued', stage: 'upload', progress: 0 })} label="Converting" />);
    expect(screen.getByTestId('progress-stage')).toHaveTextContent('Queued');
  });
});

describe('ApiErrorBanner', () => {
  it.each([
    [413, 'too_large', 'File too large'],
    [415, 'unsupported_media_type', 'Unsupported file type'],
    [422, 'invalid_image', 'Image could not be read'],
  ])('renders HTTP %i as an alert', (status, code, title) => {
    const onDismiss = vi.fn();
    render(<ApiErrorBanner error={new ApiError(status, { code, message: 'details', stage: 'upload' })} onDismiss={onDismiss} />);
    const alert = screen.getByRole('alert');
    expect(alert).toHaveTextContent(title);
    expect(alert).toHaveTextContent(`HTTP ${status}`);
    expect(alert).toHaveTextContent('stage: Uploading');
    fireEvent.click(within(alert).getByRole('button', { name: 'Dismiss' }));
    expect(onDismiss).toHaveBeenCalled();
  });
});

describe('UploadDropzone', () => {
  const limits = { max_upload_bytes: 100, job_ttl_seconds: 3600, max_pixels: 1e6, accepted_media_types: ['image/png', 'image/jpeg'] };

  it('accepts a dropped PNG and rejects other files with a message', () => {
    const onFile = vi.fn();
    const onReject = vi.fn();
    render(<UploadDropzone limits={limits} file={null} onFile={onFile} onReject={onReject} />);
    const zone = screen.getByTestId('dropzone');
    const png = new File(['x'], 'a.png', { type: 'image/png' });
    fireEvent.drop(zone, { dataTransfer: { files: [png], dropEffect: 'copy' } });
    expect(onFile).toHaveBeenCalledWith(png);
    fireEvent.drop(zone, { dataTransfer: { files: [new File(['x'], 'a.gif', { type: 'image/gif' })] } });
    expect(onReject).toHaveBeenLastCalledWith(expect.stringContaining('PNG or JPG'));
    fireEvent.drop(zone, { dataTransfer: { files: [new File(['x'.repeat(101)], 'b.png', { type: 'image/png' })] } });
    expect(onReject).toHaveBeenLastCalledWith(expect.stringContaining('maximum upload size is 100 B'));
    fireEvent.drop(zone, { dataTransfer: { files: [png, png] } });
    expect(onReject).toHaveBeenLastCalledWith('Please use a single image at a time.');
  });

  it('uses the picker via a keyboard-focusable button and shows limits', async () => {
    const user = userEvent.setup();
    const onFile = vi.fn();
    render(<UploadDropzone limits={limits} file={null} onFile={onFile} onReject={() => undefined} />);
    expect(screen.getByRole('button', { name: 'Choose image' })).toHaveAccessibleDescription('PNG or JPG, up to 100 B');
    expect(screen.getByLabelText('Image file')).toHaveAttribute('accept', 'image/png,image/jpeg,.png,.jpg,.jpeg');
    const png = new File(['x'], 'pick.png', { type: 'image/png' });
    await user.upload(screen.getByLabelText('Image file'), png);
    expect(onFile).toHaveBeenCalledWith(png);
  });
});

describe('ComparisonViewer', () => {
  it('renders both images via <img>, with a keyboard slider and synchronized zoom', async () => {
    const user = userEvent.setup();
    const { container } = render(<ComparisonViewer originalUrl="blob:orig" vectorUrl="/api/v1/jobs/j/files/svg" width={400} height={200} />);
    expect(screen.getByAltText('Vector result (SVG)')).toHaveAttribute('src', '/api/v1/jobs/j/files/svg');
    expect(container.querySelector('g')).toBeNull();

    const split = screen.getByRole('slider', { name: /Divider position/ });
    fireEvent.change(split, { target: { value: '30' } });
    expect(screen.getByTestId('vector-clip')).toHaveStyle({ clipPath: 'inset(0 0 0 30%)' });

    const pane = screen.getByTestId('pane-overlay');
    pane.focus();
    await user.keyboard('+');
    expect(screen.getByTestId('zoom-level')).toHaveTextContent('125%');
    await user.keyboard('{ArrowRight}');
    await user.click(screen.getByRole('button', { name: 'Side by side' }));
    const original = screen.getByTestId('pane-original').firstElementChild as HTMLElement;
    const vector = screen.getByTestId('pane-vector').firstElementChild as HTMLElement;
    expect(original.style.transform).toContain('scale(1.25)');
    expect(original.style.transform).toBe(vector.style.transform);

    act(() => {
      screen.getByTestId('pane-vector').dispatchEvent(new WheelEvent('wheel', { deltaY: -100, bubbles: true, cancelable: true }));
    });
    expect(screen.getByTestId('zoom-level')).toHaveTextContent('156%');
    await user.click(screen.getByRole('button', { name: 'Reset view' }));
    expect(screen.getByTestId('zoom-level')).toHaveTextContent('100%');
    expect(screen.getByRole('button', { name: 'Zoom out' })).toBeDisabled();
  });

  it('shows a message when an image cannot be loaded', () => {
    render(<ComparisonViewer originalUrl="blob:orig" vectorUrl="/missing.svg" width={10} height={10} />);
    fireEvent.error(screen.getByAltText('Vector result (SVG)'));
    expect(screen.getByText('Vector result (SVG) could not be loaded.')).toBeInTheDocument();
  });
});

describe('Downloads', () => {
  it('lists result files except the original and downloads via fetch', async () => {
    const user = userEvent.setup();
    const createUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:x');
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined);
    render(
      <Downloads
        sourceFilename="logo.png"
        files={[
          { kind: 'original', url: '/api/v1/jobs/x/files/original', media_type: 'image/png', size_bytes: 10 },
          { kind: 'png', url: '/api/v1/jobs/x/files/png', media_type: 'image/png', size_bytes: 2048 },
          { kind: 'svg', url: '/api/v1/jobs/x/files/svg', media_type: 'image/svg+xml', size_bytes: 100 },
        ]}
      />,
    );
    const buttons = screen.getAllByRole('button');
    expect(buttons.map((b) => b.textContent)).toEqual(['Download SVG100 B', 'Download PNG preview2.0 KB']);
    await user.click(screen.getByRole('button', { name: /Download SVG/ }));
    // Unknown job in the mock -> 404 is surfaced as an alert.
    expect(await screen.findByRole('alert')).toHaveTextContent('SVG: Unknown job');
    expect(click).not.toHaveBeenCalled();
    createUrl.mockRestore();
    click.mockRestore();
  });
});
