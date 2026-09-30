import { useCallback, useEffect, useRef, useState } from 'react';
import { DEFAULT_POLL_OPTIONS, type PollOptions } from './api/poll';
import type { JobResponse, Settings } from './api/types';
import { ComparisonViewer } from './components/ComparisonViewer';
import { Downloads } from './components/Downloads';
import { ApiErrorBanner, ErrorBanner } from './components/ErrorBanner';
import { PaletteEditor } from './components/PaletteEditor';
import { ProgressIndicator } from './components/ProgressIndicator';
import { QualityBadge } from './components/QualityBadge';
import { ResultSummary } from './components/ResultSummary';
import { SettingsPanel } from './components/SettingsPanel';
import { UploadDropzone } from './components/UploadDropzone';
import { useConfig } from './hooks/useConfig';
import { useJobRunner } from './hooks/useJobRunner';
import { STAGE_LABELS } from './lib/format';

interface AppProps {
  /** Poll backoff; tests pass shorter delays. */
  pollOptions?: PollOptions;
}

function Header() {
  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-7xl flex-wrap items-baseline gap-x-3 gap-y-1 px-4 py-4">
        <h1 className="text-xl font-bold tracking-tight text-slate-900">VectorForge</h1>
        <p className="text-sm text-slate-600">Turn PNG and JPG images into clean SVG, AI and EPS vectors.</p>
      </div>
    </header>
  );
}

function FailedJob({ job }: { job: JobResponse }) {
  const stage = job.error?.stage ?? job.stage;
  return (
    <ErrorBanner
      title="Conversion failed"
      message={job.error?.message ?? `The job failed during ${STAGE_LABELS[stage]}.`}
      detail={job.error ?? { code: 'job_failed', message: '', stage }}
    />
  );
}

function ResultView({
  job,
  originalUrl,
  busy,
  onPaletteRerun,
  onResetPalette,
}: {
  job: JobResponse;
  originalUrl: string | null;
  busy: boolean;
  onPaletteRerun: (palette: string[]) => void;
  onResetPalette: () => void;
}) {
  const result = job.result;
  if (!result) return null;
  const svg = result.files.find((f) => f.kind === 'svg');
  const original = originalUrl ?? result.files.find((f) => f.kind === 'original')?.url ?? null;
  return (
    <div className={`space-y-6 ${busy ? 'opacity-60' : ''}`} aria-busy={busy} data-testid="result">
      <ResultSummary job={job} result={result} />
      {svg && original ? (
        <ComparisonViewer originalUrl={original} vectorUrl={svg.url} width={result.width} height={result.height} />
      ) : (
        <p className="panel text-sm text-slate-700">No SVG preview is available for this job.</p>
      )}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <QualityBadge quality={result.quality} />
        <PaletteEditor
          key={job.job_id}
          palette={result.palette_hex}
          overrideActive={job.settings.palette_override !== null}
          busy={busy}
          onRerun={onPaletteRerun}
          onResetAuto={onResetPalette}
        />
      </div>
      <Downloads files={result.files} sourceFilename={job.filename} />
    </div>
  );
}

export default function App({ pollOptions = DEFAULT_POLL_OPTIONS }: AppProps) {
  const [configState, retryConfig] = useConfig();
  const runner = useJobRunner(pollOptions);
  const [settingsDraft, setSettingsDraft] = useState<Settings | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [clientError, setClientError] = useState<string | null>(null);
  const fileUrlRef = useRef<string | null>(null);

  useEffect(() => () => {
    if (fileUrlRef.current) URL.revokeObjectURL(fileUrlRef.current);
  }, []);

  const { state, error, startConvert, startRerun, reset, dismissError } = runner;
  const busy = state.phase === 'submitting' || state.phase === 'polling';
  const finished = state.phase === 'finished' ? state.job : null;
  const shownJob = finished ?? (state.phase === 'submitting' || state.phase === 'polling' ? state.previous : null);

  const announcement =
    state.phase === 'finished' ? (state.job.status === 'succeeded' ? 'Conversion complete.' : 'Conversion failed.') : '';

  const onFile = useCallback(
    (picked: File) => {
      reset();
      setClientError(null);
      if (fileUrlRef.current) URL.revokeObjectURL(fileUrlRef.current);
      const url = URL.createObjectURL(picked);
      fileUrlRef.current = url;
      setFileUrl(url);
      setFile(picked);
    },
    [reset],
  );

  const onReject = useCallback((message: string) => setClientError(message), []);

  if (configState.status !== 'ready') {
    return (
      <>
        <Header />
        <main className="mx-auto max-w-7xl px-4 py-6">
          {configState.status === 'loading' ? (
            <p className="text-sm text-slate-700" role="status">
              Loading settings…
            </p>
          ) : (
            <ApiErrorBanner error={configState.error} onRetry={retryConfig} />
          )}
        </main>
      </>
    );
  }

  const { config } = configState;
  const settings = settingsDraft ?? config.defaults;
  const canRerunCurrent = finished !== null && file !== null;

  const submit = () => {
    if (!file) return;
    setClientError(null);
    if (canRerunCurrent) {
      startRerun(finished.job_id, { ...settings, palette_override: finished.settings.palette_override });
    } else {
      startConvert(file, { ...settings, palette_override: null });
    }
  };

  const startOver = () => {
    reset();
    setClientError(null);
    setFile(null);
    if (fileUrlRef.current) URL.revokeObjectURL(fileUrlRef.current);
    fileUrlRef.current = null;
    setFileUrl(null);
  };

  return (
    <>
      <Header />
      <main className="mx-auto grid max-w-7xl grid-cols-1 gap-6 px-4 py-6 lg:grid-cols-[340px_minmax(0,1fr)]">
        <div className="min-w-0 space-y-6">
          <UploadDropzone limits={config.limits} file={file} disabled={busy} onFile={onFile} onReject={onReject} />
          <SettingsPanel
            value={settings}
            onChange={setSettingsDraft}
            disabled={busy}
            paletteOverrideCount={finished?.settings.palette_override?.length ?? null}
          />
          <div className="flex flex-wrap gap-2">
            <button type="button" className="btn btn-primary flex-1" onClick={submit} disabled={!file || busy}>
              {canRerunCurrent ? 'Re-run with these settings' : 'Convert'}
            </button>
            {(file ?? shownJob) && (
              <button type="button" className="btn btn-secondary" onClick={startOver} disabled={busy}>
                Start over
              </button>
            )}
          </div>
        </div>

        <div className="min-w-0 space-y-6" aria-live="off">
          <p className="sr-only" role="status" data-testid="announcer">
            {announcement}
          </p>
          {clientError && (
            <ErrorBanner title="File not accepted" message={clientError} onDismiss={() => setClientError(null)} />
          )}
          {error && <ApiErrorBanner error={error} onDismiss={dismissError} />}
          {state.phase === 'submitting' && (
            <ProgressIndicator job={null} label={state.kind === 'rerun' ? 'Re-running' : 'Uploading'} />
          )}
          {state.phase === 'polling' && (
            <ProgressIndicator job={state.job} label={state.job.source_job_id ? 'Re-running' : 'Converting'} />
          )}
          {finished?.status === 'failed' && <FailedJob job={finished} />}
          {shownJob?.status === 'succeeded' && (
            <ResultView
              job={shownJob}
              originalUrl={fileUrl}
              busy={busy}
              onPaletteRerun={(palette) => startRerun(shownJob.job_id, { ...shownJob.settings, palette_override: palette })}
              onResetPalette={() => startRerun(shownJob.job_id, { ...shownJob.settings, palette_override: null })}
            />
          )}
          {state.phase === 'idle' && !error && !clientError && (
            <div className="panel text-sm text-slate-700" data-testid="empty-state">
              <h2 className="panel-title">How it works</h2>
              <ol className="list-decimal space-y-1 pl-5">
                <li>Upload a PNG or JPG (drag and drop or choose a file).</li>
                <li>Adjust the settings, or keep the automatic defaults.</li>
                <li>Convert, compare original and vector, tweak the palette, and download SVG, AI, EPS or PNG.</li>
              </ol>
            </div>
          )}
        </div>
      </main>
    </>
  );
}
