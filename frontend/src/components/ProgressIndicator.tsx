import { PIPELINE_STAGES } from '../api/enums';
import type { JobResponse } from '../api/types';
import { STAGE_LABELS } from '../lib/format';
import { CheckIcon, Spinner } from './Icons';

interface ProgressIndicatorProps {
  /** Null while the upload / re-run request itself is in flight. */
  job: JobResponse | null;
  label: string;
}

/** Progress bar + current stage name + stage checklist for a running job. */
export function ProgressIndicator({ job, label }: ProgressIndicatorProps) {
  const stage = job?.stage ?? 'upload';
  const percent = Math.round((job?.progress ?? 0) * 100);
  const currentIndex = PIPELINE_STAGES.indexOf(stage);
  const stageText = job ? STAGE_LABELS[stage] : 'Uploading';
  const status = job?.status === 'queued' ? 'Queued' : stageText;

  return (
    <section aria-labelledby="progress-heading" className="panel" data-testid="progress">
      <h2 id="progress-heading" className="panel-title flex items-center gap-2">
        <Spinner className="h-4 w-4 text-sky-700" />
        {label}
      </h2>
      <p className="text-sm text-slate-700" aria-live="polite" aria-atomic="true">
        Stage: <span className="font-semibold" data-testid="progress-stage">{status}</span>
      </p>
      <div
        role="progressbar"
        aria-label="Conversion progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        aria-valuetext={`${percent}% - ${status}`}
        className="mt-3 h-2.5 w-full overflow-hidden rounded-full bg-slate-200"
      >
        <div
          className="h-full rounded-full bg-sky-600 transition-[width] duration-300 motion-reduce:transition-none"
          style={{ width: `${Math.max(percent, 3)}%` }}
        />
      </div>
      <ol className="mt-4 grid grid-cols-1 gap-1 text-sm sm:grid-cols-2" aria-label="Pipeline stages">
        {PIPELINE_STAGES.filter((s) => s !== 'done').map((s, i) => {
          const done = job !== null && (i < currentIndex || job.status === 'succeeded');
          const current = i === currentIndex && !done;
          return (
            <li
              key={s}
              className={`flex items-center gap-2 ${done ? 'text-slate-700' : current ? 'font-semibold text-sky-800' : 'text-slate-500'}`}
            >
              {done ? (
                <CheckIcon className="h-4 w-4 text-emerald-700" />
              ) : current ? (
                <Spinner className="h-4 w-4" />
              ) : (
                <span className="inline-block h-4 w-4 rounded-full border border-slate-400" aria-hidden="true" />
              )}
              <span>{STAGE_LABELS[s]}</span>
              <span className="sr-only">{done ? '(complete)' : current ? '(in progress)' : '(pending)'}</span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
