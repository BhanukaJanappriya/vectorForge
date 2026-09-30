import type { JobResponse, JobResult } from '../api/types';
import { IMAGE_CLASS_LABELS } from '../lib/format';

interface ResultSummaryProps {
  job: JobResponse;
  result: JobResult;
}

/** Short facts about the finished job: detected class, size, layers, warnings, expiry. */
export function ResultSummary({ job, result }: ResultSummaryProps) {
  const cls = result.image_class;
  const expires = new Date(job.expires_at);
  return (
    <section aria-labelledby="result-heading" className="panel" data-testid="result-summary">
      <h2 id="result-heading" className="panel-title">
        Result
      </h2>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm sm:grid-cols-4">
        <div>
          <dt className="text-slate-600">Detected type</dt>
          <dd className="font-medium" data-testid="image-class">
            {IMAGE_CLASS_LABELS[cls.label]}
            <span className="font-normal text-slate-600">
              {cls.forced ? ' (forced)' : ` (${Math.round(cls.confidence * 100)}%)`}
            </span>
          </dd>
        </div>
        <div>
          <dt className="text-slate-600">Size</dt>
          <dd className="font-medium tabular-nums">
            {result.width} × {result.height} px
          </dd>
        </div>
        <div>
          <dt className="text-slate-600">Layers</dt>
          <dd className="font-medium tabular-nums">{result.layer_count}</dd>
        </div>
        <div>
          <dt className="text-slate-600">Available until</dt>
          <dd className="font-medium tabular-nums">
            <time dateTime={job.expires_at}>
              {expires.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </time>
          </dd>
        </div>
      </dl>
      {job.source_job_id && (
        <p className="mt-3 text-sm text-slate-700" data-testid="rerun-note">
          Re-run of an earlier conversion{job.settings.palette_override ? ' with a custom palette' : ''}.
        </p>
      )}
      {(result.warnings ?? []).length > 0 && (
        <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-amber-900">
          {(result.warnings ?? []).map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}
    </section>
  );
}
