import { useId } from 'react';
import type { MetricCheck, QualityReport } from '../api/types';
import { METRIC_LABELS, formatBytes, formatMetric } from '../lib/format';
import { CheckIcon, CrossIcon } from './Icons';

function PassFail({ passed, compact = false }: { passed: boolean; compact?: boolean }) {
  return (
    <span
      className={`inline-flex items-center gap-1 font-medium ${passed ? 'text-emerald-800' : 'text-red-800'} ${compact ? 'text-sm' : ''}`}
    >
      {passed ? <CheckIcon /> : <CrossIcon />}
      {passed ? 'Pass' : 'Fail'}
    </span>
  );
}

function thresholdText(check: MetricCheck): string {
  if (check.name === 'svg_valid') return 'required';
  return `${check.comparator} ${formatMetric(check.name, check.threshold)}`;
}

/** Overall pass/fail badge plus the per-check table and summary stats. */
export function QualityBadge({ quality }: { quality: QualityReport }) {
  const id = useId();
  const failed = quality.checks.filter((c) => !c.passed).length;
  const total = quality.checks.length;

  return (
    <section aria-labelledby={`${id}-heading`} className="panel" data-testid="quality">
      <h2 id={`${id}-heading`} className="panel-title">
        Quality
      </h2>
      <p
        data-testid="quality-badge"
        className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm font-semibold ${
          quality.passed ? 'bg-emerald-100 text-emerald-900' : 'bg-red-100 text-red-900'
        }`}
      >
        {quality.passed ? <CheckIcon className="h-5 w-5" /> : <CrossIcon className="h-5 w-5" />}
        {quality.passed ? `All ${total} quality checks passed` : `${failed} of ${total} quality checks failed`}
      </p>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-left text-sm">
          <caption className="sr-only">Quality checks</caption>
          <thead className="text-xs text-slate-600 uppercase">
            <tr>
              <th scope="col" className="py-1 pr-2 font-medium">
                Check
              </th>
              <th scope="col" className="py-1 pr-2 font-medium">
                Value
              </th>
              <th scope="col" className="py-1 pr-2 font-medium">
                Target
              </th>
              <th scope="col" className="py-1 font-medium">
                Result
              </th>
            </tr>
          </thead>
          <tbody>
            {quality.checks.map((check) => (
              <tr key={check.name} className="border-t border-slate-200" data-testid={`check-${check.name}`}>
                <th scope="row" className="py-1.5 pr-2 font-medium">
                  {METRIC_LABELS[check.name]}
                </th>
                <td className="py-1.5 pr-2 tabular-nums">{formatMetric(check.name, check.value)}</td>
                <td className="py-1.5 pr-2 whitespace-nowrap tabular-nums">{thresholdText(check)}</td>
                <td className="py-1.5">
                  <PassFail passed={check.passed} compact />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 text-sm sm:grid-cols-3">
        <div>
          <dt className="text-slate-600">SSIM</dt>
          <dd className="font-medium tabular-nums">{formatMetric('ssim', quality.ssim)}</dd>
        </div>
        <div>
          <dt className="text-slate-600">Mean / max ΔE</dt>
          <dd className="font-medium tabular-nums">
            {formatMetric('mean_delta_e', quality.mean_delta_e)} / {formatMetric('max_delta_e', quality.max_delta_e)}
          </dd>
        </div>
        <div>
          <dt className="text-slate-600">Gap ratio</dt>
          <dd className="font-medium tabular-nums">{formatMetric('gap_ratio', quality.gap_ratio)}</dd>
        </div>
        {quality.alpha_iou !== null && (
          <div>
            <dt className="text-slate-600">Alpha IoU</dt>
            <dd className="font-medium tabular-nums">{formatMetric('alpha_iou', quality.alpha_iou)}</dd>
          </div>
        )}
        <div>
          <dt className="text-slate-600">Nodes</dt>
          <dd className="font-medium tabular-nums" data-testid="stat-nodes">
            {quality.node_count.toLocaleString('en-US')}
          </dd>
        </div>
        <div>
          <dt className="text-slate-600">SVG size</dt>
          <dd className="font-medium tabular-nums">{formatBytes(quality.file_size_bytes)}</dd>
        </div>
        <div>
          <dt className="text-slate-600">Time</dt>
          <dd className="font-medium tabular-nums">{formatMetric('processing_time_s', quality.processing_time_s)}</dd>
        </div>
      </dl>
    </section>
  );
}
