import type { ApiError } from '../api/client';
import type { ErrorDetail } from '../api/types';
import { errorTitle } from '../lib/errors';
import { STAGE_LABELS } from '../lib/format';
import { AlertIcon } from './Icons';

interface ErrorBannerProps {
  title: string;
  message: string;
  detail?: ErrorDetail | null;
  status?: number;
  onDismiss?: () => void;
  onRetry?: () => void;
}

export function ErrorBanner({ title, message, detail, status, onDismiss, onRetry }: ErrorBannerProps) {
  const stage = detail?.stage ? STAGE_LABELS[detail.stage] : null;
  const meta = [status ? `HTTP ${status}` : null, detail?.code ? `code: ${detail.code}` : null, stage ? `stage: ${stage}` : null]
    .filter(Boolean)
    .join(' · ');
  return (
    <div role="alert" className="flex gap-3 rounded-lg border border-red-300 bg-red-50 p-4 text-red-900" data-testid="error-banner">
      <AlertIcon className="mt-0.5 h-5 w-5 text-red-600" />
      <div className="min-w-0 flex-1">
        <p className="font-semibold">{title}</p>
        <p className="mt-1 text-sm break-words">{message}</p>
        {meta && <p className="mt-1 text-xs break-words text-red-800">{meta}</p>}
        {(onRetry ?? onDismiss) && (
          <div className="mt-3 flex flex-wrap gap-2">
            {onRetry && (
              <button type="button" onClick={onRetry} className="btn btn-secondary">
                Try again
              </button>
            )}
            {onDismiss && (
              <button type="button" onClick={onDismiss} className="btn btn-secondary">
                Dismiss
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

interface ApiErrorBannerProps {
  error: ApiError;
  onDismiss?: () => void;
  onRetry?: () => void;
}

export function ApiErrorBanner({ error, onDismiss, onRetry }: ApiErrorBannerProps) {
  return (
    <ErrorBanner
      title={errorTitle(error.status, error.detail)}
      message={error.detail.message}
      detail={error.detail}
      status={error.status || undefined}
      onDismiss={onDismiss}
      onRetry={onRetry}
    />
  );
}
