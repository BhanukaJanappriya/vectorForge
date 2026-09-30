import type { ErrorDetail } from '../api/types';

/** Short title for an HTTP error, following the status codes documented in the spec. */
export function errorTitle(status: number, detail: ErrorDetail): string {
  switch (status) {
    case 0:
      return 'Connection problem';
    case 404:
      return 'Job not found or expired';
    case 413:
      return 'File too large';
    case 415:
      return 'Unsupported file type';
    case 422:
      if (detail.code === 'invalid_image') return 'Image could not be read';
      if (detail.code === 'invalid_settings') return 'Invalid settings';
      return 'Request rejected';
    default:
      return status >= 500 ? 'Server error' : 'Something went wrong';
  }
}
