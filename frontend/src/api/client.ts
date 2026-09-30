/**
 * Thin typed fetch wrapper for the VectorForge HTTP API (`/api/v1`, same origin).
 * Request/response shapes come from the generated OpenAPI types.
 */
import type {
  ConfigResponse,
  ConvertBody,
  ErrorDetail,
  JobFile,
  JobResponse,
  RerunRequest,
  Settings,
} from './types';

export const API_PREFIX = '/api/v1';

/** Error for any non-2xx response. `detail` is the spec's `ErrorResponse.error` when the body had one. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: ErrorDetail;

  constructor(status: number, detail: ErrorDetail) {
    super(detail.message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

/** Extracts an `ErrorDetail` from an `ErrorResponse` body; falls back to a generic one. */
export function parseErrorBody(status: number, body: unknown, fallbackText = ''): ErrorDetail {
  if (isRecord(body) && isRecord(body.error)) {
    const { code, message, stage } = body.error;
    if (typeof code === 'string' && typeof message === 'string') {
      return { code, message, stage: (stage ?? null) as ErrorDetail['stage'] };
    }
  }
  // FastAPI's default validation shape ({"detail": ...}) as a courtesy fallback.
  if (isRecord(body) && 'detail' in body) {
    const detail = body.detail;
    const message = typeof detail === 'string' ? detail : JSON.stringify(detail);
    return { code: status === 422 ? 'invalid_request' : `http_${status}`, message, stage: null };
  }
  return {
    code: `http_${status}`,
    message: fallbackText.trim() || `Request failed with HTTP ${status}.`,
    stage: null,
  };
}

async function request<T>(input: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(input, init);
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err;
    throw new ApiError(0, {
      code: 'network_error',
      message: 'Could not reach the VectorForge server. Check your connection and try again.',
      stage: null,
    });
  }
  const text = await response.text();
  let body: unknown = undefined;
  if (text) {
    try {
      body = JSON.parse(text) as unknown;
    } catch {
      body = undefined;
    }
  }
  if (!response.ok) {
    throw new ApiError(response.status, parseErrorBody(response.status, body, body === undefined ? text : ''));
  }
  return body as T;
}

export function getConfig(signal?: AbortSignal): Promise<ConfigResponse> {
  return request<ConfigResponse>(`${API_PREFIX}/config`, { signal });
}

/** Builds the multipart body described by `ConvertBody` (settings is JSON-encoded). */
export function buildConvertForm(file: File, settings: Settings): FormData {
  const fields: Omit<ConvertBody, 'file'> = { settings: JSON.stringify(settings) };
  const form = new FormData();
  form.append('file', file, file.name);
  if (fields.settings !== undefined) form.append('settings', fields.settings);
  return form;
}

export function convert(file: File, settings: Settings, signal?: AbortSignal): Promise<JobResponse> {
  return request<JobResponse>(`${API_PREFIX}/convert`, {
    method: 'POST',
    body: buildConvertForm(file, settings),
    signal,
  });
}

export function getJob(jobId: string, signal?: AbortSignal): Promise<JobResponse> {
  return request<JobResponse>(`${API_PREFIX}/jobs/${encodeURIComponent(jobId)}`, { signal });
}

export function rerunJob(jobId: string, settings: Settings, signal?: AbortSignal): Promise<JobResponse> {
  const body: RerunRequest = { settings };
  return request<JobResponse>(`${API_PREFIX}/jobs/${encodeURIComponent(jobId)}/rerun`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
}

/** Fetches a result file as a Blob (used for downloads so errors such as an expired job can be shown). */
export async function fetchJobFile(file: JobFile, signal?: AbortSignal): Promise<Blob> {
  let response: Response;
  try {
    response = await fetch(file.url, { signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err;
    throw new ApiError(0, { code: 'network_error', message: 'Download failed: server unreachable.', stage: null });
  }
  if (!response.ok) {
    const text = await response.text();
    let body: unknown = undefined;
    try {
      body = JSON.parse(text) as unknown;
    } catch {
      body = undefined;
    }
    throw new ApiError(response.status, parseErrorBody(response.status, body, body === undefined ? text : ''));
  }
  return response.blob();
}
