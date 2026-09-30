/** MSW request handlers implementing api/openapi.yaml on top of MockJobStore. */
import { http, HttpResponse, type HttpHandler } from 'msw';
import { FILE_KINDS } from '../api/enums';
import type { ErrorResponse, FileKind, HealthResponse, PipelineStage, RerunRequest } from '../api/types';
import { sniffImage } from './image';
import { MOCK_CONFIG, SettingsError, parseSettings, type MockJobStore } from './store';

const API = '/api/v1';

function errorResponse(status: number, code: string, message: string, stage: PipelineStage | null = null) {
  const body: ErrorResponse = { error: { code, message, stage } };
  return HttpResponse.json(body, { status });
}

/** Multipart file parts; checked structurally because File classes differ between realms (jsdom vs. Node). */
function isFilePart(value: FormDataEntryValue | null): value is File {
  return value !== null && typeof value !== 'string' && typeof value.arrayBuffer === 'function';
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new SettingsError('settings: invalid JSON');
  }
}

export function createHandlers(store: MockJobStore): HttpHandler[] {
  return [
    http.get(`${API}/health`, () => {
      const body: HealthResponse = {
        status: 'ok',
        version: '1.0.0-mock',
        schema_version: '1.0.0',
        capabilities: { inkscape: true, vtracer: true, potrace: true },
      };
      return HttpResponse.json(body);
    }),

    http.get(`${API}/config`, () => HttpResponse.json(MOCK_CONFIG)),

    http.post(`${API}/convert`, async ({ request }) => {
      let form: FormData;
      try {
        form = await request.formData();
      } catch {
        return errorResponse(422, 'invalid_request', 'Expected a multipart/form-data body.');
      }
      const file = form.get('file');
      if (!isFilePart(file)) return errorResponse(422, 'invalid_request', 'Missing "file" field.');
      if (file.size > MOCK_CONFIG.limits.max_upload_bytes) {
        return errorResponse(413, 'too_large', `File exceeds ${MOCK_CONFIG.limits.max_upload_bytes} bytes.`, 'upload');
      }
      const declared = file.type;
      const accepted = MOCK_CONFIG.limits.accepted_media_types ?? [];
      if (declared && !accepted.includes(declared)) {
        return errorResponse(415, 'unsupported_media_type', `Unsupported media type ${declared}; use PNG or JPEG.`, 'upload');
      }
      const bytes = new Uint8Array(await file.arrayBuffer());
      const image = sniffImage(bytes);
      if (!image) {
        return errorResponse(422, 'invalid_image', `Could not decode "${file.name}" as a PNG or JPEG image.`, 'upload');
      }
      if (image.width * image.height > MOCK_CONFIG.limits.max_pixels) {
        return errorResponse(413, 'too_large', 'Image has too many pixels.', 'upload');
      }
      let settings;
      try {
        const raw = form.get('settings');
        const text = isFilePart(raw) ? await raw.text() : raw;
        settings = parseSettings(text === null || text === '' ? undefined : parseJson(text));
      } catch (err) {
        if (err instanceof SettingsError) return errorResponse(422, 'invalid_settings', err.message);
        throw err;
      }
      const job = store.create({ bytes, image, filename: file.name }, settings);
      return HttpResponse.json(job, { status: 202 });
    }),

    http.post(`${API}/jobs/:jobId/rerun`, async ({ request, params }) => {
      let settings;
      try {
        const body = parseJson(await request.text()) as Partial<RerunRequest> | null;
        if (!body || typeof body !== 'object' || !('settings' in body)) {
          throw new SettingsError('body.settings: field required');
        }
        settings = parseSettings(body.settings);
      } catch (err) {
        if (err instanceof SettingsError) return errorResponse(422, 'invalid_settings', err.message);
        throw err;
      }
      const job = store.rerun(String(params.jobId), settings);
      if (!job) return errorResponse(404, 'not_found', 'Unknown or expired job.');
      return HttpResponse.json(job, { status: 202 });
    }),

    http.get(`${API}/jobs/:jobId`, ({ params }) => {
      const job = store.poll(String(params.jobId));
      return job ? HttpResponse.json(job) : errorResponse(404, 'not_found', 'Unknown job.');
    }),

    http.delete(`${API}/jobs/:jobId`, ({ params }) =>
      store.delete(String(params.jobId))
        ? new HttpResponse(null, { status: 204 })
        : errorResponse(404, 'not_found', 'Unknown job.'),
    ),

    http.get(`${API}/jobs/:jobId/files/:kind`, ({ params, request }) => {
      const kind = String(params.kind);
      if (!(FILE_KINDS as readonly string[]).includes(kind)) {
        return errorResponse(422, 'invalid_request', `Unknown file kind "${kind}".`);
      }
      const file = store.file(String(params.jobId), kind as FileKind);
      if (!file) return errorResponse(404, 'not_found', 'Unknown job, or file not produced / not ready.');
      const headers: Record<string, string> = { 'Content-Type': file.contentType };
      if (new URL(request.url).searchParams.get('download') === 'true') {
        headers['Content-Disposition'] = `attachment; filename="output.${kind}"`;
      }
      const body = typeof file.body === 'string' ? file.body : file.body.slice().buffer;
      return new HttpResponse(body, { status: 200, headers });
    }),
  ];
}
