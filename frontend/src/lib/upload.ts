/** Client-side upload validation against `GET /api/v1/config` limits. */
import type { Limits } from '../api/types';
import { formatBytes } from './format';

const EXTENSION_TYPES: Record<string, string> = {
  png: 'image/png',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  webp: 'image/webp',
  bmp: 'image/bmp',
  tif: 'image/tiff',
  tiff: 'image/tiff',
};

/**
 * Types the browser can draw in an <img>. The API also accepts TIFF, but the picker leaves it out because
 * the original-vs-vector comparison could not show the uploaded image.
 */
const PREVIEWABLE_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp', 'image/bmp']);

export const DEFAULT_ACCEPTED_TYPES = ['image/png', 'image/jpeg'];

export function acceptedTypes(limits: Limits): string[] {
  return (limits.accepted_media_types ?? DEFAULT_ACCEPTED_TYPES).filter((type) => PREVIEWABLE_TYPES.has(type));
}

/** MIME type of a File, inferred from the extension when the browser leaves it empty. */
export function fileMediaType(file: File): string {
  if (file.type) return file.type;
  const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
  return EXTENSION_TYPES[ext] ?? '';
}

/** `accept` attribute for the file input, e.g. "image/png,image/jpeg,.png,.jpg,.jpeg". */
export function acceptAttribute(limits: Limits): string {
  const types = acceptedTypes(limits);
  const exts = Object.entries(EXTENSION_TYPES)
    .filter(([, type]) => types.includes(type))
    .map(([ext]) => `.${ext}`);
  return [...types, ...exts].join(',');
}

export function describeAccepted(limits: Limits): string {
  const names = acceptedTypes(limits).map((t) => t.replace('image/', '').toUpperCase().replace('JPEG', 'JPG'));
  return names.length > 1 ? `${names.slice(0, -1).join(', ')} or ${names[names.length - 1]}` : names.join('');
}

/** Returns a user-facing error message, or null if the file is acceptable. */
export function validateUpload(file: File, limits: Limits): string | null {
  const type = fileMediaType(file);
  if (!acceptedTypes(limits).includes(type)) {
    return `"${file.name}" is not a supported image. Please choose a ${describeAccepted(limits)} file.`;
  }
  if (file.size > limits.max_upload_bytes) {
    return `"${file.name}" is ${formatBytes(file.size)}; the maximum upload size is ${formatBytes(limits.max_upload_bytes)}.`;
  }
  if (file.size === 0) {
    return `"${file.name}" is empty.`;
  }
  return null;
}
