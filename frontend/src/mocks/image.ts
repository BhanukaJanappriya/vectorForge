/** Minimal PNG/JPEG header sniffing for the mock API (validates the upload and reads its size). */

export interface SniffedImage {
  mediaType: 'image/png' | 'image/jpeg';
  width: number;
  height: number;
}

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

function readPng(bytes: Uint8Array): SniffedImage | null {
  if (bytes.length < 24 || !PNG_SIGNATURE.every((b, i) => bytes[i] === b)) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const chunkType = String.fromCharCode(...bytes.subarray(12, 16));
  if (chunkType !== 'IHDR') return null;
  const width = view.getUint32(16);
  const height = view.getUint32(20);
  return width > 0 && height > 0 ? { mediaType: 'image/png', width, height } : null;
}

function readJpeg(bytes: Uint8Array): SniffedImage | null {
  if (bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 2;
  while (offset + 9 < bytes.length) {
    if (bytes[offset] !== 0xff) return null;
    const marker = bytes[offset + 1] ?? 0;
    if (marker === 0xff) {
      offset += 1;
      continue;
    }
    const length = view.getUint16(offset + 2);
    const isStartOfFrame = marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc;
    if (isStartOfFrame) {
      const height = view.getUint16(offset + 5);
      const width = view.getUint16(offset + 7);
      return width > 0 && height > 0 ? { mediaType: 'image/jpeg', width, height } : null;
    }
    offset += 2 + length;
  }
  return null;
}

/** Returns the image type and dimensions, or null when the bytes are not a decodable PNG/JPEG header. */
export function sniffImage(bytes: Uint8Array): SniffedImage | null {
  return readPng(bytes) ?? readJpeg(bytes);
}
