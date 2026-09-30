/** Test helpers: tiny image files with valid headers (enough for the mock's header sniffing). */

/** A PNG signature + IHDR chunk with the given size. `extraBytes` pads the file (for size-limit tests). */
export function makePng(name: string, width = 512, height = 512, extraBytes = 0): File {
  const bytes = new Uint8Array(33 + extraBytes);
  bytes.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a], 0);
  const view = new DataView(bytes.buffer);
  view.setUint32(8, 13);
  bytes.set([0x49, 0x48, 0x44, 0x52], 12); // "IHDR"
  view.setUint32(16, width);
  view.setUint32(20, height);
  bytes.set([8, 6, 0, 0, 0], 24);
  return new File([bytes], name, { type: 'image/png' });
}

/** A JPEG SOI + APP0 + SOF0 header with the given size. */
export function makeJpeg(name: string, width = 64, height = 48): File {
  const bytes = new Uint8Array([
    0xff, 0xd8, 0xff, 0xe0, 0x00, 0x04, 0x00, 0x00, 0xff, 0xc0, 0x00, 0x11, 0x08, height >> 8, height & 0xff, width >> 8,
    width & 0xff, 0x03, 0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xff, 0xd9,
  ]);
  return new File([bytes], name, { type: 'image/jpeg' });
}

/** Bytes of a File (jsdom's File lacks arrayBuffer in some versions). */
export function fileBytes(file: File): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(new Uint8Array(reader.result as ArrayBuffer));
    reader.onerror = () => reject(reader.error ?? new Error('read failed'));
    reader.readAsArrayBuffer(file);
  });
}
