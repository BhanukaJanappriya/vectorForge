/** Human-readable labels and number formatting for API values. */
import type { FileKind, ImageClassLabel, MetricName, PipelineStage, ProcessingMode } from '../api/types';

export const STAGE_LABELS: Record<PipelineStage, string> = {
  upload: 'Uploading',
  preprocess: 'Preprocessing',
  classify: 'Classifying image',
  quantize: 'Quantizing colors',
  extract_lines: 'Extracting lines',
  vectorize: 'Tracing vectors',
  assemble: 'Assembling SVG',
  export: 'Exporting files',
  evaluate: 'Evaluating quality',
  done: 'Done',
};

export const MODE_LABELS: Record<ProcessingMode, string> = {
  auto: 'Auto-detect',
  line_art: 'Line art',
  flat_color: 'Flat color',
  mixed: 'Mixed',
};

export const IMAGE_CLASS_LABELS: Record<ImageClassLabel, string> = {
  line_art: 'Line art',
  flat_color: 'Flat color',
  mixed: 'Mixed',
};

export const METRIC_LABELS: Record<MetricName, string> = {
  ssim: 'SSIM',
  mean_delta_e: 'Mean ΔE',
  max_delta_e: 'Max ΔE',
  gap_ratio: 'Gap ratio',
  alpha_iou: 'Alpha IoU',
  processing_time_s: 'Processing time',
  svg_valid: 'Valid SVG',
};

export const FILE_KIND_LABELS: Record<FileKind, string> = {
  original: 'Original',
  svg: 'SVG',
  ai: 'AI (Illustrator)',
  eps: 'EPS',
  png: 'PNG preview',
};

export const FILE_EXTENSIONS: Record<FileKind, string> = {
  original: '',
  svg: 'svg',
  ai: 'ai',
  eps: 'eps',
  png: 'png',
};

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(kb < 10 ? 1 : 0)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(mb < 10 ? 1 : 0)} MB`;
}

/** Formats a metric value with a precision that suits the metric. */
export function formatMetric(name: MetricName, value: number): string {
  switch (name) {
    case 'ssim':
    case 'alpha_iou':
      return value.toFixed(3);
    case 'mean_delta_e':
    case 'max_delta_e':
      return value.toFixed(2);
    case 'gap_ratio':
      return value.toFixed(5);
    case 'processing_time_s':
      return `${value.toFixed(2)} s`;
    case 'svg_valid':
      return value >= 1 ? 'yes' : 'no';
  }
}

/** Removes the extension from a filename (for naming downloads). */
export function baseName(filename: string): string {
  const dot = filename.lastIndexOf('.');
  return dot >= 0 ? filename.slice(0, dot) : filename;
}

/** Download filename for a result file, e.g. "logo.svg" or "logo_preview.png". */
export function downloadName(sourceFilename: string, kind: FileKind): string {
  const base = baseName(sourceFilename) || 'output';
  return kind === 'png' ? `${base}_preview.png` : `${base}.${FILE_EXTENSIONS[kind]}`;
}
