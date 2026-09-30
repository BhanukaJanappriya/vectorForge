/**
 * Runtime lists of the enum values in the OpenAPI spec (for rendering form options and
 * walking stages). `exhaustive()` makes `tsc` fail if the generated union gains or loses
 * a member that is not reflected here, so these lists cannot drift from the spec.
 */
import type {
  DetailLevel,
  FileKind,
  ImageClassLabel,
  LineMode,
  MetricName,
  OutputFormat,
  PipelineStage,
  ProcessingMode,
} from './types';

type Exhaustive<U extends string, L extends readonly U[]> = [U] extends [L[number]] ? L : never;

function exhaustive<U extends string>() {
  return <const L extends readonly U[]>(list: Exhaustive<U, L>): L => list;
}

export const PROCESSING_MODES = exhaustive<ProcessingMode>()(['auto', 'line_art', 'flat_color', 'mixed']);
export const DETAIL_LEVELS = exhaustive<DetailLevel>()(['low', 'medium', 'high']);
export const LINE_MODES = exhaustive<LineMode>()(['outline', 'centerline']);
export const OUTPUT_FORMATS = exhaustive<OutputFormat>()(['svg', 'ai', 'eps', 'png']);
export const FILE_KINDS = exhaustive<FileKind>()(['original', 'svg', 'ai', 'eps', 'png']);
export const IMAGE_CLASS_LABELS = exhaustive<ImageClassLabel>()(['line_art', 'flat_color', 'mixed']);
export const METRIC_NAMES = exhaustive<MetricName>()([
  'ssim',
  'max_delta_e',
  'mean_delta_e',
  'gap_ratio',
  'alpha_iou',
  'processing_time_s',
  'svg_valid',
]);
/** Pipeline order, as in CLAUDE.md: preprocess -> classify -> quantize -> lines -> vectorize -> assemble -> export -> evaluate. */
export const PIPELINE_STAGES = exhaustive<PipelineStage>()([
  'upload',
  'preprocess',
  'classify',
  'quantize',
  'extract_lines',
  'vectorize',
  'assemble',
  'export',
  'evaluate',
  'done',
]);
