/**
 * Aliases over the GENERATED OpenAPI types (src/api/schema.d.ts, `npm run gen:api`).
 * Never declare API shapes by hand here: only re-export and derive from `components`.
 */
import type { components, operations } from './schema';

type Schemas = components['schemas'];

export type Settings = Schemas['Settings'];
export type ProcessingMode = Schemas['ProcessingMode'];
export type DetailLevel = Schemas['DetailLevel'];
export type LineMode = Schemas['LineMode'];
export type OutputFormat = Schemas['OutputFormat'];
export type ConfigResponse = Schemas['ConfigResponse'];
export type Limits = Schemas['Limits'];
export type JobResponse = Schemas['JobResponse'];
export type JobResult = Schemas['JobResult'];
export type JobStatus = Schemas['JobStatus'];
export type JobFile = Schemas['JobFile'];
export type FileKind = Schemas['FileKind'];
export type PipelineStage = Schemas['PipelineStage'];
export type QualityReport = Schemas['QualityReport'];
export type MetricCheck = Schemas['MetricCheck'];
export type MetricName = Schemas['MetricName'];
export type ErrorDetail = Schemas['ErrorDetail'];
export type ErrorResponse = Schemas['ErrorResponse'];
export type RerunRequest = Schemas['RerunRequest'];
export type HealthResponse = Schemas['HealthResponse'];
export type ImageClassLabel = Schemas['ImageClassLabel'];

/** Multipart body of `POST /api/v1/convert` as described by the spec. */
export type ConvertBody = operations['convert']['requestBody']['content']['multipart/form-data'];
