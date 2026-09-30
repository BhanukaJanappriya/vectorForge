import { useId, useState } from 'react';
import { ApiError, fetchJobFile } from '../api/client';
import { FILE_KINDS } from '../api/enums';
import type { FileKind, JobFile } from '../api/types';
import { FILE_KIND_LABELS, downloadName, formatBytes } from '../lib/format';
import { saveBlob } from '../lib/download';
import { DownloadIcon, Spinner } from './Icons';

interface DownloadsProps {
  files: readonly JobFile[];
  sourceFilename: string;
}

/** One button per produced output file (`result.files`, excluding the original upload). */
export function Downloads({ files, sourceFilename }: DownloadsProps) {
  const id = useId();
  const [pending, setPending] = useState<FileKind | null>(null);
  const [error, setError] = useState<string | null>(null);
  const outputs = FILE_KINDS.flatMap((kind) => (kind === 'original' ? [] : files.filter((f) => f.kind === kind)));

  const download = async (file: JobFile) => {
    setPending(file.kind);
    setError(null);
    try {
      const blob = await fetchJobFile(file);
      saveBlob(blob, downloadName(sourceFilename, file.kind));
    } catch (err) {
      const message = err instanceof ApiError ? err.detail.message : 'Download failed.';
      setError(`${FILE_KIND_LABELS[file.kind]}: ${message}`);
    } finally {
      setPending(null);
    }
  };

  return (
    <section aria-labelledby={`${id}-heading`} className="panel" data-testid="downloads">
      <h2 id={`${id}-heading`} className="panel-title">
        Downloads
      </h2>
      {outputs.length === 0 ? (
        <p className="text-sm text-slate-600">No output files were produced.</p>
      ) : (
        <ul className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {outputs.map((file) => (
            <li key={file.kind}>
              <button
                type="button"
                className="btn btn-secondary w-full justify-between"
                onClick={() => void download(file)}
                disabled={pending !== null}
                aria-busy={pending === file.kind}
              >
                <span className="inline-flex items-center gap-2">
                  {pending === file.kind ? <Spinner /> : <DownloadIcon />}
                  Download {FILE_KIND_LABELS[file.kind]}
                </span>
                <span className="text-xs text-slate-600 tabular-nums">{formatBytes(file.size_bytes)}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {error && (
        <p role="alert" className="mt-2 text-sm text-red-700">
          {error}
        </p>
      )}
    </section>
  );
}
