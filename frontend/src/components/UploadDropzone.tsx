import { useId, useRef, useState, type DragEvent } from 'react';
import type { Limits } from '../api/types';
import { formatBytes } from '../lib/format';
import { acceptAttribute, describeAccepted, validateUpload } from '../lib/upload';
import { UploadIcon } from './Icons';

interface UploadDropzoneProps {
  limits: Limits;
  file: File | null;
  disabled?: boolean;
  onFile: (file: File) => void;
  onReject: (message: string) => void;
}

/** Drag-and-drop area plus a keyboard-accessible "Choose image" button. */
export function UploadDropzone({ limits, file, disabled = false, onFile, onReject }: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const id = useId();

  const accept = (files: FileList | null) => {
    const picked = files?.[0];
    if (!files || !picked) return;
    if (files.length > 1) {
      onReject('Please use a single image at a time.');
      return;
    }
    const problem = validateUpload(picked, limits);
    if (problem) onReject(problem);
    else onFile(picked);
  };

  const onDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (disabled) return;
    event.dataTransfer.dropEffect = 'copy';
    setDragging(true);
  };
  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    if (!disabled) accept(event.dataTransfer.files);
  };

  return (
    <section aria-labelledby={`${id}-heading`} className="panel">
      <h2 id={`${id}-heading`} className="panel-title">
        1. Upload
      </h2>
      {/* Drag and drop is a pointer enhancement; the "Choose image" button is the keyboard path. */}
      {/* eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions */}
      <div
        data-testid="dropzone"
        role="group"
        aria-label="Image drop zone"
        onDragOver={onDragOver}
        onDragEnter={onDragOver}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`flex flex-col items-center gap-3 rounded-lg border-2 border-dashed p-5 text-center transition-colors ${
          dragging ? 'border-sky-500 bg-sky-50' : 'border-slate-300 bg-white'
        } ${disabled ? 'opacity-60' : ''}`}
      >
        <UploadIcon className="h-8 w-8 text-slate-500" />
        <p className="text-sm text-slate-700">Drag and drop an image here, or</p>
        <button
          type="button"
          className="btn btn-primary"
          disabled={disabled}
          aria-describedby={`${id}-hint`}
          onClick={() => inputRef.current?.click()}
        >
          Choose image
        </button>
        <input
          ref={inputRef}
          type="file"
          className="sr-only"
          tabIndex={-1}
          aria-label="Image file"
          accept={acceptAttribute(limits)}
          disabled={disabled}
          data-testid="file-input"
          onChange={(event) => {
            accept(event.currentTarget.files);
            event.currentTarget.value = '';
          }}
        />
        <p id={`${id}-hint`} className="text-xs text-slate-600">
          {describeAccepted(limits)}, up to {formatBytes(limits.max_upload_bytes)}
        </p>
      </div>
      {file && (
        <p className="mt-3 text-sm break-all text-slate-700" data-testid="selected-file">
          Selected: <span className="font-medium">{file.name}</span> ({formatBytes(file.size)})
        </p>
      )}
    </section>
  );
}
