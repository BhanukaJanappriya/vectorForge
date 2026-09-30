/** Decorative inline icons (always paired with visible text; hidden from assistive tech). */
interface IconProps {
  className?: string;
}

const base = 'inline-block shrink-0';

export function CheckIcon({ className = 'h-4 w-4' }: IconProps) {
  return (
    <svg className={`${base} ${className}`} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true" focusable="false">
      <path
        fillRule="evenodd"
        d="M16.7 5.3a1 1 0 0 1 0 1.4l-8 8a1 1 0 0 1-1.4 0l-4-4a1 1 0 1 1 1.4-1.4L8 12.6l7.3-7.3a1 1 0 0 1 1.4 0Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function CrossIcon({ className = 'h-4 w-4' }: IconProps) {
  return (
    <svg className={`${base} ${className}`} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true" focusable="false">
      <path
        fillRule="evenodd"
        d="M4.3 4.3a1 1 0 0 1 1.4 0L10 8.6l4.3-4.3a1 1 0 1 1 1.4 1.4L11.4 10l4.3 4.3a1 1 0 0 1-1.4 1.4L10 11.4l-4.3 4.3a1 1 0 0 1-1.4-1.4L8.6 10 4.3 5.7a1 1 0 0 1 0-1.4Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function AlertIcon({ className = 'h-5 w-5' }: IconProps) {
  return (
    <svg className={`${base} ${className}`} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true" focusable="false">
      <path
        fillRule="evenodd"
        d="M8.3 2.9a2 2 0 0 1 3.4 0l6.2 10.8A2 2 0 0 1 16.2 17H3.8a2 2 0 0 1-1.7-3.3L8.3 2.9ZM10 7a1 1 0 0 0-1 1v3a1 1 0 1 0 2 0V8a1 1 0 0 0-1-1Zm0 8a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function UploadIcon({ className = 'h-8 w-8' }: IconProps) {
  return (
    <svg
      className={`${base} ${className}`}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      aria-hidden="true"
      focusable="false"
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 16V4m0 0-4 4m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
    </svg>
  );
}

export function DownloadIcon({ className = 'h-4 w-4' }: IconProps) {
  return (
    <svg
      className={`${base} ${className}`}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      aria-hidden="true"
      focusable="false"
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v12m0 0-4-4m4 4 4-4M4 20h16" />
    </svg>
  );
}

export function Spinner({ className = 'h-4 w-4' }: IconProps) {
  return (
    <svg
      className={`${base} animate-spin motion-reduce:animate-none ${className}`}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
      focusable="false"
    >
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}
