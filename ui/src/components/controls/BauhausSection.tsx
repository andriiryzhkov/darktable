import type { ReactNode } from "react";

interface BauhausSectionProps {
  /** Section title text — ignored when `header` is provided */
  title?: string;
  children: ReactNode;
  /** Custom header element replacing the title text but keeping the section header styling */
  header?: ReactNode;
  /** Optional controls rendered in the header row after the title */
  headerControls?: ReactNode;
}

export default function BauhausSection({ title, children, header, headerControls }: BauhausSectionProps) {
  return (
    <div className="bauhaus-section">
      <div className="bauhaus-section-header">
        {header ?? (
          <>
            <span className="bauhaus-section-title">{title}</span>
            {headerControls && (
              <span className="bauhaus-section-controls">{headerControls}</span>
            )}
          </>
        )}
      </div>
      <div className="bauhaus-section-content">{children}</div>
    </div>
  );
}
