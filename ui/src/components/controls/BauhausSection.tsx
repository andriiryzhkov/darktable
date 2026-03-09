import type { ReactNode } from "react";

interface BauhausSectionProps {
  title: string;
  children: ReactNode;
}

export default function BauhausSection({ title, children }: BauhausSectionProps) {
  return (
    <div className="bauhaus-section">
      <div className="bauhaus-section-header">
        <span className="bauhaus-section-title">{title}</span>
      </div>
      <div className="bauhaus-section-content">{children}</div>
    </div>
  );
}
