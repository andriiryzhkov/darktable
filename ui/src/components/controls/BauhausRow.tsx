import type { ReactNode } from "react";

interface BauhausRowProps {
  label: string;
  children: ReactNode;
}

export default function BauhausRow({ label, children }: BauhausRowProps) {
  return (
    <div className="module-row">
      <span className="module-row-label">{label}</span>
      <div className="flex-1 min-w-0">{children}</div>
    </div>
  );
}
