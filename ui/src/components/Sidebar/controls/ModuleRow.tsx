import type { ReactNode } from "react";

interface ModuleRowProps {
  label: string;
  children: ReactNode;
}

export default function ModuleRow({ label, children }: ModuleRowProps) {
  return (
    <div className="module-row">
      <span className="module-row-label">{label}</span>
      <div className="flex-1 min-w-0">{children}</div>
    </div>
  );
}
