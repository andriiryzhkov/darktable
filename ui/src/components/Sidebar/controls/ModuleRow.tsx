import type { ReactNode } from "react";

interface ModuleRowProps {
  label: string;
  children: ReactNode;
}

export default function ModuleRow({ label, children }: ModuleRowProps) {
  return (
    <div className="flex items-center gap-2 mb-1.5">
      <span
        className="text-xs shrink-0"
        style={{ color: "var(--plugin-label-color)", width: 80 }}
      >
        {label}
      </span>
      <div className="flex-1 min-w-0">{children}</div>
    </div>
  );
}
