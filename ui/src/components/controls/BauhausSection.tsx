import { useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";

interface BauhausSectionProps {
  title: string;
  defaultOpen?: boolean;
  children: ReactNode;
}

export default function BauhausSection({
  title,
  defaultOpen = false,
  children,
}: BauhausSectionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="module-section">
      <button
        className="module-section-header"
        onClick={() => setOpen(!open)}
      >
        <span className="module-section-title">{title}</span>
        <ChevronDown
          size={12}
          className="module-section-chevron"
          style={{ transform: open ? "rotate(180deg)" : "none" }}
        />
      </button>
      {open && <div className="module-section-content">{children}</div>}
    </div>
  );
}
