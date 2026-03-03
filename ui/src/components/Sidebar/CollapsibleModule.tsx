import { useState, type ReactNode } from "react";
import { ChevronRight } from "lucide-react";

interface CollapsibleModuleProps {
  title: string;
  defaultOpen?: boolean;
  children: ReactNode;
}

export default function CollapsibleModule({
  title,
  defaultOpen = false,
  children,
}: CollapsibleModuleProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="module-wrapper">
      <button className="module-header" onClick={() => setOpen(!open)}>
        <ChevronRight
          size={12}
          className="module-chevron"
          style={{ transform: open ? "rotate(90deg)" : "none" }}
        />
        <span>{title}</span>
      </button>
      {open && <div className="module-content">{children}</div>}
    </div>
  );
}
