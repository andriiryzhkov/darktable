import { useState, type ReactNode } from "react";
import { ChevronRight, RotateCcw, Menu } from "lucide-react";

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
    <div className="module-wrapper" data-open={open}>
      <button className="module-header" onClick={() => setOpen(!open)}>
        <ChevronRight
          size={12}
          className="module-chevron"
          style={{ transform: open ? "rotate(90deg)" : "none" }}
        />
        <span className="flex-1">{title}</span>
        <span
          className="module-actions"
          style={{ visibility: open ? "visible" : "hidden" }}
          onClick={(e) => e.stopPropagation()}
        >
          <span title="Reset" className="module-action-btn">
            <RotateCcw size={10} />
          </span>
          <span title="Presets" className="module-action-btn">
            <Menu size={10} />
          </span>
        </span>
      </button>
      {open && <div className="module-content">{children}</div>}
    </div>
  );
}
