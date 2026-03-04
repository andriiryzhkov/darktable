import { useState, type ReactNode } from "react";
import { ChevronRight, RotateCcw, Menu } from "lucide-react";
import ModuleButton from "./controls/ModuleButton";

interface CollapsibleModuleProps {
  title: string;
  defaultOpen?: boolean;
  onReset?: () => void;
  extraButtons?: ReactNode;
  children: ReactNode;
}

export default function CollapsibleModule({
  title,
  defaultOpen = false,
  onReset,
  extraButtons,
  children,
}: CollapsibleModuleProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="module-wrapper" data-open={open}>
      <div className="module-header" onClick={() => setOpen(!open)}>
        <ChevronRight
          size={12}
          className="module-chevron"
          style={{ transform: open ? "rotate(90deg)" : "none" }}
        />
        <span className="flex-1">{title}</span>
        <span
          className="module-actions"
          onClick={(e) => e.stopPropagation()}
        >
          {extraButtons}
          <ModuleButton icon={<RotateCcw size={12} />} onClick={onReset} />
          <ModuleButton icon={<Menu size={12} />} />
        </span>
      </div>
      {open && <div className="module-content">{children}</div>}
    </div>
  );
}
