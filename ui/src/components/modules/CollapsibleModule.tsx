import { useState, type ReactNode } from "react";
import { CircleChevronRight, CircleChevronDown, RotateCcw, Menu } from "lucide-react";
import BauhausButton from "../controls/BauhausButton";
import BauhausTooltip from "../controls/BauhausTooltip";

interface CollapsibleModuleProps {
  title: string;
  description?: string;
  defaultOpen?: boolean;
  onReset?: () => void;
  extraButtons?: ReactNode;
  children: ReactNode;
}

export default function CollapsibleModule({
  title,
  description,
  defaultOpen = false,
  onReset,
  extraButtons,
  children,
}: CollapsibleModuleProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="module-wrapper" data-open={open}>
      <div className="module-header" onClick={() => setOpen(!open)}>
        {open
          ? <CircleChevronDown size={12} className="module-chevron" />
          : <CircleChevronRight size={12} className="module-chevron" />
        }
        {description ? (
          <BauhausTooltip content={<div className="module-desc-tooltip">{description}</div>} placement="bottom-start" delay={700}>
            <span className="flex-1">{title}</span>
          </BauhausTooltip>
        ) : (
          <span className="flex-1">{title}</span>
        )}
        <span
          className="module-actions"
          onClick={(e) => e.stopPropagation()}
        >
          {extraButtons}
          <BauhausButton icon={<RotateCcw size={12} />} onClick={onReset} />
          <BauhausButton icon={<Menu size={12} />} />
        </span>
      </div>
      {open && <div className="module-content">{children}</div>}
    </div>
  );
}
