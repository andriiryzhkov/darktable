import { useState, type ReactNode } from "react";
import { RotateCcw, Menu } from "lucide-react";
import BauhausButton from "../controls/BauhausButton";
import BauhausTooltip from "../controls/BauhausTooltip";

interface ModuleCardProps {
  title: string;
  tooltip?: ReactNode;
  open?: boolean;
  defaultOpen?: boolean;
  onToggle?: (open: boolean) => void;
  /** Icon to the left of the title; receives current open state */
  leftIcon?: ReactNode | ((open: boolean) => ReactNode);
  /** Extra elements after the title (trouble warnings, etc.) */
  afterTitle?: ReactNode;
  /** Extra action buttons before reset/menu */
  extraButtons?: ReactNode;
  onReset?: () => void;
  wrapperRef?: React.Ref<HTMLDivElement>;
  children: ReactNode;
}

export default function ModuleCard({
  title,
  tooltip,
  open: controlledOpen,
  defaultOpen = false,
  onToggle,
  leftIcon,
  afterTitle,
  extraButtons,
  onReset,
  wrapperRef,
  children,
}: ModuleCardProps) {
  const [internalOpen, setInternalOpen] = useState(defaultOpen);
  const open = controlledOpen ?? internalOpen;

  const toggle = () => {
    const next = !open;
    setInternalOpen(next);
    onToggle?.(next);
  };

  const icon = typeof leftIcon === "function" ? leftIcon(open) : leftIcon;

  return (
    <div ref={wrapperRef} className="module-wrapper" data-open={open}>
      <div className="module-header" onClick={toggle}>
        {icon}

        {tooltip ? (
          <BauhausTooltip content={tooltip} placement="bottom-start" delay={700}>
            <span className="flex-1">{title}</span>
          </BauhausTooltip>
        ) : (
          <span className="flex-1">{title}</span>
        )}

        {afterTitle}

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
