import { useState, type ReactNode } from "react";
import { RotateCcw, Menu } from "lucide-react";
import BauhausButton from "../controls/BauhausButton";
import BauhausTooltip from "../controls/BauhausTooltip";

interface ModuleCardProps {
  title: ReactNode;
  tooltip?: ReactNode;
  open?: boolean;
  defaultOpen?: boolean;
  onToggle?: (open: boolean, shiftKey?: boolean) => void;
  /** Icon to the left of the title; receives current open state */
  leftIcon?: ReactNode | ((open: boolean) => ReactNode);
  /** Tooltip for the left icon */
  leftIconTooltip?: string;
  /** Extra elements after the title (trouble warnings, etc.) */
  afterTitle?: ReactNode;
  /** Extra action buttons before reset/menu */
  extraButtons?: ReactNode;
  onReset?: () => void;
  resetTooltip?: ReactNode;
  onPresets?: () => void;
  presetsTooltip?: ReactNode;
  presetsButtonRef?: React.RefObject<HTMLElement | null>;
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
  leftIconTooltip,
  afterTitle,
  extraButtons,
  onReset,
  resetTooltip = "reset",
  onPresets,
  presetsTooltip = "presets and preferences",
  presetsButtonRef,
  wrapperRef,
  children,
}: ModuleCardProps) {
  const [internalOpen, setInternalOpen] = useState(defaultOpen);
  const open = controlledOpen ?? internalOpen;

  const toggle = (e: React.MouseEvent) => {
    const next = !open;
    setInternalOpen(next);
    onToggle?.(next, e.shiftKey);
  };

  const icon = typeof leftIcon === "function" ? leftIcon(open) : leftIcon;

  const titleContent = (
    <>
      <span className="flex-1">{title}</span>
      {afterTitle}
    </>
  );

  return (
    <div ref={wrapperRef} className="module-wrapper" data-open={open}>
      <div className="module-header" onClick={toggle}>
        {leftIconTooltip ? (
          <BauhausTooltip content={leftIconTooltip} placement="bottom">
            <span>{icon}</span>
          </BauhausTooltip>
        ) : icon}

        {tooltip ? (
          <BauhausTooltip content={tooltip} placement="bottom-start" delay={700}>
            <span className="module-header-main">
              {titleContent}
            </span>
          </BauhausTooltip>
        ) : (
          <span className="module-header-main">
            {titleContent}
          </span>
        )}

        <span
          className="module-actions"
          onClick={(e) => e.stopPropagation()}
        >
          {extraButtons}
          {onReset ? (
            <BauhausTooltip content={resetTooltip} placement="bottom">
              <BauhausButton icon={<RotateCcw size={12} />} onClick={onReset} />
            </BauhausTooltip>
          ) : (
            <BauhausButton icon={<RotateCcw size={12} />} disabled />
          )}
          <span className="module-presets-wrapper" ref={presetsButtonRef as React.Ref<HTMLSpanElement>}>
            {onPresets ? (
              <BauhausTooltip content={presetsTooltip} placement="bottom">
                <BauhausButton icon={<Menu size={12} />} onClick={onPresets} />
              </BauhausTooltip>
            ) : (
              <BauhausButton icon={<Menu size={12} />} disabled />
            )}
          </span>
        </span>
      </div>
      {open && <div className="module-content">{children}</div>}
    </div>
  );
}
