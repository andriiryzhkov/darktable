import { useState, type ReactNode } from "react";
import { RotateCcw, Menu, Power, CircleDot, CircleChevronRight, CircleChevronDown } from "lucide-react";
import BauhausButton from "../controls/BauhausButton";
import BauhausTooltip from "../controls/BauhausTooltip";

type LeftButton =
  | { kind: "power"; enabled: boolean; mandatory?: boolean; moduleName: string; onToggle: () => void }
  | { kind: "chevron" };

interface ModuleCardProps {
  title: ReactNode;
  tooltip?: ReactNode;
  open?: boolean;
  defaultOpen?: boolean;
  onToggle?: (open: boolean, shiftKey?: boolean) => void;
  leftButton?: LeftButton;
  indicators?: ReactNode;
  rightButtons?: ReactNode;
  onReset?: () => void;
  resetTooltip?: ReactNode;
  onPresets?: () => void;
  presetsTooltip?: ReactNode;
  presetsButtonRef?: React.RefObject<HTMLElement | null>;
  wrapperRef?: React.Ref<HTMLDivElement>;
  children: ReactNode;
}

function LeftButtonIcon({ button, open }: { button: LeftButton; open: boolean }) {
  if (button.kind === "chevron") {
    return (
      <BauhausTooltip content="show module" placement="bottom">
        <span className="module-actions" onClick={(e) => e.stopPropagation()}>
          <BauhausButton
            icon={open ? <CircleChevronDown size={12} /> : <CircleChevronRight size={12} />}
          />
        </span>
      </BauhausTooltip>
    );
  }

  if (button.mandatory) {
    return (
      <BauhausTooltip content={`'${button.moduleName}' is switched on`} placement="bottom">
        <CircleDot size={12} className="module-mandatory-icon" />
      </BauhausTooltip>
    );
  }

  return (
    <BauhausTooltip content={`'${button.moduleName}' is switched ${button.enabled ? "on" : "off"}`} placement="bottom">
      <span className="module-actions" onClick={(e) => e.stopPropagation()}>
        <BauhausButton
          icon={<Power size={12} />}
          active={button.enabled}
          onClick={button.onToggle}
        />
      </span>
    </BauhausTooltip>
  );
}

export default function ModuleCard({
  title,
  tooltip,
  open: controlledOpen,
  defaultOpen = false,
  onToggle,
  leftButton,
  indicators,
  rightButtons,
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

  return (
    <div ref={wrapperRef} className="module-wrapper" data-open={open}>
      <div className="module-header" onClick={toggle}>
        {leftButton && <LeftButtonIcon button={leftButton} open={open} />}

        {tooltip ? (
          <BauhausTooltip content={tooltip} placement="bottom-start" delay={700}>
            <span className="module-header-main">{title}</span>
          </BauhausTooltip>
        ) : (
          <span className="module-header-main">{title}</span>
        )}

        {indicators}

        <span
          className="module-actions"
          onClick={(e) => e.stopPropagation()}
        >
          {rightButtons}
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
