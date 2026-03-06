import { useState, useRef, useEffect, useCallback, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Circle } from "lucide-react";

interface BauhausDropdownProps {
  icon?: ReactNode;
  label?: string;
  options: string[];
  value: string;
  onChange: (value: string) => void;
  /** Extra content rendered inline after a specific option */
  optionSuffix?: (option: string) => ReactNode;
  footer?: ReactNode;
}

export default function BauhausDropdown({
  icon,
  label,
  options,
  value,
  onChange,
  optionSuffix,
  footer,
}: BauhausDropdownProps) {
  const [open, setOpen] = useState(false);
  const btnRef = useRef<HTMLButtonElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);
  const [popupPos, setPopupPos] = useState({ top: 0, left: 0 });

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (
        btnRef.current && !btnRef.current.contains(target) &&
        popupRef.current && !popupRef.current.contains(target)
      ) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const handleToggle = useCallback(() => {
    setOpen((prev) => {
      if (!prev && btnRef.current) {
        const rect = btnRef.current.getBoundingClientRect();
        setPopupPos({ top: rect.bottom + 2, left: rect.left });
      }
      return !prev;
    });
  }, []);

  const handleSelect = useCallback(
    (opt: string) => {
      onChange(opt);
      setOpen(false);
    },
    [onChange],
  );

  const iconOnly = icon && !label;

  return (
    <>
      <button
        ref={btnRef}
        className={`bauhaus-button bauhaus-button-transparent${iconOnly ? " bauhaus-button-icon-only" : ""}`}
        data-active={open}
        onClick={handleToggle}
      >
        {icon && <span className="bauhaus-button-icon">{icon}</span>}
        {label && <span className="bauhaus-button-label">{label}</span>}
      </button>
      {open && createPortal(
        <div
          ref={popupRef}
          className="bauhaus-combo-popup"
          style={{ top: popupPos.top, left: popupPos.left, minWidth: 220 }}
        >
          {options.map((opt) => (
            <div
              key={opt}
              className="bauhaus-dropdown-option"
              data-selected={opt === value}
              onClick={() => handleSelect(opt)}
            >
              <span className="bauhaus-dropdown-dot">{opt === value && <Circle size={6} fill="currentColor" />}</span>
              {opt}
              {optionSuffix?.(opt)}
            </div>
          ))}
          {footer}
        </div>,
        document.body,
      )}
    </>
  );
}
