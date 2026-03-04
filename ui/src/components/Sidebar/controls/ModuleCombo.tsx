import { useState, useRef, useEffect, useCallback, type ReactNode } from "react";
import { createPortal } from "react-dom";

export interface ComboGroup {
  label: string;
  options: string[];
}

interface ModuleComboProps {
  label?: string;
  hideLabel?: boolean;
  options?: string[];
  groups?: ComboGroup[];
  value?: string;
  onChange?: (value: string) => void;
  actionIcon?: ReactNode;
  onAction?: () => void;
}

export default function ModuleCombo({
  label,
  hideLabel,
  options,
  groups,
  value,
  onChange,
  actionIcon,
  onAction,
}: ModuleComboProps) {
  // Flatten groups into a single options list if groups are provided
  const allOptions = groups
    ? groups.flatMap((g) => g.options)
    : options ?? [];

  const [internal, setInternal] = useState(allOptions[0] ?? "");
  const selected = value ?? internal;
  const [open, setOpen] = useState(false);
  const comboRef = useRef<HTMLDivElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);
  const [popupPos, setPopupPos] = useState({ top: 0, left: 0, width: 0 });

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (
        comboRef.current && !comboRef.current.contains(target) &&
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
      if (!prev && comboRef.current) {
        const rect = comboRef.current.getBoundingClientRect();
        setPopupPos({ top: rect.top - 2, left: rect.left - 6, width: rect.width + 12 });
      }
      return !prev;
    });
  }, []);

  const handleSelect = useCallback(
    (opt: string) => {
      setInternal(opt);
      onChange?.(opt);
      setOpen(false);
    },
    [onChange],
  );

  const renderOption = (opt: string, isFirst: boolean) =>
    isFirst && label ? (
      <div key={opt} className="bauhaus-combo-popup-row">
        <span className="bauhaus-combo-popup-label">{label}</span>
        <span
          className="bauhaus-combo-option-inline"
          data-selected={opt === selected}
          onClick={() => handleSelect(opt)}
        >
          {opt}
        </span>
      </div>
    ) : (
      <div
        key={opt}
        className="bauhaus-combo-option"
        data-selected={opt === selected}
        onClick={() => handleSelect(opt)}
      >
        {opt}
      </div>
    );

  return (
    <div className="bauhaus-combo" ref={comboRef}>
      <div className="bauhaus-combo-body" onClick={handleToggle}>
        <div className="bauhaus-combo-header">
          {label && !hideLabel && <span className="bauhaus-combo-label">{label}</span>}
          <span className="bauhaus-combo-value">{selected}</span>
          <span className="bauhaus-combo-indicator">
            <svg viewBox="0 0 10 6" xmlns="http://www.w3.org/2000/svg">
              <path d="M0 0l5 6 5-6z" />
            </svg>
          </span>
        </div>
      </div>
      {actionIcon && (
        <div className="bauhaus-combo-action" onClick={onAction}>
          {actionIcon}
        </div>
      )}
      {open && createPortal(
        <div
          ref={popupRef}
          className="bauhaus-combo-popup"
          style={{ top: popupPos.top, left: popupPos.left, minWidth: popupPos.width }}
        >
          {groups
            ? groups.map((group) => (
                <div key={group.label}>
                  <div className="bauhaus-combo-group-header">
                    {group.label}
                  </div>
                  {group.options.map((opt) => (
                    <div
                      key={opt}
                      className="bauhaus-combo-option bauhaus-combo-option-grouped"
                      data-selected={opt === selected}
                      onClick={() => handleSelect(opt)}
                    >
                      {opt}
                    </div>
                  ))}
                </div>
              ))
            : allOptions.map((opt, i) => renderOption(opt, i === 0))}
        </div>,
        document.body,
      )}
    </div>
  );
}
