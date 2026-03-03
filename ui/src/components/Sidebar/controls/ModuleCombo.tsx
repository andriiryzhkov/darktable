import { useState, useRef, useEffect, useCallback, type ReactNode } from "react";

interface ModuleComboProps {
  label: string;
  options: string[];
  value?: string;
  onChange?: (value: string) => void;
  actionIcon?: ReactNode;
  onAction?: () => void;
}

export default function ModuleCombo({
  label,
  options,
  value,
  onChange,
  actionIcon,
  onAction,
}: ModuleComboProps) {
  const [internal, setInternal] = useState(options[0] ?? "");
  const selected = value ?? internal;
  const [open, setOpen] = useState(false);
  const comboRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (comboRef.current && !comboRef.current.contains(e.target as Node)) {
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
    setOpen((prev) => !prev);
  }, []);

  const handleSelect = useCallback(
    (opt: string) => {
      setInternal(opt);
      onChange?.(opt);
      setOpen(false);
    },
    [onChange],
  );

  return (
    <div className="bauhaus-combo" ref={comboRef}>
      <div className="bauhaus-combo-body" onClick={handleToggle}>
        <div className="bauhaus-combo-header">
          <span className="bauhaus-combo-label">{label}</span>
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
      {open && (
        <div className="bauhaus-combo-popup">
          {options.map((opt, i) =>
            i === 0 ? (
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
            ),
          )}
        </div>
      )}
    </div>
  );
}
