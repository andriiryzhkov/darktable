import { useState, useCallback } from "react";

interface BauhausCheckboxProps {
  label: string;
  checked?: boolean;
  align?: "left" | "right";
  onChange?: (checked: boolean) => void;
}

export default function BauhausCheckbox({
  label,
  checked: controlledChecked,
  align = "left",
  onChange,
}: BauhausCheckboxProps) {
  const [internal, setInternal] = useState(false);
  const checked = controlledChecked ?? internal;

  const handleClick = useCallback(() => {
    const next = !checked;
    setInternal(next);
    onChange?.(next);
  }, [checked, onChange]);

  const box = (
    <span className="bauhaus-checkbox-box" data-checked={checked}>
      {checked && (
        <svg viewBox="0 0 12 10" xmlns="http://www.w3.org/2000/svg">
          <path d="M1 5l3.5 3.5L11 1" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )}
    </span>
  );

  return (
    <div className="bauhaus-checkbox" data-align={align} onClick={handleClick}>
      {align === "left" && box}
      <span className="bauhaus-checkbox-label">{label}</span>
      {align === "right" && box}
    </div>
  );
}
