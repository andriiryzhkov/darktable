import type { ReactNode } from "react";

interface BauhausButtonProps {
  label?: string;
  icon?: ReactNode;
  disabled?: boolean;
  onClick?: () => void;
}

export default function BauhausButton({ label, icon, disabled, onClick }: BauhausButtonProps) {
  const iconOnly = icon && !label;
  return (
    <button
      className={`bauhaus-button${iconOnly ? " bauhaus-button-icon-only" : ""}`}
      disabled={disabled}
      onClick={onClick}
    >
      {icon && <span className="bauhaus-button-icon">{icon}</span>}
      {label && <span className="bauhaus-button-label">{label}</span>}
    </button>
  );
}
