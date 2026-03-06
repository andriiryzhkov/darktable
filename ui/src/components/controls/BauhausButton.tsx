import type { ReactNode } from "react";

interface BauhausButtonProps {
  label?: string;
  icon?: ReactNode;
  disabled?: boolean;
  active?: boolean;
  title?: string;
  transparent?: boolean;
  onClick?: () => void;
  onMouseDown?: (e: React.MouseEvent) => void;
  onContextMenu?: (e: React.MouseEvent) => void;
}

export default function BauhausButton({ label, icon, disabled, active, title, transparent, onClick, onMouseDown, onContextMenu }: BauhausButtonProps) {
  const iconOnly = icon && !label;
  const classes = [
    "bauhaus-button",
    iconOnly ? "bauhaus-button-icon-only" : "",
    transparent ? "bauhaus-button-transparent" : "",
  ].filter(Boolean).join(" ");
  return (
    <button
      className={classes}
      disabled={disabled}
      data-active={active}
      title={title}
      onClick={onClick}
      onMouseDown={onMouseDown}
      onContextMenu={onContextMenu}
    >
      {icon && <span className="bauhaus-button-icon">{icon}</span>}
      {label && <span className="bauhaus-button-label">{label}</span>}
    </button>
  );
}
