import { useState, useCallback, type ReactNode } from "react";

interface BauhausButtonProps {
  label?: string;
  icon?: ReactNode;
  disabled?: boolean;
  /** Controlled active state */
  active?: boolean;
  /** Toggle mode: manages internal on/off state, calls onToggle with new value */
  toggle?: boolean;
  /** Callback for toggle mode */
  onToggle?: (pressed: boolean) => void;
  accent?: boolean;
  title?: string;
  transparent?: boolean;
  onClick?: () => void;
  onMouseDown?: (e: React.MouseEvent) => void;
  onContextMenu?: (e: React.MouseEvent) => void;
}

export default function BauhausButton({ label, icon, disabled, active, toggle, onToggle, accent, title, transparent, onClick, onMouseDown, onContextMenu }: BauhausButtonProps) {
  const [pressed, setPressed] = useState(active ?? false);
  const isActive = toggle ? pressed : active;

  const handleClick = useCallback(() => {
    if (toggle) {
      const next = !isActive;
      setPressed(next);
      onToggle?.(next);
    }
    onClick?.();
  }, [toggle, isActive, onToggle, onClick]);

  const iconOnly = icon && !label;
  const classes = [
    "bauhaus-button",
    iconOnly ? "bauhaus-button-icon-only" : "",
    transparent ? "bauhaus-button-transparent" : "",
    accent ? "bauhaus-button-accent" : "",
  ].filter(Boolean).join(" ");
  return (
    <button
      className={classes}
      disabled={disabled}
      data-active={isActive}
      title={title}
      onClick={handleClick}
      onMouseDown={onMouseDown}
      onContextMenu={onContextMenu}
    >
      {icon && <span className="bauhaus-button-icon">{icon}</span>}
      {label && <span className="bauhaus-button-label">{label}</span>}
    </button>
  );
}
