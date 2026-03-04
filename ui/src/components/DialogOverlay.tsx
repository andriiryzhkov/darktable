import { useEffect, useCallback, useRef, useState, type ReactNode } from "react";
import { usePlatform } from "../hooks/usePlatform";
import WindowControls from "./WindowControls";

interface DialogOverlayProps {
  title?: string;
  onClose?: () => void;
  className?: string;
  zIndex?: number;
  children: ReactNode;
}

export default function DialogOverlay({
  title,
  onClose,
  className,
  zIndex = 200,
  children,
}: DialogOverlayProps) {
  const os = usePlatform();

  // Drag-to-move state
  const [position, setPosition] = useState<{ x: number; y: number } | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const startX = useRef(0);
  const startY = useRef(0);
  const startPosX = useRef(0);
  const startPosY = useRef(0);

  // Reset position each time the overlay mounts
  useEffect(() => setPosition(null), []);

  const onHeaderPointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      dragging.current = true;
      startX.current = e.clientX;
      startY.current = e.clientY;
      if (dialogRef.current && position === null) {
        const rect = dialogRef.current.getBoundingClientRect();
        startPosX.current = rect.left;
        startPosY.current = rect.top;
      } else {
        startPosX.current = position?.x ?? 0;
        startPosY.current = position?.y ?? 0;
      }
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    },
    [position],
  );

  const onHeaderPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragging.current) return;
    setPosition({
      x: startPosX.current + e.clientX - startX.current,
      y: startPosY.current + e.clientY - startY.current,
    });
  }, []);

  const onHeaderPointerUp = useCallback((e: React.PointerEvent) => {
    if (!dragging.current) return;
    dragging.current = false;
    const el = e.currentTarget as HTMLElement;
    if (el.hasPointerCapture(e.pointerId)) {
      el.releasePointerCapture(e.pointerId);
    }
  }, []);

  useEffect(() => {
    if (!onClose) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose]);

  return (
    <div
      className={`dialog-overlay ${className ?? ""}`}
      style={{ zIndex }}
      onClick={onClose}
    >
      <div
        ref={dialogRef}
        onClick={(e) => e.stopPropagation()}
        style={
          position
            ? { position: "absolute", left: position.x, top: position.y }
            : undefined
        }
      >
        {title && (
          <div
            className="dialog-header"
            onPointerDown={onHeaderPointerDown}
            onPointerMove={onHeaderPointerMove}
            onPointerUp={onHeaderPointerUp}
          >
            {os === "macos" && onClose && <WindowControls onClose={onClose} />}
            <span className="dialog-header-title">{title}</span>
            {os && os !== "macos" && onClose && <WindowControls onClose={onClose} />}
          </div>
        )}
        {children}
      </div>
    </div>
  );
}
