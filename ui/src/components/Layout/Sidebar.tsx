import { useCallback, useRef, type ReactNode } from "react";

const SIDEBAR_MIN = 150;
const SIDEBAR_MAX = 400;
const clamp = (v: number) => Math.max(SIDEBAR_MIN, Math.min(SIDEBAR_MAX, v));

interface SidebarProps {
  side: "left" | "right";
  open: boolean;
  width: number;
  onToggle: () => void;
  onResize: (w: number) => void;
  /** When true, sidebar-scroll won't scroll; children handle their own scrolling */
  innerScroll?: boolean;
  children: ReactNode;
}

export default function Sidebar({
  side,
  open,
  width,
  onToggle: _onToggle,
  onResize,
  innerScroll = false,
  children,
}: SidebarProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const startX = useRef(0);
  const startW = useRef(0);
  const dragging = useRef(false);

  const onPointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      startX.current = e.clientX;
      startW.current = width;
      dragging.current = true;
      const el = e.currentTarget as HTMLElement;
      el.setPointerCapture(e.pointerId);
      if (containerRef.current) {
        containerRef.current.style.transition = "none";
      }
    },
    [width],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragging.current) return;
      const dx = e.clientX - startX.current;
      const newW = clamp(startW.current + (side === "left" ? dx : -dx));
      if (containerRef.current) {
        containerRef.current.style.width = `${newW}px`;
      }
    },
    [side],
  );

  const onPointerUp = useCallback(
    (e: React.PointerEvent) => {
      if (!dragging.current) return;
      dragging.current = false;
      const el = e.currentTarget as HTMLElement;
      if (el.hasPointerCapture(e.pointerId)) {
        el.releasePointerCapture(e.pointerId);
      }
      if (containerRef.current) {
        containerRef.current.style.transition = "";
        const finalW = parseInt(containerRef.current.style.width, 10);
        onResize(finalW);
      }
    },
    [onResize],
  );

  return (
    <div
      ref={containerRef}
      className="sidebar"
      style={{
        width: open ? width : 0,
        flexDirection: side === "left" ? "row" : "row-reverse",
      }}
    >
      {open && (
        <>
          <div
            className={innerScroll ? "sidebar-scroll sidebar-scroll-inner" : "sidebar-scroll"}
            style={{ direction: side === "left" ? "rtl" : "ltr" }}
          >
            <div style={{ direction: "ltr", display: "flex", flexDirection: "column", flex: 1, minHeight: 0 }}>{children}</div>
          </div>
          <div
            className="sidebar-resize-handle"
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
          />
        </>
      )}
    </div>
  );
}
