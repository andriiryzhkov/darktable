import { useState, useRef, useEffect, useCallback, type ReactNode } from "react";
import { createPortal } from "react-dom";

interface BauhausTooltipProps {
  content: ReactNode;
  children: ReactNode;
  /** Delay in ms before showing (default 500) */
  delay?: number;
  /** Placement: "top", "bottom", "left", "right" with optional alignment suffix "-start" or "-end" (default "bottom") */
  placement?: "top" | "top-start" | "top-end" | "bottom" | "bottom-start" | "bottom-end" | "left" | "left-start" | "left-end" | "right" | "right-start" | "right-end";
}

export default function BauhausTooltip({
  content,
  children,
  delay = 500,
  placement = "bottom",
}: BauhausTooltipProps) {
  const [visible, setVisible] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const triggerRef = useRef<HTMLSpanElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();

  const show = useCallback(() => {
    timerRef.current = setTimeout(() => {
      if (!triggerRef.current) return;
      const rect = triggerRef.current.getBoundingClientRect();
      // Position will be adjusted after render in useEffect
      setPos(computePosition(rect, placement, null));
      setVisible(true);
    }, delay);
  }, [delay, placement]);

  const hide = useCallback(() => {
    clearTimeout(timerRef.current);
    setVisible(false);
  }, []);

  // Adjust position after tooltip renders to avoid overflow
  useEffect(() => {
    if (!visible || !tooltipRef.current || !triggerRef.current) return;
    const triggerRect = triggerRef.current.getBoundingClientRect();
    const tipRect = tooltipRef.current.getBoundingClientRect();
    setPos(computePosition(triggerRect, placement, tipRect));
  }, [visible, placement]);

  return (
    <>
      <span
        ref={triggerRef}
        onMouseEnter={show}
        onMouseLeave={hide}
        onPointerDown={hide}
        className="bauhaus-tooltip-trigger"
      >
        {children}
      </span>
      {visible &&
        createPortal(
          <div
            ref={tooltipRef}
            className="bauhaus-tooltip"
            style={{ top: pos.top, left: pos.left }}
          >
            {content}
          </div>,
          document.body,
        )}
    </>
  );
}

function computePosition(
  trigger: DOMRect,
  placement: string,
  tip: DOMRect | null,
): { top: number; left: number } {
  const gap = 6;
  const tipW = tip?.width ?? 0;
  const tipH = tip?.height ?? 0;

  const [side, align] = placement.split("-") as [string, string | undefined];

  // Horizontal alignment for top/bottom sides
  function alignH(): number {
    if (align === "start") return trigger.left;
    if (align === "end") return trigger.right - tipW;
    return trigger.left + trigger.width / 2 - tipW / 2; // center
  }

  // Vertical alignment for left/right sides
  function alignV(): number {
    if (align === "start") return trigger.top;
    if (align === "end") return trigger.bottom - tipH;
    return trigger.top + trigger.height / 2 - tipH / 2; // center
  }

  let top: number;
  let left: number;

  switch (side) {
    case "bottom":
      top = trigger.bottom + gap;
      left = alignH();
      break;
    case "left":
      top = alignV();
      left = trigger.left - tipW - gap;
      break;
    case "right":
      top = alignV();
      left = trigger.right + gap;
      break;
    default: // top
      top = trigger.top - tipH - gap;
      left = alignH();
      break;
  }

  // Clamp to viewport
  if (tip) {
    const pad = 4;
    left = Math.max(pad, Math.min(left, window.innerWidth - tipW - pad));
    top = Math.max(pad, Math.min(top, window.innerHeight - tipH - pad));
  }

  return { top, left };
}
