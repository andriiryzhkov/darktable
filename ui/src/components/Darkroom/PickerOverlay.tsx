import { useRef, useCallback, useEffect, useState, type RefObject } from "react";
import { usePickerStore, type PickerBox } from "../../stores/pickerStore";

type Handle = "tl" | "tr" | "bl" | "br" | "body";

interface PickerOverlayProps {
  targetRef: RefObject<HTMLElement | null>;
}

/**
 * Renders a dashed selection rectangle over the preview image
 * when a picker is active. Positions itself to exactly match
 * the target element (canvas/img) within the preview container.
 */
export default function PickerOverlay({ targetRef }: PickerOverlayProps) {
  const active = usePickerStore((s) => s.active);
  const storeBox = usePickerStore((s) => s.box);
  const setBox = usePickerStore((s) => s.setBox);
  const overlayRef = useRef<HTMLDivElement>(null);
  const [localBox, setLocalBox] = useState<PickerBox | null>(null);
  const box = localBox ?? storeBox;
  const dragRef = useRef<{
    handle: Handle;
    startX: number;
    startY: number;
    startBox: PickerBox;
  } | null>(null);

  // Track the target element's position within the container
  const [imageRect, setImageRect] = useState<{ left: number; top: number; width: number; height: number } | null>(null);

  useEffect(() => {
    if (!active) return;
    const target = targetRef.current;
    const container = target?.parentElement;
    if (!target || !container) return;

    const update = () => {
      const cRect = container.getBoundingClientRect();
      const tRect = target.getBoundingClientRect();
      setImageRect({
        left: tRect.left - cRect.left,
        top: tRect.top - cRect.top,
        width: tRect.width,
        height: tRect.height,
      });
    };

    update();
    const ro = new ResizeObserver(update);
    ro.observe(target);
    ro.observe(container);
    return () => ro.disconnect();
  }, [active, targetRef]);

  const handlePointerDown = useCallback(
    (e: React.PointerEvent, handle: Handle) => {
      e.preventDefault();
      e.stopPropagation();
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      dragRef.current = {
        handle,
        startX: e.clientX,
        startY: e.clientY,
        startBox: { ...box },
      };
    },
    [box],
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      const drag = dragRef.current;
      if (!drag || !imageRect) return;

      const dx = (e.clientX - drag.startX) / imageRect.width;
      const dy = (e.clientY - drag.startY) / imageRect.height;
      const b = drag.startBox;

      let next: PickerBox;

      if (drag.handle === "body") {
        const nx = Math.max(0, Math.min(1 - b.w, b.x + dx));
        const ny = Math.max(0, Math.min(1 - b.h, b.y + dy));
        next = { x: nx, y: ny, w: b.w, h: b.h };
      } else {
        let x0 = b.x;
        let y0 = b.y;
        let x1 = b.x + b.w;
        let y1 = b.y + b.h;

        if (drag.handle === "tl" || drag.handle === "bl") x0 = Math.max(0, Math.min(x1 - 0.02, b.x + dx));
        if (drag.handle === "tr" || drag.handle === "br") x1 = Math.min(1, Math.max(x0 + 0.02, b.x + b.w + dx));
        if (drag.handle === "tl" || drag.handle === "tr") y0 = Math.max(0, Math.min(y1 - 0.02, b.y + dy));
        if (drag.handle === "bl" || drag.handle === "br") y1 = Math.min(1, Math.max(y0 + 0.02, b.y + b.h + dy));

        next = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
      }

      setLocalBox(next);
    },
    [imageRect],
  );

  const handlePointerUp = useCallback(() => {
    if (dragRef.current) {
      dragRef.current = null;
      setLocalBox((b) => {
        if (b) setBox(b);
        return null;
      });
    }
  }, [setBox]);

  if (!active || !imageRect) return null;

  const pct = (v: number) => `${v * 100}%`;

  return (
    <div
      ref={overlayRef}
      className="picker-overlay"
      style={{
        left: imageRect.left,
        top: imageRect.top,
        width: imageRect.width,
        height: imageRect.height,
      }}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    >
      <div
        className="picker-box"
        style={{
          left: pct(box.x),
          top: pct(box.y),
          width: pct(box.w),
          height: pct(box.h),
        }}
        onPointerDown={(e) => handlePointerDown(e, "body")}
      >
        <div className="picker-handle picker-handle-tl" onPointerDown={(e) => handlePointerDown(e, "tl")} />
        <div className="picker-handle picker-handle-tr" onPointerDown={(e) => handlePointerDown(e, "tr")} />
        <div className="picker-handle picker-handle-bl" onPointerDown={(e) => handlePointerDown(e, "bl")} />
        <div className="picker-handle picker-handle-br" onPointerDown={(e) => handlePointerDown(e, "br")} />
      </div>
    </div>
  );
}
