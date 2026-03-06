import { useEffect, useRef } from "react";
import { useUIStore } from "../../stores/uiStore";

interface Props {
  /** Ref to the element (canvas or img) to overlay guides on */
  targetRef: React.RefObject<HTMLElement | null>;
}

export default function GuidesOverlay({ targetRef }: Props) {
  const showGuides = useUIStore((s) => s.showGuides && s.guidesModuleOpen);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const target = targetRef.current;
    const canvas = canvasRef.current;
    if (!target || !canvas) return;

    if (!showGuides) {
      canvas.style.display = "none";
      return;
    }

    const draw = () => {
      const tr = target.getBoundingClientRect();
      const container = target.closest(".preview-container");
      if (!container) return;
      const pr = container.getBoundingClientRect();

      const w = Math.round(tr.width);
      const h = Math.round(tr.height);
      if (w === 0 || h === 0) return;

      canvas.style.display = "block";
      canvas.style.left = `${tr.left - pr.left}px`;
      canvas.style.top = `${tr.top - pr.top}px`;
      canvas.width = w;
      canvas.height = h;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
      ctx.lineWidth = 1;
      ctx.setLineDash([6, 6]);

      for (const frac of [1 / 3, 2 / 3]) {
        const x = Math.round(w * frac) + 0.5;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }

      for (const frac of [1 / 3, 2 / 3]) {
        const y = Math.round(h * frac) + 0.5;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }
    };

    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(target);
    // Redraw when zoom/pan changes the target's style (transform, transformOrigin)
    const mo = new MutationObserver(draw);
    mo.observe(target, { attributes: true, attributeFilter: ["style"] });
    return () => { ro.disconnect(); mo.disconnect(); };
  }, [showGuides, targetRef]);

  return <canvas ref={canvasRef} className="guides-overlay" />;
}
