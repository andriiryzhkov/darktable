import { useEffect, useRef } from "react";
import { useDevelopStore } from "../../stores/developStore";
import { MASKS_TYPE } from "../../types/protocol";
import type {
  MaskForm,
  MaskPointsCircle,
  MaskPointsEllipse,
  MaskPointPath,
  MaskPointBrush,
  MaskPointsGradient,
} from "../../types/protocol";

interface Props {
  targetRef: React.RefObject<HTMLElement | null>;
}

const MASK_COLOR = "rgba(255, 255, 80, 0.6)";
const MASK_BORDER_COLOR = "rgba(255, 255, 80, 0.25)";

function drawCircle(ctx: CanvasRenderingContext2D, w: number, h: number, pts: MaskPointsCircle) {
  const cx = pts.center[0] * w;
  const cy = pts.center[1] * h;
  // Radius is normalized relative to the smaller dimension
  const r = pts.radius * Math.min(w, h);
  const borderR = (pts.radius + pts.border) * Math.min(w, h);

  // Border (feather zone)
  ctx.beginPath();
  ctx.arc(cx, cy, borderR, 0, Math.PI * 2);
  ctx.arc(cx, cy, r, 0, Math.PI * 2, true);
  ctx.fillStyle = MASK_BORDER_COLOR;
  ctx.fill();

  // Main circle outline
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.strokeStyle = MASK_COLOR;
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawEllipse(ctx: CanvasRenderingContext2D, w: number, h: number, pts: MaskPointsEllipse) {
  const cx = pts.center[0] * w;
  const cy = pts.center[1] * h;
  const rx = pts.radius[0] * Math.min(w, h);
  const ry = pts.radius[1] * Math.min(w, h);
  const rot = (pts.rotation * Math.PI) / 180;
  const borderRx = (pts.radius[0] + pts.border) * Math.min(w, h);
  const borderRy = (pts.radius[1] + pts.border) * Math.min(w, h);

  // Border (feather zone)
  ctx.beginPath();
  ctx.ellipse(cx, cy, borderRx, borderRy, rot, 0, Math.PI * 2);
  ctx.ellipse(cx, cy, rx, ry, rot, 0, Math.PI * 2, true);
  ctx.fillStyle = MASK_BORDER_COLOR;
  ctx.fill();

  // Main ellipse outline
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, rot, 0, Math.PI * 2);
  ctx.strokeStyle = MASK_COLOR;
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawPath(ctx: CanvasRenderingContext2D, w: number, h: number, pts: MaskPointPath[]) {
  if (pts.length < 2) return;

  ctx.beginPath();
  // Move to first corner
  ctx.moveTo(pts[0].corner[0] * w, pts[0].corner[1] * h);

  // Draw cubic bezier segments between consecutive points
  for (let i = 0; i < pts.length; i++) {
    const curr = pts[i];
    const next = pts[(i + 1) % pts.length];
    ctx.bezierCurveTo(
      curr.ctrl2[0] * w, curr.ctrl2[1] * h,
      next.ctrl1[0] * w, next.ctrl1[1] * h,
      next.corner[0] * w, next.corner[1] * h,
    );
  }

  ctx.closePath();
  ctx.fillStyle = MASK_BORDER_COLOR;
  ctx.fill();
  ctx.strokeStyle = MASK_COLOR;
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawBrush(ctx: CanvasRenderingContext2D, w: number, h: number, pts: MaskPointBrush[]) {
  if (pts.length < 2) return;

  ctx.beginPath();
  ctx.moveTo(pts[0].corner[0] * w, pts[0].corner[1] * h);

  for (let i = 0; i < pts.length - 1; i++) {
    const curr = pts[i];
    const next = pts[i + 1];
    ctx.bezierCurveTo(
      curr.ctrl2[0] * w, curr.ctrl2[1] * h,
      next.ctrl1[0] * w, next.ctrl1[1] * h,
      next.corner[0] * w, next.corner[1] * h,
    );
  }

  ctx.strokeStyle = MASK_COLOR;
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.stroke();
}

function drawGradient(ctx: CanvasRenderingContext2D, w: number, h: number, pts: MaskPointsGradient) {
  const ax = pts.anchor[0] * w;
  const ay = pts.anchor[1] * h;
  const rot = (pts.rotation * Math.PI) / 180;
  const comp = pts.compression;

  // Gradient extends perpendicular to the rotation angle
  // The anchor is the center; compression controls the width of the transition
  const halfSpan = Math.max(w, h) * 0.5;
  const transitionHalf = halfSpan * (1 - comp);

  // Direction perpendicular to the gradient line
  const dx = Math.sin(rot);
  const dy = -Math.cos(rot);

  // Draw the gradient line through the anchor
  const lineLen = Math.max(w, h) * 1.5;
  const lx = Math.cos(rot);
  const ly = Math.sin(rot);

  // Main gradient line
  ctx.beginPath();
  ctx.moveTo(ax - lx * lineLen, ay - ly * lineLen);
  ctx.lineTo(ax + lx * lineLen, ay + ly * lineLen);
  ctx.strokeStyle = MASK_COLOR;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Upper and lower bounds of the transition zone
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = MASK_BORDER_COLOR;
  ctx.lineWidth = 1;

  ctx.beginPath();
  ctx.moveTo(ax + dx * transitionHalf - lx * lineLen, ay + dy * transitionHalf - ly * lineLen);
  ctx.lineTo(ax + dx * transitionHalf + lx * lineLen, ay + dy * transitionHalf + ly * lineLen);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(ax - dx * transitionHalf - lx * lineLen, ay - dy * transitionHalf - ly * lineLen);
  ctx.lineTo(ax - dx * transitionHalf + lx * lineLen, ay - dy * transitionHalf + ly * lineLen);
  ctx.stroke();

  ctx.setLineDash([]);
}

function drawForm(ctx: CanvasRenderingContext2D, w: number, h: number, form: MaskForm) {
  if (!form.points) return;
  const baseType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);

  switch (baseType) {
    case MASKS_TYPE.CIRCLE:
      drawCircle(ctx, w, h, form.points as MaskPointsCircle);
      break;
    case MASKS_TYPE.ELLIPSE:
      drawEllipse(ctx, w, h, form.points as MaskPointsEllipse);
      break;
    case MASKS_TYPE.PATH:
      drawPath(ctx, w, h, form.points as MaskPointPath[]);
      break;
    case MASKS_TYPE.BRUSH:
      drawBrush(ctx, w, h, form.points as MaskPointBrush[]);
      break;
    case MASKS_TYPE.GRADIENT:
      drawGradient(ctx, w, h, form.points as MaskPointsGradient);
      break;
  }
}

export default function MaskOverlay({ targetRef }: Props) {
  const showMasks = useDevelopStore((s) => s.showMasks);
  const maskForms = useDevelopStore((s) => s.maskForms);
  const maskUsage = useDevelopStore((s) => s.maskUsage);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const target = targetRef.current;
    const canvas = canvasRef.current;
    if (!target || !canvas) return;

    if (!showMasks || maskForms.length === 0) {
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

      // Build set of forms used by modules (non-group leaf shapes)
      const usedFormIds = new Set<number>();
      for (const u of maskUsage) {
        const group = maskForms.find((f) => f.formid === u.mask_id);
        if (group?.children) {
          for (const c of group.children) usedFormIds.add(c.formid);
        }
      }

      // Draw all used leaf shapes
      for (const form of maskForms) {
        if (usedFormIds.has(form.formid)) {
          drawForm(ctx, w, h, form);
        }
      }
    };

    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(target);
    const mo = new MutationObserver(draw);
    mo.observe(target, { attributes: true, attributeFilter: ["style"] });
    return () => { ro.disconnect(); mo.disconnect(); };
  }, [showMasks, maskForms, maskUsage, targetRef]);

  return <canvas ref={canvasRef} className="mask-overlay" />;
}
