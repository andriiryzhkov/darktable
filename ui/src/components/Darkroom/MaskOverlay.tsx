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
  MaskTransformedCircle,
  MaskTransformedEllipse,
  MaskTransformedGradient,
  MaskTransformedPath,
  MaskTransformedBrush,
} from "../../types/protocol";

interface Props {
  targetRef: React.RefObject<HTMLElement | null>;
}

// DT dual-stroke: dark background stroke + bright foreground stroke
const DARK = "rgba(40, 40, 40, 0.5)";
const DARK_SEL = "rgba(40, 40, 40, 0.8)";
const BRIGHT = "rgba(200, 200, 200, 0.6)";
const BRIGHT_SEL = "rgba(218, 218, 218, 0.9)";
const MASK_HANDLE_SIZE = 6;
const HANDLE_HIT_RADIUS = 12;

// Line widths matching DT
const LW_MASK = 1.7;
const LW_BORDER = 1.0;
const LW_SEL_MULT = 1.5;

/** Dual-stroke a pre-built path: dark bg then bright fg */
function dualStroke(ctx: CanvasRenderingContext2D, border: boolean, selected: boolean) {
  const baseW = border ? LW_BORDER : LW_MASK;
  const lw = baseW * (selected ? LW_SEL_MULT : 1);
  const dash: [number, number] = [4, 4];

  // Background stroke (dark, full width)
  ctx.strokeStyle = selected ? DARK_SEL : DARK;
  ctx.lineWidth = lw;
  if (border) ctx.setLineDash(dash);
  else ctx.setLineDash([]);
  ctx.stroke();

  // Foreground stroke (bright, thinner)
  ctx.strokeStyle = selected ? BRIGHT_SEL : BRIGHT;
  ctx.lineWidth = selected && !border ? lw : lw / 2;
  if (border) ctx.setLineDash(dash);
  ctx.stroke();

  ctx.setLineDash([]);
}

/** Dual-stroke a Path2D object: dark bg then bright fg */
function dualStrokePath2D(ctx: CanvasRenderingContext2D, path: Path2D, border: boolean, selected: boolean) {
  const baseW = border ? LW_BORDER : LW_MASK;
  const lw = baseW * (selected ? LW_SEL_MULT : 1);
  const dash: [number, number] = [4, 4];

  ctx.strokeStyle = selected ? DARK_SEL : DARK;
  ctx.lineWidth = lw;
  if (border) ctx.setLineDash(dash);
  else ctx.setLineDash([]);
  ctx.stroke(path);

  ctx.strokeStyle = selected ? BRIGHT_SEL : BRIGHT;
  ctx.lineWidth = selected && !border ? lw : lw / 2;
  if (border) ctx.setLineDash(dash);
  ctx.stroke(path);

  ctx.setLineDash([]);
}

function isNearHandle(hx: number, hy: number, mx: number | null, my: number | null): boolean {
  if (mx === null || my === null) return false;
  const dx = hx - mx, dy = hy - my;
  return dx * dx + dy * dy <= HANDLE_HIT_RADIUS * HANDLE_HIT_RADIUS;
}

function drawHandle(ctx: CanvasRenderingContext2D, x: number, y: number, mx: number | null, my: number | null) {
  const near = isNearHandle(x, y, mx, my);
  const hs = near ? MASK_HANDLE_SIZE * 1.5 : MASK_HANDLE_SIZE;
  ctx.fillStyle = "rgba(200, 200, 200, 0.9)";
  ctx.strokeStyle = "rgba(40, 40, 40, 0.8)";
  ctx.lineWidth = 1;
  ctx.fillRect(x - hs / 2, y - hs / 2, hs, hs);
  ctx.strokeRect(x - hs / 2, y - hs / 2, hs, hs);
}

function drawCircle(ctx: CanvasRenderingContext2D, w: number, h: number, pts: MaskPointsCircle, hovered = false, mx: number | null = null, my: number | null = null) {
  const cx = pts.center[0] * w;
  const cy = pts.center[1] * h;
  const dim = Math.min(w, h);
  const r = pts.radius * dim;
  const borderR = (pts.radius + pts.border) * dim;

  // Main circle — solid
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  dualStroke(ctx, false, hovered);

  // Feather border — dashed
  ctx.beginPath();
  ctx.arc(cx, cy, borderR, 0, Math.PI * 2);
  dualStroke(ctx, true, hovered);

  // Handles
  drawHandle(ctx, cx + r, cy, mx, my);
  drawHandle(ctx, cx + borderR, cy, mx, my);
}

function drawCirclePolyline(ctx: CanvasRenderingContext2D, w: number, h: number, t: MaskTransformedCircle, hovered = false, mx: number | null = null, my: number | null = null) {
  // Main circle — solid polyline
  const mainPath = buildPolylinePath2D(t.main_polyline, w, h, true);
  dualStrokePath2D(ctx, mainPath, false, hovered);

  // Feather border — dashed polyline
  const borderPath = buildPolylinePath2D(t.border_polyline, w, h, true);
  dualStrokePath2D(ctx, borderPath, true, hovered);

  // Handles: pick first point on each polyline (angle=0, rightmost)
  if (t.main_polyline.length >= 2) {
    drawHandle(ctx, t.main_polyline[0] * w, t.main_polyline[1] * h, mx, my);
  }
  if (t.border_polyline.length >= 2) {
    drawHandle(ctx, t.border_polyline[0] * w, t.border_polyline[1] * h, mx, my);
  }
}

function hitTestCircle(w: number, h: number, pts: MaskPointsCircle, px: number, py: number): boolean {
  const cx = pts.center[0] * w;
  const cy = pts.center[1] * h;
  const dim = Math.min(w, h);
  const r = pts.radius * dim;
  const borderR = (pts.radius + pts.border) * dim;
  const dx = px - cx;
  const dy = py - cy;
  if (dx * dx + dy * dy <= borderR * borderR) return true;
  if (isNearHandle(cx + r, cy, px, py)) return true;
  if (isNearHandle(cx + borderR, cy, px, py)) return true;
  return false;
}

function hitTestCirclePolyline(ctx: CanvasRenderingContext2D, w: number, h: number, t: MaskTransformedCircle, px: number, py: number): boolean {
  const borderPath = buildPolylinePath2D(t.border_polyline, w, h, true);
  if (ctx.isPointInPath(borderPath, px, py)) return true;
  if (t.main_polyline.length >= 2 && isNearHandle(t.main_polyline[0] * w, t.main_polyline[1] * h, px, py)) return true;
  if (t.border_polyline.length >= 2 && isNearHandle(t.border_polyline[0] * w, t.border_polyline[1] * h, px, py)) return true;
  return false;
}

function drawEllipse(ctx: CanvasRenderingContext2D, w: number, h: number, pts: MaskPointsEllipse, hovered = false, mx: number | null = null, my: number | null = null) {
  const cx = pts.center[0] * w;
  const cy = pts.center[1] * h;
  const dim = Math.min(w, h);
  const rx = pts.radius[0] * dim;
  const ry = pts.radius[1] * dim;
  const rot = (pts.rotation * Math.PI) / 180;
  const borderRx = (pts.radius[0] + pts.border) * dim;
  const borderRy = (pts.radius[1] + pts.border) * dim;

  // Main ellipse — solid
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, rot, 0, Math.PI * 2);
  dualStroke(ctx, false, hovered);

  // Feather border — dashed
  ctx.beginPath();
  ctx.ellipse(cx, cy, borderRx, borderRy, rot, 0, Math.PI * 2);
  dualStroke(ctx, true, hovered);

  // Handles on all 4 sides of both ellipses (rotated)
  const cosR = Math.cos(rot);
  const sinR = Math.sin(rot);
  drawHandle(ctx, cx + rx * cosR, cy + rx * sinR, mx, my);
  drawHandle(ctx, cx - rx * cosR, cy - rx * sinR, mx, my);
  drawHandle(ctx, cx - ry * sinR, cy + ry * cosR, mx, my);
  drawHandle(ctx, cx + ry * sinR, cy - ry * cosR, mx, my);
  drawHandle(ctx, cx + borderRx * cosR, cy + borderRx * sinR, mx, my);
  drawHandle(ctx, cx - borderRx * cosR, cy - borderRx * sinR, mx, my);
  drawHandle(ctx, cx - borderRy * sinR, cy + borderRy * cosR, mx, my);
  drawHandle(ctx, cx + borderRy * sinR, cy - borderRy * cosR, mx, my);
}

function drawEllipsePolyline(ctx: CanvasRenderingContext2D, w: number, h: number, t: MaskTransformedEllipse, hovered = false, mx: number | null = null, my: number | null = null) {
  // Main ellipse — solid polyline
  const mainPath = buildPolylinePath2D(t.main_polyline, w, h, true);
  dualStrokePath2D(ctx, mainPath, false, hovered);

  // Feather border — dashed polyline
  const borderPath = buildPolylinePath2D(t.border_polyline, w, h, true);
  dualStrokePath2D(ctx, borderPath, true, hovered);

  // Handles: pick 4 axis points from main polyline (at 0°, 90°, 180°, 270°)
  // and corresponding border points
  const nMain = t.main_polyline.length / 2;
  const nBorder = t.border_polyline.length / 2;
  for (let q = 0; q < 4; q++) {
    const mi = Math.round(q * nMain / 4) % nMain;
    drawHandle(ctx, t.main_polyline[mi * 2] * w, t.main_polyline[mi * 2 + 1] * h, mx, my);
    const bi = Math.round(q * nBorder / 4) % nBorder;
    drawHandle(ctx, t.border_polyline[bi * 2] * w, t.border_polyline[bi * 2 + 1] * h, mx, my);
  }
}

function hitTestEllipse(w: number, h: number, pts: MaskPointsEllipse, px: number, py: number): boolean {
  const cx = pts.center[0] * w;
  const cy = pts.center[1] * h;
  const dim = Math.min(w, h);
  const rx = pts.radius[0] * dim;
  const borderRx = (pts.radius[0] + pts.border) * dim;
  const borderRy = (pts.radius[1] + pts.border) * dim;
  const rot = (pts.rotation * Math.PI) / 180;
  const cosR = Math.cos(-rot);
  const sinR = Math.sin(-rot);
  const dx = px - cx;
  const dy = py - cy;
  const lx = dx * cosR - dy * sinR;
  const ly = dx * sinR + dy * cosR;
  if ((lx * lx) / (borderRx * borderRx) + (ly * ly) / (borderRy * borderRy) <= 1) return true;
  const ry = pts.radius[1] * dim;
  const hCos = Math.cos(rot);
  const hSin = Math.sin(rot);
  if (isNearHandle(cx + rx * hCos, cy + rx * hSin, px, py)) return true;
  if (isNearHandle(cx - rx * hCos, cy - rx * hSin, px, py)) return true;
  if (isNearHandle(cx - ry * hSin, cy + ry * hCos, px, py)) return true;
  if (isNearHandle(cx + ry * hSin, cy - ry * hCos, px, py)) return true;
  if (isNearHandle(cx + borderRx * hCos, cy + borderRx * hSin, px, py)) return true;
  if (isNearHandle(cx - borderRx * hCos, cy - borderRx * hSin, px, py)) return true;
  if (isNearHandle(cx - borderRy * hSin, cy + borderRy * hCos, px, py)) return true;
  if (isNearHandle(cx + borderRy * hSin, cy - borderRy * hCos, px, py)) return true;
  return false;
}

function hitTestEllipsePolyline(ctx: CanvasRenderingContext2D, w: number, h: number, t: MaskTransformedEllipse, px: number, py: number): boolean {
  const borderPath = buildPolylinePath2D(t.border_polyline, w, h, true);
  if (ctx.isPointInPath(borderPath, px, py)) return true;
  // Check handle proximity at 4 axis points
  const nMain = t.main_polyline.length / 2;
  const nBorder = t.border_polyline.length / 2;
  for (let q = 0; q < 4; q++) {
    const mi = Math.round(q * nMain / 4) % nMain;
    if (isNearHandle(t.main_polyline[mi * 2] * w, t.main_polyline[mi * 2 + 1] * h, px, py)) return true;
    const bi = Math.round(q * nBorder / 4) % nBorder;
    if (isNearHandle(t.border_polyline[bi * 2] * w, t.border_polyline[bi * 2 + 1] * h, px, py)) return true;
  }
  return false;
}

function drawCtrlHandle(ctx: CanvasRenderingContext2D, x: number, y: number, mx: number | null, my: number | null) {
  const near = isNearHandle(x, y, mx, my);
  const r = near ? 4 : 3;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(200, 200, 200, 0.9)";
  ctx.strokeStyle = "rgba(40, 40, 40, 0.8)";
  ctx.lineWidth = 1;
  ctx.fill();
  ctx.stroke();
}

/** Build a Path2D from a flat array of pre-transformed border coordinates [x,y,x,y,...] in normalized space */
function buildPolylinePath2D(polyline: number[], w: number, h: number, close: boolean): Path2D {
  const p = new Path2D();
  if (polyline.length < 4) return p;
  p.moveTo(polyline[0] * w, polyline[1] * h);
  for (let i = 2; i < polyline.length; i += 2) {
    p.lineTo(polyline[i] * w, polyline[i + 1] * h);
  }
  if (close) p.closePath();
  return p;
}

function drawPath(ctx: CanvasRenderingContext2D, w: number, h: number, pts: MaskPointPath[], hovered = false, mx: number | null = null, my: number | null = null, editedIdx: number | null = null, serverBorderPolyline?: number[]) {
  if (pts.length < 2) return;

  // Main path — solid
  ctx.beginPath();
  ctx.moveTo(pts[0].corner[0] * w, pts[0].corner[1] * h);
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
  dualStroke(ctx, false, hovered);

  // Feather border — use server-side polyline if available, else compute client-side
  let borderPath: Path2D;
  let borderAnchors: BorderAnchor[];
  if (serverBorderPolyline && serverBorderPolyline.length >= 4) {
    borderPath = buildPolylinePath2D(serverBorderPolyline, w, h, true);
    // Border anchors from polyline: each anchor is at segment boundary
    // Server emits (BORDER_SAMPLES+1) points per segment, anchor is at index k*(BORDER_SAMPLES+1)
    const samplesPerSeg = Math.round(serverBorderPolyline.length / 2 / pts.length);
    borderAnchors = pts.map((p, i) => {
      const idx = i * samplesPerSeg * 2;
      if (idx + 1 < serverBorderPolyline.length) {
        return { x: serverBorderPolyline[idx] * w, y: serverBorderPolyline[idx + 1] * h };
      }
      return { x: p.corner[0] * w, y: p.corner[1] * h };
    });
  } else {
    const cw = pathWindingCW(pts);
    borderPath = buildBorderPolyline(pts, w, h, cw);
    borderAnchors = computeBorderAnchors(pts, w, h, cw);
  }
  dualStrokePath2D(ctx, borderPath, true, hovered);

  // Handles at each corner + border handles with connecting line on border hover
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const ba = borderAnchors[i];
    const cornerX = p.corner[0] * w;
    const cornerY = p.corner[1] * h;

    // If border handle is hovered, draw connecting line
    if (isNearHandle(ba.x, ba.y, mx, my)) {
      ctx.beginPath();
      ctx.moveTo(cornerX, cornerY);
      ctx.lineTo(ba.x, ba.y);
      dualStroke(ctx, true, false);
    }

    drawHandle(ctx, cornerX, cornerY, mx, my);
    drawHandle(ctx, ba.x, ba.y, mx, my);
  }

  // Bezier control points for edited point
  if (editedIdx !== null && editedIdx >= 0 && editedIdx < pts.length) {
    const p = pts[editedIdx];
    const cornerX = p.corner[0] * w;
    const cornerY = p.corner[1] * h;
    const c1x = p.ctrl1[0] * w, c1y = p.ctrl1[1] * h;
    const c2x = p.ctrl2[0] * w, c2y = p.ctrl2[1] * h;

    // Line from corner to ctrl1
    ctx.beginPath();
    ctx.moveTo(cornerX, cornerY);
    ctx.lineTo(c1x, c1y);
    dualStroke(ctx, true, false);

    // Line from corner to ctrl2
    ctx.beginPath();
    ctx.moveTo(cornerX, cornerY);
    ctx.lineTo(c2x, c2y);
    dualStroke(ctx, true, false);

    // Control point handles (circles)
    drawCtrlHandle(ctx, c1x, c1y, mx, my);
    drawCtrlHandle(ctx, c2x, c2y, mx, my);
  }
}

/**
 * Build a closed brush border outline with semicircular end caps.
 * Goes forward along one side, adds an end cap arc, returns along the other side,
 * and closes with a start cap arc.
 */
function buildBrushBorderOutline(pts: MaskPointBrush[], w: number, h: number): Path2D {
  const dim = Math.min(w, h);
  const nSamples = 80;

  // Collect border points on both sides
  const side1: [number, number][] = [];
  const side2: [number, number][] = [];

  for (let k = 0; k < pts.length - 1; k++) {
    const pt1 = pts[k];
    const pt2 = pts[k + 1];
    const p0x = pt1.corner[0] * w, p0y = pt1.corner[1] * h;
    const p1x = pt1.ctrl2[0] * w, p1y = pt1.ctrl2[1] * h;
    const p2x = pt2.ctrl1[0] * w, p2y = pt2.ctrl1[1] * h;
    const p3x = pt2.corner[0] * w, p3y = pt2.corner[1] * h;
    const radStart = pt1.border[1] * dim;
    const radEnd = pt2.border[0] * dim;

    for (let s = 0; s <= nSamples; s++) {
      const t = s / nSamples;
      const rad = radStart + (radEnd - radStart) * t;
      side1.push(borderPointAt(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, t, rad));
      side2.push(borderPointAt(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, t, -rad));
    }
  }

  const p = new Path2D();
  if (side1.length === 0) return p;

  // Forward along side1
  p.moveTo(side1[0][0], side1[0][1]);
  for (let i = 1; i < side1.length; i++) p.lineTo(side1[i][0], side1[i][1]);

  // End cap: semicircular arc from side1 end to side2 end
  const endPt = pts[pts.length - 1];
  const endCx = endPt.corner[0] * w, endCy = endPt.corner[1] * h;
  const endRad = endPt.border[0] * dim;
  const endS1 = side1[side1.length - 1], endS2 = side2[side2.length - 1];
  const endAngle1 = Math.atan2(endS1[1] - endCy, endS1[0] - endCx);
  const endAngle2 = Math.atan2(endS2[1] - endCy, endS2[0] - endCx);
  p.arc(endCx, endCy, endRad, endAngle1, endAngle2, false);

  // Backward along side2
  for (let i = side2.length - 1; i >= 0; i--) p.lineTo(side2[i][0], side2[i][1]);

  // Start cap: semicircular arc from side2 start to side1 start
  const startPt = pts[0];
  const startCx = startPt.corner[0] * w, startCy = startPt.corner[1] * h;
  const startRad = startPt.border[1] * dim;
  const startAngle2 = Math.atan2(side2[0][1] - startCy, side2[0][0] - startCx);
  const startAngle1 = Math.atan2(side1[0][1] - startCy, side1[0][0] - startCx);
  p.arc(startCx, startCy, startRad, startAngle2, startAngle1, false);

  p.closePath();
  return p;
}

function drawBrush(ctx: CanvasRenderingContext2D, w: number, h: number, pts: MaskPointBrush[], hovered = false, mx: number | null = null, my: number | null = null, serverPolyline1?: number[], serverPolyline2?: number[], hoveredSeg = -1) {
  if (pts.length < 2) return;

  // Main brush spline — draw each segment individually for per-segment highlighting
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (let i = 0; i < pts.length - 1; i++) {
    const curr = pts[i];
    const next = pts[i + 1];
    const segPath = new Path2D();
    segPath.moveTo(curr.corner[0] * w, curr.corner[1] * h);
    segPath.bezierCurveTo(
      curr.ctrl2[0] * w, curr.ctrl2[1] * h,
      next.ctrl1[0] * w, next.ctrl1[1] * h,
      next.corner[0] * w, next.corner[1] * h,
    );
    const segSelected = hovered && (hoveredSeg === -1 || hoveredSeg === i);
    dualStrokePath2D(ctx, segPath, false, segSelected);
  }

  // Feather border — use server-side polylines if available
  if (serverPolyline1 && serverPolyline1.length >= 4 && serverPolyline2 && serverPolyline2.length >= 4) {
    // Build a closed outline: side1 forward, end cap, side2 backward, start cap
    const outline = new Path2D();
    outline.moveTo(serverPolyline1[0] * w, serverPolyline1[1] * h);
    for (let i = 2; i < serverPolyline1.length; i += 2) {
      outline.lineTo(serverPolyline1[i] * w, serverPolyline1[i + 1] * h);
    }
    // End cap arc
    const endPt = pts[pts.length - 1];
    const endCx = endPt.corner[0] * w, endCy = endPt.corner[1] * h;
    const endS1x = serverPolyline1[serverPolyline1.length - 2] * w;
    const endS1y = serverPolyline1[serverPolyline1.length - 1] * h;
    const endS2x = serverPolyline2[serverPolyline2.length - 2] * w;
    const endS2y = serverPolyline2[serverPolyline2.length - 1] * h;
    const endRad = Math.sqrt((endS1x - endCx) ** 2 + (endS1y - endCy) ** 2);
    const endAngle1 = Math.atan2(endS1y - endCy, endS1x - endCx);
    const endAngle2 = Math.atan2(endS2y - endCy, endS2x - endCx);
    outline.arc(endCx, endCy, endRad, endAngle1, endAngle2, false);
    // Side2 backward
    for (let i = serverPolyline2.length - 2; i >= 0; i -= 2) {
      outline.lineTo(serverPolyline2[i] * w, serverPolyline2[i + 1] * h);
    }
    // Start cap arc
    const startPt = pts[0];
    const startCx = startPt.corner[0] * w, startCy = startPt.corner[1] * h;
    const startS2x = serverPolyline2[0] * w, startS2y = serverPolyline2[1] * h;
    const startS1x = serverPolyline1[0] * w, startS1y = serverPolyline1[1] * h;
    const startRad = Math.sqrt((startS2x - startCx) ** 2 + (startS2y - startCy) ** 2);
    const startAngle2 = Math.atan2(startS2y - startCy, startS2x - startCx);
    const startAngle1 = Math.atan2(startS1y - startCy, startS1x - startCx);
    outline.arc(startCx, startCy, startRad, startAngle2, startAngle1, false);
    outline.closePath();
    dualStrokePath2D(ctx, outline, true, hovered);
  } else {
    const borderPath = buildBrushBorderOutline(pts, w, h);
    dualStrokePath2D(ctx, borderPath, true, hovered);
  }

  // Corner handles
  for (let i = 0; i < pts.length; i++) {
    drawHandle(ctx, pts[i].corner[0] * w, pts[i].corner[1] * h, mx, my);
  }
}

/** Returns the hovered segment index (0-based), or -1 if no segment is hit */
function hitTestBrushSegment(ctx: CanvasRenderingContext2D, w: number, h: number, pts: MaskPointBrush[], px: number, py: number): number {
  if (pts.length < 2) return -1;

  // Test each segment individually
  ctx.lineWidth = 10;
  for (let i = 0; i < pts.length - 1; i++) {
    const curr = pts[i];
    const next = pts[i + 1];
    const segPath = new Path2D();
    segPath.moveTo(curr.corner[0] * w, curr.corner[1] * h);
    segPath.bezierCurveTo(
      curr.ctrl2[0] * w, curr.ctrl2[1] * h,
      next.ctrl1[0] * w, next.ctrl1[1] * h,
      next.corner[0] * w, next.corner[1] * h,
    );
    if (ctx.isPointInStroke(segPath, px, py)) return i;
  }

  // Check handle proximity — return the segment starting at that handle
  for (let i = 0; i < pts.length; i++) {
    if (isNearHandle(pts[i].corner[0] * w, pts[i].corner[1] * h, px, py)) {
      return Math.min(i, pts.length - 2);
    }
  }
  return -1;
}

function hitTestBrush(ctx: CanvasRenderingContext2D, w: number, h: number, pts: MaskPointBrush[], px: number, py: number): boolean {
  return hitTestBrushSegment(ctx, w, h, pts, px, py) >= 0;
}

function drawArrow(ctx: CanvasRenderingContext2D, fromX: number, fromY: number, toX: number, toY: number) {
  const headLen = 10;
  const angle = Math.atan2(toY - fromY, toX - fromX);
  const wingAngle = 0.4;

  // Shaft
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.stroke();

  // Arrowhead wings
  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX - headLen * Math.cos(angle - wingAngle), toY - headLen * Math.sin(angle - wingAngle));
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX - headLen * Math.cos(angle + wingAngle), toY - headLen * Math.sin(angle + wingAngle));
  ctx.stroke();

  // Tail circle — filled
  ctx.beginPath();
  ctx.arc(fromX, fromY, 3, 0, Math.PI * 2);
  ctx.fillStyle = ctx.strokeStyle;
  ctx.fill();
  ctx.stroke();
}

function drawGradient(ctx: CanvasRenderingContext2D, w: number, h: number, pts: MaskPointsGradient, hovered = false, mx: number | null = null, my: number | null = null) {
  const ax = pts.anchor[0] * w;
  const ay = pts.anchor[1] * h;
  const rot = (pts.rotation * Math.PI) / 180;
  const comp = pts.compression;

  // Border offset: DT uses compression * diagonal (see _gradient_get_pts_border)
  const transitionHalf = comp * Math.sqrt(w * w + h * h);

  // Gradient direction: DT uses -(rotation-90°) for pivot, giving (sin, cos) in y-down coords
  const dx = Math.sin(rot);
  const dy = Math.cos(rot);

  // Line direction: DT uses v = -rotation, giving (cos, -sin)
  const lineLen = Math.max(w, h) * 1.5;
  const lx = Math.cos(rot);
  const ly = -Math.sin(rot);

  // Main gradient line — solid
  ctx.beginPath();
  ctx.moveTo(ax - lx * lineLen, ay - ly * lineLen);
  ctx.lineTo(ax + lx * lineLen, ay + ly * lineLen);
  dualStroke(ctx, false, hovered);

  // Border lines — dashed
  ctx.beginPath();
  ctx.moveTo(ax + dx * transitionHalf - lx * lineLen, ay + dy * transitionHalf - ly * lineLen);
  ctx.lineTo(ax + dx * transitionHalf + lx * lineLen, ay + dy * transitionHalf + ly * lineLen);
  dualStroke(ctx, true, hovered);

  ctx.beginPath();
  ctx.moveTo(ax - dx * transitionHalf - lx * lineLen, ay - dy * transitionHalf - ly * lineLen);
  ctx.lineTo(ax - dx * transitionHalf + lx * lineLen, ay - dy * transitionHalf + ly * lineLen);
  dualStroke(ctx, true, hovered);

  // Arrow — dual-stroke manually (dark bg then bright fg)
  const pivotDist = 0.1 * Math.min(w, h);
  const tailX = ax - dx * pivotDist, tailY = ay - dy * pivotDist;
  const headX = ax + dx * pivotDist, headY = ay + dy * pivotDist;
  const arrowLw = LW_MASK * (hovered ? LW_SEL_MULT : 1);
  ctx.lineWidth = arrowLw;
  ctx.strokeStyle = hovered ? DARK_SEL : DARK;
  drawArrow(ctx, tailX, tailY, headX, headY);
  ctx.lineWidth = hovered ? arrowLw : arrowLw / 2;
  ctx.strokeStyle = hovered ? BRIGHT_SEL : BRIGHT;
  drawArrow(ctx, tailX, tailY, headX, headY);

  // Anchor handle
  drawHandle(ctx, ax, ay, mx, my);
}

function drawGradientPolyline(ctx: CanvasRenderingContext2D, w: number, h: number, t: MaskTransformedGradient, hovered = false, mx: number | null = null, my: number | null = null) {
  const ax = t.anchor[0] * w;
  const ay = t.anchor[1] * h;

  // Main gradient line — solid polyline
  const mainPath = buildPolylinePath2D(t.main_polyline, w, h, false);
  dualStrokePath2D(ctx, mainPath, false, hovered);

  // Border lines — dashed polylines
  if (t.border_polyline1.length >= 4) {
    const b1 = buildPolylinePath2D(t.border_polyline1, w, h, false);
    dualStrokePath2D(ctx, b1, true, hovered);
  }
  if (t.border_polyline2.length >= 4) {
    const b2 = buildPolylinePath2D(t.border_polyline2, w, h, false);
    dualStrokePath2D(ctx, b2, true, hovered);
  }

  // Arrow — use transformed rotation
  const rot = (t.rotation * Math.PI) / 180;
  const dx = Math.sin(rot);
  const dy = Math.cos(rot);
  const pivotDist = 0.1 * Math.min(w, h);
  const tailX = ax - dx * pivotDist, tailY = ay - dy * pivotDist;
  const headX = ax + dx * pivotDist, headY = ay + dy * pivotDist;
  const arrowLw = LW_MASK * (hovered ? LW_SEL_MULT : 1);
  ctx.lineWidth = arrowLw;
  ctx.strokeStyle = hovered ? DARK_SEL : DARK;
  drawArrow(ctx, tailX, tailY, headX, headY);
  ctx.lineWidth = hovered ? arrowLw : arrowLw / 2;
  ctx.strokeStyle = hovered ? BRIGHT_SEL : BRIGHT;
  drawArrow(ctx, tailX, tailY, headX, headY);

  // Anchor handle
  drawHandle(ctx, ax, ay, mx, my);
}

function hitTestGradientPolyline(ctx: CanvasRenderingContext2D, w: number, h: number, t: MaskTransformedGradient, px: number, py: number): boolean {
  // Check if point is between the two border lines by testing stroke proximity
  ctx.lineWidth = 10;
  const mainPath = buildPolylinePath2D(t.main_polyline, w, h, false);
  if (ctx.isPointInStroke(mainPath, px, py)) return true;
  if (t.border_polyline1.length >= 4) {
    const b1 = buildPolylinePath2D(t.border_polyline1, w, h, false);
    if (ctx.isPointInStroke(b1, px, py)) return true;
  }
  if (t.border_polyline2.length >= 4) {
    const b2 = buildPolylinePath2D(t.border_polyline2, w, h, false);
    if (ctx.isPointInStroke(b2, px, py)) return true;
  }
  if (isNearHandle(t.anchor[0] * w, t.anchor[1] * h, px, py)) return true;
  return false;
}

function hitTestGradient(w: number, h: number, pts: MaskPointsGradient, px: number, py: number): boolean {
  const ax = pts.anchor[0] * w;
  const ay = pts.anchor[1] * h;
  const rot = (pts.rotation * Math.PI) / 180;
  const comp = pts.compression;
  const transitionHalf = comp * Math.sqrt(w * w + h * h);
  const dx = Math.sin(rot);
  const dy = Math.cos(rot);
  const relX = px - ax;
  const relY = py - ay;
  const dist = Math.abs(relX * dx + relY * dy);
  if (dist <= transitionHalf) return true;
  if (isNearHandle(ax, ay, px, py)) return true;
  return false;
}

function buildPathBezier(pts: MaskPointPath[], w: number, h: number): Path2D {
  const p = new Path2D();
  p.moveTo(pts[0].corner[0] * w, pts[0].corner[1] * h);
  for (let i = 0; i < pts.length; i++) {
    const curr = pts[i];
    const next = pts[(i + 1) % pts.length];
    p.bezierCurveTo(
      curr.ctrl2[0] * w, curr.ctrl2[1] * h,
      next.ctrl1[0] * w, next.ctrl1[1] * h,
      next.corner[0] * w, next.corner[1] * h,
    );
  }
  p.closePath();
  return p;
}

/** Border anchor position for a path point (for handles + hit testing) */
interface BorderAnchor { x: number; y: number }

/**
 * Evaluate cubic bezier position at parameter t.
 */
function bezierPos(p0x: number, p0y: number, p1x: number, p1y: number,
                   p2x: number, p2y: number, p3x: number, p3y: number,
                   t: number): [number, number] {
  const ti = 1 - t;
  const ti2 = ti * ti, ti3 = ti2 * ti;
  const t2 = t * t, t3 = t2 * t;
  return [
    ti3 * p0x + 3 * ti2 * t * p1x + 3 * ti * t2 * p2x + t3 * p3x,
    ti3 * p0y + 3 * ti2 * t * p1y + 3 * ti * t2 * p2y + t3 * p3y,
  ];
}

/**
 * Evaluate cubic bezier derivative at parameter t.
 */
function bezierDeriv(p0x: number, p0y: number, p1x: number, p1y: number,
                     p2x: number, p2y: number, p3x: number, p3y: number,
                     t: number): [number, number] {
  const ti = 1 - t;
  const ti2 = ti * ti, t2 = t * t, t_ti = t * ti;
  const a = 3 * ti2;
  const b = 3 * (ti2 - 2 * t_ti);
  const c = 3 * (2 * t_ti - t2);
  const d = 3 * t2;
  return [
    -p0x * a + p1x * b + p2x * c + p3x * d,
    -p0y * a + p1y * b + p2y * c + p3y * d,
  ];
}

/**
 * Compute border point at parameter t of a bezier segment, matching DT's _path_border_get_XY.
 * Offsets perpendicular to the derivative by rad (linearly interpolated).
 */
function borderPointAt(p0x: number, p0y: number, p1x: number, p1y: number,
                       p2x: number, p2y: number, p3x: number, p3y: number,
                       t: number, rad: number): [number, number] {
  const [cx, cy] = bezierPos(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, t);
  const [dx, dy] = bezierDeriv(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, t);
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 1e-10) return [cx, cy];
  return [cx + rad * dy / len, cy - rad * dx / len];
}

/**
 * Compute border anchor positions for each path point (at t=0 of outgoing segment).
 * Matches DT's gpt->border[k*6] positions.
 */
function computeBorderAnchors(pts: MaskPointPath[], w: number, h: number, cw: number): BorderAnchor[] {
  const dim = Math.min(w, h);
  return pts.map((p, i) => {
    const rad = cw * p.border[1] * dim;
    if (Math.abs(rad) < 0.5) return { x: p.corner[0] * w, y: p.corner[1] * h };
    const next = pts[(i + 1) % pts.length];
    // t=0 of segment: corner_k → ctrl2_k → ctrl1_{k+1} → corner_{k+1}
    const [bx, by] = borderPointAt(
      p.corner[0] * w, p.corner[1] * h,
      p.ctrl2[0] * w, p.ctrl2[1] * h,
      next.ctrl1[0] * w, next.ctrl1[1] * h,
      next.corner[0] * w, next.corner[1] * h,
      0, rad,
    );
    return { x: bx, y: by };
  });
}

/** Number of line segments to sample per bezier segment for the border polyline */
const BORDER_SAMPLES = 20;

/**
 * Build the border as a densely-sampled polyline Path2D, matching DT's approach.
 * For each bezier segment, sample border positions at multiple t values.
 */
function buildBorderPolyline(pts: MaskPointPath[], w: number, h: number, cw: number): Path2D {
  const dim = Math.min(w, h);
  const nb = pts.length;
  const p = new Path2D();
  let first = true;

  for (let k = 0; k < nb; k++) {
    const pt1 = pts[k];
    const pt2 = pts[(k + 1) % nb];
    // Segment bezier: p0=corner_k, p1=ctrl2_k, p2=ctrl1_{k+1}, p3=corner_{k+1}
    const p0x = pt1.corner[0] * w, p0y = pt1.corner[1] * h;
    const p1x = pt1.ctrl2[0] * w, p1y = pt1.ctrl2[1] * h;
    const p2x = pt2.ctrl1[0] * w, p2y = pt2.ctrl1[1] * h;
    const p3x = pt2.corner[0] * w, p3y = pt2.corner[1] * h;
    // DT: start rad = cw * pt1.border[1], end rad = cw * pt2.border[0]
    const radStart = cw * pt1.border[1] * dim;
    const radEnd = cw * pt2.border[0] * dim;

    for (let s = 0; s <= BORDER_SAMPLES; s++) {
      const t = s / BORDER_SAMPLES;
      const rad = radStart + (radEnd - radStart) * t;
      const [bx, by] = borderPointAt(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, t, rad);
      if (first) { p.moveTo(bx, by); first = false; }
      else p.lineTo(bx, by);
    }
  }
  p.closePath();
  return p;
}

/** Compute winding direction matching DT's _path_is_clockwise */
function pathWindingCW(pts: MaskPointPath[]): number {
  let area = 0;
  for (let i = 0; i < pts.length; i++) {
    const curr = pts[i].corner;
    const next = pts[(i + 1) % pts.length].corner;
    area += (next[0] - curr[0]) * (next[1] + curr[1]);
  }
  // DT: sum < 0 → clockwise → cw=1; sum >= 0 → cw=-1
  return area < 0 ? 1 : -1;
}

function hitTestPath(ctx: CanvasRenderingContext2D, w: number, h: number, pts: MaskPointPath[], px: number, py: number, serverBorderPolyline?: number[]): boolean {
  if (pts.length < 2) return false;

  // Check inside main bezier path
  const mainPath = buildPathBezier(pts, w, h);
  if (ctx.isPointInPath(mainPath, px, py)) return true;

  // Check inside border polyline path
  let borderPath: Path2D;
  if (serverBorderPolyline && serverBorderPolyline.length >= 4) {
    borderPath = buildPolylinePath2D(serverBorderPolyline, w, h, true);
  } else {
    const cw = pathWindingCW(pts);
    borderPath = buildBorderPolyline(pts, w, h, cw);
  }
  if (ctx.isPointInPath(borderPath, px, py)) return true;

  // Check handle proximity (corner)
  for (let i = 0; i < pts.length; i++) {
    if (isNearHandle(pts[i].corner[0] * w, pts[i].corner[1] * h, px, py)) return true;
  }
  return false;
}

/** Type guards for server-side polyline format */
function isTransformedPolyline(t: unknown): t is MaskTransformedCircle | MaskTransformedEllipse {
  const o = t as Record<string, unknown>;
  return t !== null && typeof t === "object" && "main_polyline" in o && !("border_polyline1" in o);
}

function isTransformedGradient(t: unknown): t is MaskTransformedGradient {
  const o = t as Record<string, unknown>;
  return t !== null && typeof t === "object" && "main_polyline" in o && "border_polyline1" in o;
}

function isTransformedPath(t: unknown): t is MaskTransformedPath {
  return t !== null && typeof t === "object" && "controls" in (t as Record<string, unknown>);
}

function isTransformedBrush(t: unknown): t is MaskTransformedBrush {
  return t !== null && typeof t === "object" && "border_polyline1" in (t as Record<string, unknown>);
}

function hitTestForm(ctx: CanvasRenderingContext2D, w: number, h: number, form: MaskForm, px: number, py: number): boolean {
  if (!form.points) return false;
  const baseType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);
  switch (baseType) {
    case MASKS_TYPE.CIRCLE: {
      const t = form.transformed;
      if (t && isTransformedPolyline(t)) return hitTestCirclePolyline(ctx, w, h, t as MaskTransformedCircle, px, py);
      return hitTestCircle(w, h, (t || form.points) as MaskPointsCircle, px, py);
    }
    case MASKS_TYPE.ELLIPSE: {
      const t = form.transformed;
      if (t && isTransformedPolyline(t)) return hitTestEllipsePolyline(ctx, w, h, t as MaskTransformedEllipse, px, py);
      return hitTestEllipse(w, h, (t || form.points) as MaskPointsEllipse, px, py);
    }
    case MASKS_TYPE.PATH: {
      const t = form.transformed;
      if (t && isTransformedPath(t)) {
        return hitTestPath(ctx, w, h, t.controls, px, py, t.border_polyline);
      }
      return hitTestPath(ctx, w, h, (t || form.points) as MaskPointPath[], px, py);
    }
    case MASKS_TYPE.BRUSH: {
      const t = form.transformed;
      if (t && isTransformedBrush(t)) {
        return hitTestBrush(ctx, w, h, t.controls, px, py);
      }
      return hitTestBrush(ctx, w, h, (t || form.points) as MaskPointBrush[], px, py);
    }
    case MASKS_TYPE.GRADIENT: {
      const t = form.transformed;
      if (t && isTransformedGradient(t)) return hitTestGradientPolyline(ctx, w, h, t, px, py);
      return hitTestGradient(w, h, (t || form.points) as MaskPointsGradient, px, py);
    }
    default:
      return false;
  }
}

function drawForm(ctx: CanvasRenderingContext2D, w: number, h: number, form: MaskForm, hovered = false, mx: number | null = null, my: number | null = null, editedPoint: { formid: number; index: number } | null = null, hoveredSeg = -1) {
  if (!form.points) return;
  const baseType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);

  switch (baseType) {
    case MASKS_TYPE.CIRCLE: {
      const t = form.transformed;
      if (t && isTransformedPolyline(t)) drawCirclePolyline(ctx, w, h, t as MaskTransformedCircle, hovered, mx, my);
      else drawCircle(ctx, w, h, (t || form.points) as MaskPointsCircle, hovered, mx, my);
      break;
    }
    case MASKS_TYPE.ELLIPSE: {
      const t = form.transformed;
      if (t && isTransformedPolyline(t)) drawEllipsePolyline(ctx, w, h, t as MaskTransformedEllipse, hovered, mx, my);
      else drawEllipse(ctx, w, h, (t || form.points) as MaskPointsEllipse, hovered, mx, my);
      break;
    }
    case MASKS_TYPE.PATH: {
      const editIdx = editedPoint?.formid === form.formid ? editedPoint.index : null;
      const t = form.transformed;
      if (t && isTransformedPath(t)) {
        drawPath(ctx, w, h, t.controls, hovered, mx, my, editIdx, t.border_polyline);
      } else {
        drawPath(ctx, w, h, (t || form.points) as MaskPointPath[], hovered, mx, my, editIdx);
      }
      break;
    }
    case MASKS_TYPE.BRUSH: {
      const t = form.transformed;
      if (t && isTransformedBrush(t)) {
        drawBrush(ctx, w, h, t.controls, hovered, mx, my, t.border_polyline1, t.border_polyline2, hoveredSeg);
      } else {
        drawBrush(ctx, w, h, (t || form.points) as MaskPointBrush[], hovered, mx, my, undefined, undefined, hoveredSeg);
      }
      break;
    }
    case MASKS_TYPE.GRADIENT: {
      const t = form.transformed;
      if (t && isTransformedGradient(t)) drawGradientPolyline(ctx, w, h, t, hovered, mx, my);
      else drawGradient(ctx, w, h, (t || form.points) as MaskPointsGradient, hovered, mx, my);
      break;
    }
  }
}

export default function MaskOverlay({ targetRef }: Props) {
  const showMasks = useDevelopStore((s) => s.showMasks);
  const maskForms = useDevelopStore((s) => s.maskForms);
  const maskUsage = useDevelopStore((s) => s.maskUsage);
  const selectedMaskId = useDevelopStore((s) => s.selectedMaskId);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hoveredIdRef = useRef<number | null>(null);
  const hoveredSegRef = useRef<number>(-1);
  const mouseRef = useRef<{ x: number; y: number } | null>(null);
  const editedPointRef = useRef<{ formid: number; index: number } | null>(null);

  useEffect(() => {
    const target = targetRef.current;
    const canvas = canvasRef.current;
    if (!target || !canvas) return;

    const selectedForm = selectedMaskId !== null
      ? maskForms.find((f) => f.formid === selectedMaskId) ?? null
      : null;

    const hasAnythingToDraw = (showMasks && maskForms.length > 0) || selectedForm;
    if (!hasAnythingToDraw) {
      canvas.style.display = "none";
      canvas.style.pointerEvents = "none";
      return;
    }

    canvas.style.pointerEvents = "auto";

    const drawableForms: MaskForm[] = [];
    if (showMasks) {
      const usedFormIds = new Set<number>();
      for (const u of maskUsage) {
        const group = maskForms.find((f) => f.formid === u.mask_id);
        if (group?.children) {
          for (const c of group.children) usedFormIds.add(c.formid);
        }
      }
      for (const form of maskForms) {
        if (usedFormIds.has(form.formid)) drawableForms.push(form);
      }
    }
    if (selectedForm && !drawableForms.some((f) => f.formid === selectedForm.formid)) {
      drawableForms.push(selectedForm);
    }

    const draw = () => {
      const tr = target.getBoundingClientRect();
      const container = target.closest(".preview-container");
      if (!container) return;
      const pr = container.getBoundingClientRect();

      // Compute the actual rendered content area for object-contain
      // The image canvas intrinsic size may differ from its CSS box
      const imgCanvas = target as HTMLCanvasElement;
      const iw = imgCanvas.width;
      const ih = imgCanvas.height;
      const cssW = tr.width;
      const cssH = tr.height;

      let contentW: number, contentH: number, contentX: number, contentY: number;
      if (iw > 0 && ih > 0) {
        // object-contain: scale preserving aspect ratio, centered
        const scale = Math.min(cssW / iw, cssH / ih);
        contentW = iw * scale;
        contentH = ih * scale;
        contentX = (tr.left - pr.left) + (cssW - contentW) / 2;
        contentY = (tr.top - pr.top) + (cssH - contentH) / 2;
      } else {
        contentW = cssW;
        contentH = cssH;
        contentX = tr.left - pr.left;
        contentY = tr.top - pr.top;
      }

      const w = Math.round(contentW);
      const h = Math.round(contentH);
      if (w === 0 || h === 0) return;

      canvas.style.display = "block";
      canvas.style.left = `${contentX}px`;
      canvas.style.top = `${contentY}px`;
      canvas.width = w;
      canvas.height = h;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, w, h);

      const m = mouseRef.current;
      const ep = editedPointRef.current;
      for (const form of drawableForms) {
        const hovered = hoveredIdRef.current === form.formid;
        const seg = hovered ? hoveredSegRef.current : -1;
        drawForm(ctx, w, h, form, hovered, m?.x ?? null, m?.y ?? null, ep, seg);
      }
    };

    const onMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      mouseRef.current = { x: px, y: py };
      const w = canvas.width;
      const h = canvas.height;

      let newHovered: number | null = null;
      let newSeg = -1;
      for (const form of drawableForms) {
        const ctx2 = canvas.getContext("2d");
        if (ctx2 && hitTestForm(ctx2, w, h, form, px, py)) {
          newHovered = form.formid;
          // Detect per-segment hover for brush masks
          const baseType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);
          if (baseType === MASKS_TYPE.BRUSH) {
            const t = form.transformed;
            const pts = (t && isTransformedBrush(t)) ? t.controls : (t || form.points) as MaskPointBrush[];
            newSeg = hitTestBrushSegment(ctx2, w, h, pts, px, py);
          }
          break;
        }
      }

      hoveredIdRef.current = newHovered;
      hoveredSegRef.current = newSeg;
      draw();
    };

    const onMouseLeave = () => {
      mouseRef.current = null;
      hoveredIdRef.current = null;
      hoveredSegRef.current = -1;
      draw();
    };

    const onClick = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const w = canvas.width;
      const h = canvas.height;

      // Check if a path corner handle was clicked
      for (const form of drawableForms) {
        const baseType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);
        if (baseType !== MASKS_TYPE.PATH || !form.points) continue;
        const pts = form.points as MaskPointPath[];
        for (let i = 0; i < pts.length; i++) {
          if (isNearHandle(pts[i].corner[0] * w, pts[i].corner[1] * h, px, py)) {
            const cur = editedPointRef.current;
            if (cur && cur.formid === form.formid && cur.index === i) {
              editedPointRef.current = null; // toggle off
            } else {
              editedPointRef.current = { formid: form.formid, index: i };
            }
            draw();
            return;
          }
        }
      }
      // Clicked elsewhere — clear edited point
      if (editedPointRef.current) {
        editedPointRef.current = null;
        draw();
      }
    };

    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mouseleave", onMouseLeave);
    canvas.addEventListener("pointerup", onClick);

    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(target);
    const mo = new MutationObserver(draw);
    mo.observe(target, { attributes: true, attributeFilter: ["style"] });
    return () => {
      ro.disconnect();
      mo.disconnect();
      canvas.removeEventListener("mousemove", onMouseMove);
      canvas.removeEventListener("mouseleave", onMouseLeave);
      canvas.removeEventListener("pointerup", onClick);
    };
  }, [showMasks, maskForms, maskUsage, selectedMaskId, targetRef]);

  return <canvas ref={canvasRef} className="mask-overlay" />;
}
