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
  DistortionGrid,
} from "../../types/protocol";
import { generateCirclePolyline, generateEllipsePolyline, transformPathPoints, transformBrushPoints, generateGradientPolylines, forwardTransform, inverseTransform } from "../../lib/distortionGrid";

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

function hitTestCirclePolyline(ctx: CanvasRenderingContext2D, w: number, h: number, t: MaskTransformedCircle, px: number, py: number): boolean {
  const borderPath = buildPolylinePath2D(t.border_polyline, w, h, true);
  if (ctx.isPointInPath(borderPath, px, py)) return true;
  if (t.main_polyline.length >= 2 && isNearHandle(t.main_polyline[0] * w, t.main_polyline[1] * h, px, py)) return true;
  if (t.border_polyline.length >= 2 && isNearHandle(t.border_polyline[0] * w, t.border_polyline[1] * h, px, py)) return true;
  return false;
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
  // Arrow endpoints (rotation handles)
  const rot = (t.rotation * Math.PI) / 180;
  const arrowDist = 0.1 * Math.min(w, h);
  const ax = t.anchor[0] * w, ay = t.anchor[1] * h;
  const headX = ax + Math.sin(rot) * arrowDist, headY = ay + Math.cos(rot) * arrowDist;
  const tailX = ax - Math.sin(rot) * arrowDist, tailY = ay - Math.cos(rot) * arrowDist;
  if (isNearHandle(headX, headY, px, py) || isNearHandle(tailX, tailY, px, py)) return true;
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

function hitTestForm(ctx: CanvasRenderingContext2D, w: number, h: number, form: MaskForm, px: number, py: number, grid: DistortionGrid | null = null): boolean {
  if (!form.points) return false;
  const baseType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);
  switch (baseType) {
    case MASKS_TYPE.CIRCLE: {
      if (!grid) return false;
      const pts = form.points as MaskPointsCircle;
      const poly = generateCirclePolyline(grid, pts.center, pts.radius, pts.border);
      return hitTestCirclePolyline(ctx, w, h, poly, px, py);
    }
    case MASKS_TYPE.ELLIPSE: {
      if (!grid) return false;
      const pts = form.points as MaskPointsEllipse;
      const poly = generateEllipsePolyline(grid, pts.center, pts.radius, pts.rotation, pts.border, pts.flags);
      return hitTestEllipsePolyline(ctx, w, h, poly, px, py);
    }
    case MASKS_TYPE.PATH: {
      if (!grid) return false;
      const pts = form.points as MaskPointPath[];
      const tPts = transformPathPoints(grid, pts);
      return hitTestPath(ctx, w, h, tPts, px, py);
    }
    case MASKS_TYPE.BRUSH: {
      if (!grid) return false;
      const pts = form.points as MaskPointBrush[];
      const tPts = transformBrushPoints(grid, pts);
      return hitTestBrush(ctx, w, h, tPts, px, py);
    }
    case MASKS_TYPE.GRADIENT: {
      if (!grid) return false;
      const pts = form.points as MaskPointsGradient;
      const tGrad = (form.transformed as MaskTransformedGradient | undefined)
        ?? { ...generateGradientPolylines(grid, pts.anchor, pts.rotation, pts.compression, pts.curvature),
            compression: pts.compression, steepness: pts.steepness, curvature: pts.curvature, state: pts.state };
      return hitTestGradientPolyline(ctx, w, h, tGrad, px, py);
    }
    default:
      return false;
  }
}

function drawForm(ctx: CanvasRenderingContext2D, w: number, h: number, form: MaskForm, hovered = false, mx: number | null = null, my: number | null = null, editedPoint: { formid: number; index: number } | null = null, hoveredSeg = -1, grid: DistortionGrid | null = null) {
  if (!form.points) return;
  const baseType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);

  switch (baseType) {
    case MASKS_TYPE.CIRCLE: {
      if (!grid) break;
      const pts = form.points as MaskPointsCircle;
      const poly = generateCirclePolyline(grid, pts.center, pts.radius, pts.border);
      drawCirclePolyline(ctx, w, h, poly, hovered, mx, my);
      break;
    }
    case MASKS_TYPE.ELLIPSE: {
      if (!grid) break;
      const pts = form.points as MaskPointsEllipse;
      const poly = generateEllipsePolyline(grid, pts.center, pts.radius, pts.rotation, pts.border, pts.flags);
      drawEllipsePolyline(ctx, w, h, poly, hovered, mx, my);
      break;
    }
    case MASKS_TYPE.PATH: {
      if (!grid) break;
      const editIdx = editedPoint?.formid === form.formid ? editedPoint.index : null;
      const pts = form.points as MaskPointPath[];
      const tPts = transformPathPoints(grid, pts);
      drawPath(ctx, w, h, tPts, hovered, mx, my, editIdx);
      break;
    }
    case MASKS_TYPE.BRUSH: {
      if (!grid) break;
      const pts = form.points as MaskPointBrush[];
      const tPts = transformBrushPoints(grid, pts);
      drawBrush(ctx, w, h, tPts, hovered, mx, my, undefined, undefined, hoveredSeg);
      break;
    }
    case MASKS_TYPE.GRADIENT: {
      if (!grid) break;
      const pts = form.points as MaskPointsGradient;
      const tGrad = (form.transformed as MaskTransformedGradient | undefined)
        ?? { ...generateGradientPolylines(grid, pts.anchor, pts.rotation, pts.compression, pts.curvature),
            compression: pts.compression, steepness: pts.steepness, curvature: pts.curvature, state: pts.state };
      drawGradientPolyline(ctx, w, h, tGrad, hovered, mx, my);
      break;
    }
  }
}

// Hit-test result for mask editing drag targets
type DragTarget =
  | { kind: "center"; formid: number }
  | { kind: "radius"; formid: number; pointIndex?: number }
  | { kind: "border"; formid: number; pointIndex?: number }
  | { kind: "rotate"; formid: number };

interface DragState {
  target: DragTarget;
  startX: number;
  startY: number;
  origCenter: [number, number];
  origRadius: number | [number, number];
  origBorder: number;
  dragAxis?: 0 | 1;
  lastAngle?: number;
  origCompression?: number;
}

function hitTestDragTarget(
  ctx: CanvasRenderingContext2D, w: number, h: number,
  form: MaskForm, px: number, py: number, grid: DistortionGrid | null = null,
): DragTarget | null {
  const baseType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);

  if (grid && form.points) {
    // Grid mode: generate polylines to get handle positions in output space
    if (baseType === MASKS_TYPE.CIRCLE) {
      const pts = form.points as MaskPointsCircle;
      const poly = generateCirclePolyline(grid, pts.center, pts.radius, pts.border);
      const cx = poly.center[0] * w, cy = poly.center[1] * h;
      if (isNearHandle(cx, cy, px, py)) return { kind: "center", formid: form.formid };
      // Radius handle: first point of main polyline (angle=0)
      if (poly.main_polyline.length >= 2) {
        if (isNearHandle(poly.main_polyline[0] * w, poly.main_polyline[1] * h, px, py))
          return { kind: "radius", formid: form.formid };
      }
      // Border handle: first point of border polyline
      if (poly.border_polyline.length >= 2) {
        if (isNearHandle(poly.border_polyline[0] * w, poly.border_polyline[1] * h, px, py))
          return { kind: "border", formid: form.formid };
      }
      // Grab anywhere inside the border to move (form_dragging)
      const borderPath = buildPolylinePath2D(poly.border_polyline, w, h, true);
      if (ctx.isPointInPath(borderPath, px, py)) return { kind: "center", formid: form.formid };
    } else if (baseType === MASKS_TYPE.ELLIPSE) {
      const pts = form.points as MaskPointsEllipse;
      const poly = generateEllipsePolyline(grid, pts.center, pts.radius, pts.rotation, pts.border, pts.flags);
      const cx = poly.center[0] * w, cy = poly.center[1] * h;
      if (isNearHandle(cx, cy, px, py)) return { kind: "center", formid: form.formid };
      // Radius handles at 4 axis points from polyline
      const nMain = poly.main_polyline.length / 2;
      const nBorder = poly.border_polyline.length / 2;
      for (let q = 0; q < 4; q++) {
        const mi = Math.round(q * nMain / 4) % nMain;
        if (isNearHandle(poly.main_polyline[mi * 2] * w, poly.main_polyline[mi * 2 + 1] * h, px, py))
          return { kind: "radius", formid: form.formid, pointIndex: q };
        const bi = Math.round(q * nBorder / 4) % nBorder;
        if (isNearHandle(poly.border_polyline[bi * 2] * w, poly.border_polyline[bi * 2 + 1] * h, px, py))
          return { kind: "border", formid: form.formid, pointIndex: q };
      }
      // Grab anywhere inside the border to move (form_dragging)
      const borderPath = buildPolylinePath2D(poly.border_polyline, w, h, true);
      if (ctx.isPointInPath(borderPath, px, py)) return { kind: "center", formid: form.formid };
    } else if (baseType === MASKS_TYPE.GRADIENT) {
      const pts = form.points as MaskPointsGradient;
      const poly = form.transformed
        ? form.transformed as MaskTransformedGradient
        : { ...generateGradientPolylines(grid, pts.anchor, pts.rotation, pts.compression, pts.curvature),
            compression: pts.compression, steepness: pts.steepness, curvature: pts.curvature, state: pts.state };
      const ax = poly.anchor[0] * w, ay = poly.anchor[1] * h;
      // Arrow endpoint handles for rotation
      const rot = (poly.rotation * Math.PI) / 180;
      const arrowDist = 0.1 * Math.min(w, h);
      const headX = ax + Math.sin(rot) * arrowDist, headY = ay + Math.cos(rot) * arrowDist;
      const tailX = ax - Math.sin(rot) * arrowDist, tailY = ay - Math.cos(rot) * arrowDist;
      if (isNearHandle(headX, headY, px, py) || isNearHandle(tailX, tailY, px, py))
        return { kind: "rotate", formid: form.formid };
      // Anchor handle — move
      if (isNearHandle(ax, ay, px, py)) return { kind: "center", formid: form.formid };
      // Border lines — compression drag
      const b1Path = buildPolylinePath2D(poly.border_polyline1, w, h, false);
      const b2Path = buildPolylinePath2D(poly.border_polyline2, w, h, false);
      ctx.lineWidth = 16;
      if (ctx.isPointInStroke(b1Path, px, py) || ctx.isPointInStroke(b2Path, px, py))
        return { kind: "border", formid: form.formid };
      // Main line — drag to move
      const mainPath = buildPolylinePath2D(poly.main_polyline, w, h, false);
      if (ctx.isPointInStroke(mainPath, px, py))
        return { kind: "center", formid: form.formid };
    }
    return null;
  }

  return null;
}

export default function MaskOverlay({ targetRef }: Props) {
  const distortionGrid = useDevelopStore((s) => s.distortionGrid);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hoveredIdRef = useRef<number | null>(null);
  const hoveredSegRef = useRef<number>(-1);
  const mouseRef = useRef<{ x: number; y: number } | null>(null);
  const editedPointRef = useRef<{ formid: number; index: number } | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const drawableFormsRef = useRef<MaskForm[]>([]);
  const drawRef = useRef<(() => void) | null>(null);
  // Normalized cursor position during creation — used by subscription to translate new server polylines
  const creationCursorRef = useRef<[number, number] | null>(null);

  // Recompute drawable forms when structure changes
  useEffect(() => {
    const computeDrawable = () => {
      const forms = useDevelopStore.getState().maskForms;
      const usage = useDevelopStore.getState().maskUsage;
      const selId = useDevelopStore.getState().selectedMaskId;
      const creating = useDevelopStore.getState().creatingMaskId;
      const show = useDevelopStore.getState().showMasks;

      const result: MaskForm[] = [];
      if (show) {
        const usedFormIds = new Set<number>();
        for (const u of usage) {
          const group = forms.find((f) => f.formid === u.mask_id);
          if (group?.children) {
            for (const c of group.children) usedFormIds.add(c.formid);
          }
        }
        for (const form of forms) {
          if (usedFormIds.has(form.formid)) result.push(form);
        }
      }
      const selForm = selId !== null ? forms.find((f) => f.formid === selId) ?? null : null;
      if (selForm && !result.some((f) => f.formid === selForm.formid)) {
        result.push(selForm);
      }
      if (creating !== null && !result.some((f) => f.formid === creating)) {
        const creatingForm = forms.find((f) => f.formid === creating);
        if (creatingForm) result.push(creatingForm);
      }
      drawableFormsRef.current = result;
    };
    computeDrawable();
    // Subscribe to store changes for lightweight redraw (no event listener teardown)
    const unsub = useDevelopStore.subscribe((state, prev) => {
      if (state.maskForms !== prev.maskForms || state.selectedMaskId !== prev.selectedMaskId
          || state.creatingMaskId !== prev.creatingMaskId || state.showMasks !== prev.showMasks) {
        computeDrawable();
        // During creation, when server returns new polylines (e.g. from slider preview),
        // update center to match cursor so next draw is correct
        if (state.creatingMaskId !== null && state.maskForms !== prev.maskForms) {
          const cursor = creationCursorRef.current;
          // If mouse is outside image area, center the mask at (0.5, 0.5)
          const outPos: [number, number] = cursor ?? [0.5, 0.5];
          const grid = useDevelopStore.getState().distortionGrid;
          if (grid) {
            const form = drawableFormsRef.current.find((f) => f.formid === state.creatingMaskId);
            if (form) {
              const rawCenter = inverseTransform(grid, outPos[0], outPos[1]);
              const bType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);
              if (bType === MASKS_TYPE.GRADIENT) {
                const gpts = form.points as MaskPointsGradient | undefined;
                if (gpts) { gpts.anchor[0] = rawCenter[0]; gpts.anchor[1] = rawCenter[1]; }
              } else {
                const pts = form.points as { center: [number, number] } | undefined;
                if (pts) { pts.center[0] = rawCenter[0]; pts.center[1] = rawCenter[1]; }
              }
            }
          }
        }
        drawRef.current?.();
      }
    });
    return unsub;
  }, []); // stable — reads from store directly

  useEffect(() => {
    const target = targetRef.current;
    const canvas = canvasRef.current;
    if (!target || !canvas) return;

    const draw = () => {
      // Check if there's anything to draw (read latest state)
      const { maskForms, selectedMaskId: selId, showMasks: show,
              creationTool: cTool, creatingMaskId: cMaskId } = useDevelopStore.getState();
      const selForm = selId !== null ? maskForms.find((f) => f.formid === selId) ?? null : null;
      const hasAnythingToDraw = (show && drawableFormsRef.current.length > 0) || selForm || cTool || cMaskId;
      if (!hasAnythingToDraw) {
        canvas.style.display = "none";
        canvas.style.pointerEvents = "none";
        return;
      }
      canvas.style.pointerEvents = "auto";
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
      for (const form of drawableFormsRef.current) {
        const hovered = hoveredIdRef.current === form.formid;
        const seg = hovered ? hoveredSegRef.current : -1;
        drawForm(ctx, w, h, form, hovered, m?.x ?? null, m?.y ?? null, ep, seg, distortionGrid);
      }

      // Compute cursor from current state — single source of truth
      if (dragRef.current) {
        const dk = dragRef.current.target.kind;
        canvas.style.cursor = dk === "center" ? "grabbing" : dk === "rotate" ? "alias" : "crosshair";
      } else if (cTool || cMaskId) {
        canvas.style.cursor = "crosshair";
      } else if (m && selId !== null) {
        const selF = drawableFormsRef.current.find((f) => f.formid === selId);
        if (selF) {
          const hit = hitTestDragTarget(ctx, w, h, selF, m.x, m.y, distortionGrid);
          canvas.style.cursor = hit ? (hit.kind === "center" ? "grab" : hit.kind === "rotate" ? "alias" : "crosshair") : "";
        } else {
          canvas.style.cursor = "";
        }
      } else {
        canvas.style.cursor = "";
      }
    };

    drawRef.current = draw;

    const { createMask, updateMask, resetCreation, requestPreview,
            saveCreation, cancelCreation, previewMaskParam } = useDevelopStore.getState();

    const onMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      mouseRef.current = { x: px, y: py };
      const w = canvas.width;
      const h = canvas.height;

      // Creation mode: update form.points.center to follow cursor
      const currentCreatingId = useDevelopStore.getState().creatingMaskId;
      if (currentCreatingId !== null) {
        const cx = px / w;
        const cy = py / h;
        creationCursorRef.current = [cx, cy];
        const grid = useDevelopStore.getState().distortionGrid;
        if (grid) {
          const form = drawableFormsRef.current.find((f) => f.formid === currentCreatingId);
          if (form) {
            const rawPos = inverseTransform(grid, cx, cy);
            const bType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);
            if (bType === MASKS_TYPE.GRADIENT) {
              const gpts = form.points as MaskPointsGradient | undefined;
              if (gpts) { gpts.anchor[0] = rawPos[0]; gpts.anchor[1] = rawPos[1]; }
            } else {
              const pts = form.points as { center: [number, number] } | undefined;
              if (pts) { pts.center[0] = rawPos[0]; pts.center[1] = rawPos[1]; }
            }
          }
          // Background server sync — send raw-space position
          const rawSync = inverseTransform(grid, cx, cy);
          const bType2 = form?.type ? (form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE)) : 0;
          if (bType2 === MASKS_TYPE.GRADIENT) {
            previewMaskParam(currentCreatingId, { anchor: rawSync });
          } else {
            previewMaskParam(currentCreatingId, { center: rawSync });
          }
        }
        draw();
        return;
      }

      // Handle drag in progress (requires distortion grid for raw-space transforms)
      const drag = dragRef.current;
      if (drag) {
        const grid = useDevelopStore.getState().distortionGrid;
        if (!grid) { draw(); return; }
        const form = drawableFormsRef.current.find((f) => f.formid === drag.target.formid);
        if (!form) return;
        const baseType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);
        const pts = form.points as MaskPointsCircle | MaskPointsEllipse;
        const rawMouse = inverseTransform(grid, px / w, py / h);
        const iw = grid.iwidth, ih = grid.iheight;
        const dim = Math.min(iw, ih);

        if (drag.target.kind === "center") {
          const rawStart = inverseTransform(grid, drag.startX / w, drag.startY / h);
          const newX = drag.origCenter[0] + (rawMouse[0] - rawStart[0]);
          const newY = drag.origCenter[1] + (rawMouse[1] - rawStart[1]);
          if (baseType === MASKS_TYPE.GRADIENT) {
            const gp = form.points as MaskPointsGradient;
            gp.anchor[0] = newX; gp.anchor[1] = newY;
          } else {
            pts.center[0] = newX; pts.center[1] = newY;
          }
        } else if (drag.target.kind === "radius") {
          const dx = (rawMouse[0] - pts.center[0]) * iw;
          const dy = (rawMouse[1] - pts.center[1]) * ih;
          if (baseType === MASKS_TYPE.CIRCLE) {
            (pts as MaskPointsCircle).radius = Math.max(0.001, Math.sqrt(dx * dx + dy * dy) / dim);
          } else {
            // DT approach: one axis at a time, axis locked at drag start
            const ep = pts as MaskPointsEllipse;
            const rot = (ep.rotation ?? 0) * Math.PI / 180;
            if (drag.dragAxis === 0) {
              const proj = Math.abs(dx * Math.cos(rot) + dy * Math.sin(rot)) / dim;
              ep.radius[0] = Math.max(0.002, proj);
            } else {
              const proj = Math.abs(-dx * Math.sin(rot) + dy * Math.cos(rot)) / dim;
              ep.radius[1] = Math.max(0.002, proj);
            }
          }
        } else if (drag.target.kind === "border") {
          if (baseType === MASKS_TYPE.GRADIENT) {
            // Compression drag: distance from mouse to anchor along gradient direction
            const gp = form.points as MaskPointsGradient;
            const rot = (gp.rotation * Math.PI) / 180;
            const gdx = Math.sin(rot), gdy = Math.cos(rot);
            const dx = (rawMouse[0] - gp.anchor[0]) * iw;
            const dy = (rawMouse[1] - gp.anchor[1]) * ih;
            const diag = Math.sqrt(iw * iw + ih * ih);
            const proj = Math.abs(dx * gdx + dy * gdy);
            gp.compression = Math.max(0.001, Math.min(1, proj / diag));
          } else {
            const dx = (rawMouse[0] - pts.center[0]) * iw;
            const dy = (rawMouse[1] - pts.center[1]) * ih;
            if (baseType === MASKS_TYPE.CIRCLE) {
              const cp = pts as MaskPointsCircle;
              cp.border = Math.max(0.001, Math.sqrt(dx * dx + dy * dy) / dim - cp.radius);
            } else {
              const ep = pts as MaskPointsEllipse;
              const rot = (ep.rotation ?? 0) * Math.PI / 180;
              if (drag.dragAxis === 0) {
                const proj = Math.abs(dx * Math.cos(rot) + dy * Math.sin(rot)) / dim;
                ep.border = Math.max(0.001, proj - ep.radius[0]);
              } else {
                const proj = Math.abs(-dx * Math.sin(rot) + dy * Math.cos(rot)) / dim;
                ep.border = Math.max(0.001, proj - ep.radius[1]);
              }
            }
          }
        } else if (drag.target.kind === "rotate") {
          // Delta-based rotation (matches DT's Ctrl+drag)
          const refPt = baseType === MASKS_TYPE.GRADIENT
            ? (form.points as MaskPointsGradient).anchor
            : (pts as MaskPointsEllipse).center;
          const tc = forwardTransform(grid, refPt[0], refPt[1]);
          const curAngle = Math.atan2(py / h - tc[1], px / w - tc[0]);
          const delta = (curAngle - (drag.lastAngle ?? curAngle)) * (180 / Math.PI);
          if (baseType === MASKS_TYPE.GRADIENT) {
            const gp = form.points as MaskPointsGradient;
            gp.rotation = ((gp.rotation ?? 0) - delta) % 360;
            if (gp.rotation < 0) gp.rotation += 360;
          } else {
            const ep = pts as MaskPointsEllipse;
            ep.rotation = ((ep.rotation ?? 0) + delta) % 360;
            if (ep.rotation < 0) ep.rotation += 360;
          }
          drag.lastAngle = curAngle;
        }
        // Live server sync for slider feedback — only send changed params
        const dk = drag.target.kind;
        if (baseType === MASKS_TYPE.CIRCLE) {
          const cp = pts as MaskPointsCircle;
          if (dk === "center") previewMaskParam(drag.target.formid, { center: [cp.center[0], cp.center[1]] });
          else if (dk === "radius") previewMaskParam(drag.target.formid, { radius: cp.radius });
          else if (dk === "border") previewMaskParam(drag.target.formid, { border: cp.border });
        } else if (baseType === MASKS_TYPE.ELLIPSE) {
          const ep = pts as MaskPointsEllipse;
          if (dk === "center") previewMaskParam(drag.target.formid, { center: [ep.center[0], ep.center[1]] });
          else if (dk === "radius") previewMaskParam(drag.target.formid, { radius: [ep.radius[0], ep.radius[1]] });
          else if (dk === "border") previewMaskParam(drag.target.formid, { border: ep.border });
          else if (dk === "rotate") previewMaskParam(drag.target.formid, { rotation: ep.rotation });
        } else if (baseType === MASKS_TYPE.GRADIENT) {
          const gp = form.points as MaskPointsGradient;
          if (dk === "center") previewMaskParam(drag.target.formid, { anchor: [gp.anchor[0], gp.anchor[1]] });
          else if (dk === "border") previewMaskParam(drag.target.formid, { compression: gp.compression });
          else if (dk === "rotate") previewMaskParam(drag.target.formid, { rotation: gp.rotation });
        }
        draw();
        return;
      }

      // Normal hover detection
      let newHovered: number | null = null;
      let newSeg = -1;
      for (const form of drawableFormsRef.current) {
        const ctx2 = canvas.getContext("2d");
        if (ctx2 && hitTestForm(ctx2, w, h, form, px, py, distortionGrid)) {
          newHovered = form.formid;
          const baseType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);
          if (baseType === MASKS_TYPE.BRUSH && distortionGrid && form.points) {
            const tPts = transformBrushPoints(distortionGrid, form.points as MaskPointBrush[]);
            newSeg = hitTestBrushSegment(ctx2, w, h, tPts, px, py);
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
      // During creation, center the mask when mouse leaves the image area
      const currentCreatingId = useDevelopStore.getState().creatingMaskId;
      if (currentCreatingId !== null) {
        creationCursorRef.current = null;
        const grid = useDevelopStore.getState().distortionGrid;
        if (grid) {
          const form = drawableFormsRef.current.find((f) => f.formid === currentCreatingId);
          if (form) {
            const rawCenter = inverseTransform(grid, 0.5, 0.5);
            const bType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);
            if (bType === MASKS_TYPE.GRADIENT) {
              const gpts = form.points as MaskPointsGradient | undefined;
              if (gpts) { gpts.anchor[0] = rawCenter[0]; gpts.anchor[1] = rawCenter[1]; }
              previewMaskParam(currentCreatingId, { anchor: rawCenter });
            } else {
              const pts = form.points as { center: [number, number] } | undefined;
              if (pts) { pts.center[0] = rawCenter[0]; pts.center[1] = rawCenter[1]; }
              previewMaskParam(currentCreatingId, { center: rawCenter });
            }
          }
        }
      }
      draw();
    };

    const onMouseDown = (e: MouseEvent) => {
      if (e.button !== 0) return;
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const w = canvas.width;
      const h = canvas.height;

      // Creation mode: click to save mask position
      const currentCreatingId = useDevelopStore.getState().creatingMaskId;
      if (currentCreatingId !== null) {
        const cx = px / w;
        const cy = py / h;
        creationCursorRef.current = [cx, cy];
        // Server expects raw-space position (center for circle/ellipse, anchor for gradient)
        const grid = useDevelopStore.getState().distortionGrid;
        const rawPos = grid ? inverseTransform(grid, cx, cy) as [number, number] : [cx, cy] as [number, number];
        saveCreation(rawPos);
        return;
      }

      // Creation mode (from blending toolbar): click to create mask
      const tool = useDevelopStore.getState().creationTool;
      const mod = useDevelopStore.getState().creationModule;
      if (tool && mod) {
        const outCx = px / w;
        const outCy = py / h;
        // Server expects raw-space center — convert from output space if grid available
        const grid = useDevelopStore.getState().distortionGrid;
        const center = grid ? inverseTransform(grid, outCx, outCy) : [outCx, outCy];
        const params: Record<string, unknown> = {
          op: mod.op,
          instance: mod.instance,
        };
        if (tool === "gradient") {
          params.anchor = center;
          params.rotation = 0;
          params.compression = 0.05;
          params.steepness = 0;
          params.curvature = 0;
          params.state = 2; // DT_MASKS_GRADIENT_STATE_SIGMOIDAL
        } else {
          params.center = center;
          if (tool === "circle") {
            params.radius = 0.05;
            params.border = 0.025;
          } else {
            params.radius = [0.05, 0.05];
            params.border = 0.025;
          }
        }
        resetCreation();
        createMask(tool, params).then((formid) => {
          if (formid) requestPreview();
        });
        return;
      }

      // Try to start a drag on a form — returns true if drag started
      const tryStartDrag = (form: MaskForm, ctx2: CanvasRenderingContext2D): boolean => {
        const target = hitTestDragTarget(ctx2, w, h, form, px, py, distortionGrid);
        if (!target) return false;
        const baseType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);
        if (baseType === MASKS_TYPE.CIRCLE) {
          const c = form.points as MaskPointsCircle;
          dragRef.current = {
            target, startX: px, startY: py,
            origCenter: [c.center[0], c.center[1]],
            origRadius: c.radius, origBorder: c.border,
          };
        } else if (baseType === MASKS_TYPE.ELLIPSE) {
          const el = form.points as MaskPointsEllipse;
          // Lock axis at drag start: polyline q=0,2 → longer axis, q=1,3 → shorter
          const pi = (target as { pointIndex?: number }).pointIndex ?? 0;
          const isLongerAxis = (pi === 0 || pi === 2);
          const swapped = el.radius[0] < el.radius[1];
          const dragAxis: 0 | 1 = (isLongerAxis !== swapped) ? 0 : 1;
          dragRef.current = {
            target, startX: px, startY: py,
            origCenter: [el.center[0], el.center[1]],
            origRadius: [el.radius[0], el.radius[1]], origBorder: el.border,
            dragAxis,
          };
        } else if (baseType === MASKS_TYPE.GRADIENT) {
          const g = form.points as MaskPointsGradient;
          let startAngle: number | undefined;
          if (target.kind === "rotate" && distortionGrid) {
            const tc = forwardTransform(distortionGrid, g.anchor[0], g.anchor[1]);
            startAngle = Math.atan2(py / h - tc[1], px / w - tc[0]);
          }
          dragRef.current = {
            target, startX: px, startY: py,
            origCenter: [g.anchor[0], g.anchor[1]],
            origRadius: 0, origBorder: 0,
            origCompression: g.compression,
            lastAngle: startAngle,
          };
        } else {
          return false;
        }
        return true;
      };

      // Editing mode: check for drag targets on the selected mask
      const ctx2 = canvas.getContext("2d");
      const currentSelectedId = useDevelopStore.getState().selectedMaskId;
      if (ctx2 && currentSelectedId !== null) {
        const selForm = drawableFormsRef.current.find((f) => f.formid === currentSelectedId);
        if (selForm) {
          const baseType = selForm.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);
          // Ctrl+click on ellipse or gradient → rotation mode
          if (e.ctrlKey && (baseType === MASKS_TYPE.ELLIPSE || baseType === MASKS_TYPE.GRADIENT) && hitTestForm(ctx2, w, h, selForm, px, py, distortionGrid)) {
            const grid = useDevelopStore.getState().distortionGrid;
            if (grid) {
              // Get the reference point (center for ellipse, anchor for gradient)
              const refPt = baseType === MASKS_TYPE.GRADIENT
                ? (selForm.points as MaskPointsGradient).anchor
                : (selForm.points as MaskPointsEllipse).center;
              const tc = forwardTransform(grid, refPt[0], refPt[1]);
              const startAngle = Math.atan2(py / h - tc[1], px / w - tc[0]);
              dragRef.current = {
                target: { kind: "rotate", formid: selForm.formid },
                startX: px, startY: py,
                origCenter: [refPt[0], refPt[1]],
                origRadius: 0, origBorder: 0,
                lastAngle: startAngle,
              };
              e.preventDefault();
              return;
            }
          }
          if (tryStartDrag(selForm, ctx2)) {
            e.preventDefault();
            return;
          }
        }
      }

      // Click on a form to select it — also start drag if inside border
      if (ctx2) {
        for (const form of drawableFormsRef.current) {
          if (hitTestForm(ctx2, w, h, form, px, py, distortionGrid)) {
            useDevelopStore.getState().selectMask(form.formid);
            if (tryStartDrag(form, ctx2)) {
              e.preventDefault();
              return;
            }
            draw();
            return;
          }
        }
        // Clicked on empty space — deselect
        if (currentSelectedId !== null) {
          useDevelopStore.getState().selectMask(null);
          draw();
        }
      }
    };

    const onMouseUp = (e: MouseEvent) => {
      const drag = dragRef.current;
      if (!drag) {
        // Path corner handle toggle on click (only if not drag)
        const rect = canvas.getBoundingClientRect();
        const px = e.clientX - rect.left;
        const py = e.clientY - rect.top;
        const w = canvas.width;
        const h = canvas.height;
        for (const form of drawableFormsRef.current) {
          const baseType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);
          if (baseType !== MASKS_TYPE.PATH || !form.points) continue;
          const pts = form.points as MaskPointPath[];
          for (let i = 0; i < pts.length; i++) {
            if (isNearHandle(pts[i].corner[0] * w, pts[i].corner[1] * h, px, py)) {
              const cur = editedPointRef.current;
              if (cur && cur.formid === form.formid && cur.index === i) {
                editedPointRef.current = null;
              } else {
                editedPointRef.current = { formid: form.formid, index: i };
              }
              draw();
              return;
            }
          }
        }
        if (editedPointRef.current) {
          editedPointRef.current = null;
          draw();
        }
        return;
      }

      // Commit the drag to the server
      dragRef.current = null;
      const form = drawableFormsRef.current.find((f) => f.formid === drag.target.formid);
      if (!form) return;
      const baseType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);

      if (baseType === MASKS_TYPE.CIRCLE) {
        const pts = form.points as MaskPointsCircle;
        updateMask(form.formid, {
          center: [pts.center[0], pts.center[1]],
          radius: pts.radius,
          border: pts.border,
        }).then(() => requestPreview());
      } else if (baseType === MASKS_TYPE.ELLIPSE) {
        const pts = form.points as MaskPointsEllipse;
        updateMask(form.formid, {
          center: [pts.center[0], pts.center[1]],
          radius: [pts.radius[0], pts.radius[1]],
          border: pts.border,
          rotation: pts.rotation,
        }).then(() => requestPreview());
      } else if (baseType === MASKS_TYPE.GRADIENT) {
        const gp = form.points as MaskPointsGradient;
        updateMask(form.formid, {
          anchor: [gp.anchor[0], gp.anchor[1]],
          rotation: gp.rotation,
          compression: gp.compression,
          steepness: gp.steepness,
          curvature: gp.curvature,
          state: gp.state,
        }).then(() => requestPreview());
      }
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        const creating = useDevelopStore.getState().creatingMaskId;
        if (creating) {
          creationCursorRef.current = null;
          cancelCreation();
          return;
        }
        const tool = useDevelopStore.getState().creationTool;
        if (tool) {
          resetCreation();
        }
      }
    };

    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mouseleave", onMouseLeave);
    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mouseup", onMouseUp);
    window.addEventListener("keydown", onKeyDown);

    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(target);
    const mo2 = new MutationObserver(draw);
    mo2.observe(target, { attributes: true, attributeFilter: ["style"] });
    return () => {
      drawRef.current = null;
      ro.disconnect();
      mo2.disconnect();
      canvas.removeEventListener("mousemove", onMouseMove);
      canvas.removeEventListener("mouseleave", onMouseLeave);
      canvas.removeEventListener("mousedown", onMouseDown);
      canvas.removeEventListener("mouseup", onMouseUp);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [targetRef, distortionGrid]);

  return <canvas ref={canvasRef} className="mask-overlay" />;
}
