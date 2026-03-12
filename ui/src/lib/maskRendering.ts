import { MASKS_TYPE } from "../types/protocol";
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
} from "../types/protocol";
import { generateCirclePolyline, generateEllipsePolyline, transformPathPoints, transformBrushPoints, generateGradientPolylines } from "./distortionGrid";
import type { DragTarget } from "./maskDrag";
import { brushCtrl2ToFeather } from "./maskDrag";

// DT dual-stroke: dark background stroke + bright foreground stroke
const DARK = "rgba(40, 40, 40, 0.5)";
const DARK_SEL = "rgba(40, 40, 40, 0.8)";
const BRIGHT = "rgba(200, 200, 200, 0.6)";
const BRIGHT_SEL = "rgba(218, 218, 218, 0.9)";
const MASK_HANDLE_SIZE = 6;
export const HANDLE_HIT_RADIUS = 12;

// Line widths matching DT
const LW_MASK = 1.7;
const LW_BORDER = 1.0;
const LW_SEL_MULT = 1.5;

/** Dual-stroke a pre-built path: dark bg then bright fg */
export function dualStroke(ctx: CanvasRenderingContext2D, border: boolean, selected: boolean) {
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
export function dualStrokePath2D(ctx: CanvasRenderingContext2D, path: Path2D, border: boolean, selected: boolean) {
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

export function isNearHandle(hx: number, hy: number, mx: number | null, my: number | null): boolean {
  if (mx === null || my === null) return false;
  const dx = hx - mx, dy = hy - my;
  return dx * dx + dy * dy <= HANDLE_HIT_RADIUS * HANDLE_HIT_RADIUS;
}

export function drawHandle(ctx: CanvasRenderingContext2D, x: number, y: number, mx: number | null, my: number | null) {
  const near = isNearHandle(x, y, mx, my);
  const hs = near ? MASK_HANDLE_SIZE * 1.5 : MASK_HANDLE_SIZE;
  ctx.fillStyle = "rgba(200, 200, 200, 0.9)";
  ctx.strokeStyle = "rgba(40, 40, 40, 0.8)";
  ctx.lineWidth = 1;
  ctx.fillRect(x - hs / 2, y - hs / 2, hs, hs);
  ctx.strokeRect(x - hs / 2, y - hs / 2, hs, hs);
}

export function drawCirclePolyline(ctx: CanvasRenderingContext2D, w: number, h: number, t: MaskTransformedCircle, hovered = false, mx: number | null = null, my: number | null = null) {
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

export function hitTestCirclePolyline(ctx: CanvasRenderingContext2D, w: number, h: number, t: MaskTransformedCircle, px: number, py: number): boolean {
  const borderPath = buildPolylinePath2D(t.border_polyline, w, h, true);
  if (ctx.isPointInPath(borderPath, px, py)) return true;
  if (t.main_polyline.length >= 2 && isNearHandle(t.main_polyline[0] * w, t.main_polyline[1] * h, px, py)) return true;
  if (t.border_polyline.length >= 2 && isNearHandle(t.border_polyline[0] * w, t.border_polyline[1] * h, px, py)) return true;
  return false;
}

export function drawEllipsePolyline(ctx: CanvasRenderingContext2D, w: number, h: number, t: MaskTransformedEllipse, hovered = false, mx: number | null = null, my: number | null = null) {
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

export function hitTestEllipsePolyline(ctx: CanvasRenderingContext2D, w: number, h: number, t: MaskTransformedEllipse, px: number, py: number): boolean {
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

export function drawCtrlHandle(ctx: CanvasRenderingContext2D, x: number, y: number, mx: number | null, my: number | null) {
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
export function buildPolylinePath2D(polyline: number[], w: number, h: number, close: boolean): Path2D {
  const p = new Path2D();
  if (polyline.length < 4) return p;
  p.moveTo(polyline[0] * w, polyline[1] * h);
  for (let i = 2; i < polyline.length; i += 2) {
    p.lineTo(polyline[i] * w, polyline[i + 1] * h);
  }
  if (close) p.closePath();
  return p;
}

export function drawPath(ctx: CanvasRenderingContext2D, w: number, h: number, pts: MaskPointPath[], hovered = false, mx: number | null = null, my: number | null = null, editedIdx: number | null = null, serverBorderPolyline?: number[]) {
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
export function buildBrushBorderOutline(pts: MaskPointBrush[], w: number, h: number): Path2D {
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
      const rad = smoothstep(radStart, radEnd, t);
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

/**
 * Compute brush border anchor positions for each point.
 * Border anchors are perpendicular to the tangent at t=0 of the outgoing segment.
 */
export function computeBrushBorderAnchors(pts: MaskPointBrush[], w: number, h: number): BorderAnchor[] {
  const dim = Math.min(w, h);
  return pts.map((p, i) => {
    const rad = p.border[1] * dim;
    if (Math.abs(rad) < 0.5 || i >= pts.length - 1) {
      // Last point has no outgoing segment — use incoming tangent
      if (i > 0 && i === pts.length - 1) {
        const prev = pts[i - 1];
        const [bx, by] = borderPointAt(
          prev.corner[0] * w, prev.corner[1] * h,
          prev.ctrl2[0] * w, prev.ctrl2[1] * h,
          p.ctrl1[0] * w, p.ctrl1[1] * h,
          p.corner[0] * w, p.corner[1] * h,
          1, rad,
        );
        return { x: bx, y: by };
      }
      return { x: p.corner[0] * w, y: p.corner[1] * h };
    }
    const next = pts[i + 1];
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

/**
 * Compute feather handle positions for each brush point.
 * Feather handles are derived from ctrl2 via perpendicular rotation.
 */
export function computeBrushFeatherAnchors(pts: MaskPointBrush[], w: number, h: number): BorderAnchor[] {
  return pts.map((p) => {
    const [fx, fy] = brushCtrl2ToFeather(
      p.corner[0] * w, p.corner[1] * h,
      p.ctrl2[0] * w, p.ctrl2[1] * h,
      true,
    );
    return { x: fx, y: fy };
  });
}

export function drawBrush(ctx: CanvasRenderingContext2D, w: number, h: number, pts: MaskPointBrush[], hovered = false, mx: number | null = null, my: number | null = null, serverPolyline1?: number[], serverPolyline2?: number[], hoveredSeg = -1, editedIdx: number | null = null) {
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

  // Feather handle for edited point only (matching DT: one feather handle shown when point_edited >= 0)
  if (editedIdx !== null && editedIdx >= 0 && editedIdx < pts.length) {
    const featherAnchors = computeBrushFeatherAnchors(pts, w, h);
    const cornerX = pts[editedIdx].corner[0] * w;
    const cornerY = pts[editedIdx].corner[1] * h;
    const fa = featherAnchors[editedIdx];

    // Connecting line from corner to feather
    ctx.beginPath();
    ctx.moveTo(cornerX, cornerY);
    ctx.lineTo(fa.x, fa.y);
    dualStroke(ctx, true, false);

    drawCtrlHandle(ctx, fa.x, fa.y, mx, my);
  }

  // Corner handles
  for (let i = 0; i < pts.length; i++) {
    drawHandle(ctx, pts[i].corner[0] * w, pts[i].corner[1] * h, mx, my);
  }
}

/** Returns the hovered segment index (0-based), or -1 if no segment is hit */
export function hitTestBrushSegment(ctx: CanvasRenderingContext2D, w: number, h: number, pts: MaskPointBrush[], px: number, py: number): number {
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

export function hitTestBrush(ctx: CanvasRenderingContext2D, w: number, h: number, pts: MaskPointBrush[], px: number, py: number): boolean {
  return hitTestBrushSegment(ctx, w, h, pts, px, py) >= 0;
}

export function drawArrow(ctx: CanvasRenderingContext2D, fromX: number, fromY: number, toX: number, toY: number) {
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

export function drawGradientPolyline(ctx: CanvasRenderingContext2D, w: number, h: number, t: MaskTransformedGradient, hovered = false, mx: number | null = null, my: number | null = null) {
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

export function hitTestGradientPolyline(ctx: CanvasRenderingContext2D, w: number, h: number, t: MaskTransformedGradient, px: number, py: number): boolean {
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

/**
 * Find the nearest segment and parameter t on a path bezier for a given screen point.
 * Returns { segIdx, t } or null if the point is too far away.
 */
export function findNearestPathSegment(
  pts: MaskPointPath[], w: number, h: number, px: number, py: number, threshold = 10,
): { segIdx: number; t: number } | null {
  let bestDist = threshold * threshold;
  let bestSeg = -1;
  let bestT = 0;
  const samples = 20;

  for (let i = 0; i < pts.length; i++) {
    const curr = pts[i];
    const next = pts[(i + 1) % pts.length];
    for (let s = 0; s <= samples; s++) {
      const t = s / samples;
      const [bx, by] = bezierPos(
        curr.corner[0] * w, curr.corner[1] * h,
        curr.ctrl2[0] * w, curr.ctrl2[1] * h,
        next.ctrl1[0] * w, next.ctrl1[1] * h,
        next.corner[0] * w, next.corner[1] * h,
        t,
      );
      const dx = bx - px, dy = by - py;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestDist) {
        bestDist = d2;
        bestSeg = i;
        bestT = t;
      }
    }
  }

  return bestSeg >= 0 ? { segIdx: bestSeg, t: bestT } : null;
}

/**
 * Split a bezier segment at parameter t using de Casteljau's algorithm.
 * Returns two sets of control points: [left4, right4] each with 4 points.
 */
export function splitBezierAt(
  p0: [number, number], p1: [number, number], p2: [number, number], p3: [number, number], t: number,
): { left: [[number, number], [number, number], [number, number], [number, number]];
     right: [[number, number], [number, number], [number, number], [number, number]] } {
  const lerp = (a: [number, number], b: [number, number], u: number): [number, number] =>
    [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u];

  const m01 = lerp(p0, p1, t);
  const m12 = lerp(p1, p2, t);
  const m23 = lerp(p2, p3, t);
  const m012 = lerp(m01, m12, t);
  const m123 = lerp(m12, m23, t);
  const mid = lerp(m012, m123, t);

  return {
    left: [p0, m01, m012, mid],
    right: [mid, m123, m23, p3],
  };
}

export function buildPathBezier(pts: MaskPointPath[], w: number, h: number): Path2D {
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
export interface BorderAnchor { x: number; y: number }

/**
 * Evaluate cubic bezier position at parameter t.
 */
export function bezierPos(p0x: number, p0y: number, p1x: number, p1y: number,
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
export function bezierDeriv(p0x: number, p0y: number, p1x: number, p1y: number,
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

/** Smoothstep interpolation matching DT's _smoothstep: p1 + (p2-p1)*t*t*(3-2t) */
function smoothstep(p1: number, p2: number, t: number): number {
  return p1 + (p2 - p1) * t * t * (3.0 - 2.0 * t);
}

/**
 * Compute border point at parameter t of a bezier segment, matching DT's _path_border_get_XY.
 * Offsets perpendicular to the derivative by rad.
 */
export function borderPointAt(p0x: number, p0y: number, p1x: number, p1y: number,
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
export function computeBorderAnchors(pts: MaskPointPath[], w: number, h: number, cw: number): BorderAnchor[] {
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
export const BORDER_SAMPLES = 64;

/**
 * Build the border as a densely-sampled polyline Path2D, matching DT's approach.
 * For each bezier segment, sample border positions at multiple t values.
 */
export function buildBorderPolyline(pts: MaskPointPath[], w: number, h: number, cw: number): Path2D {
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
      const rad = smoothstep(radStart, radEnd, t);
      const [bx, by] = borderPointAt(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, t, rad);
      if (first) { p.moveTo(bx, by); first = false; }
      else p.lineTo(bx, by);
    }
  }
  p.closePath();
  return p;
}

/** Compute winding direction matching DT's _path_is_clockwise */
export function pathWindingCW(pts: MaskPointPath[]): number {
  let area = 0;
  for (let i = 0; i < pts.length; i++) {
    const curr = pts[i].corner;
    const next = pts[(i + 1) % pts.length].corner;
    area += (next[0] - curr[0]) * (next[1] + curr[1]);
  }
  // DT: sum < 0 → clockwise → cw=1; sum >= 0 → cw=-1
  return area < 0 ? 1 : -1;
}

export function hitTestPath(ctx: CanvasRenderingContext2D, w: number, h: number, pts: MaskPointPath[], px: number, py: number, serverBorderPolyline?: number[]): boolean {
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

export function hitTestForm(ctx: CanvasRenderingContext2D, w: number, h: number, form: MaskForm, px: number, py: number, grid: DistortionGrid | null = null): boolean {
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

export function drawForm(ctx: CanvasRenderingContext2D, w: number, h: number, form: MaskForm, hovered = false, mx: number | null = null, my: number | null = null, editedPoint: { formid: number; index: number } | null = null, hoveredSeg = -1, grid: DistortionGrid | null = null) {
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
      const editBrushIdx = editedPoint?.formid === form.formid ? editedPoint.index : null;
      const pts = form.points as MaskPointBrush[];
      const tPts = transformBrushPoints(grid, pts);
      drawBrush(ctx, w, h, tPts, hovered, mx, my, undefined, undefined, hoveredSeg, editBrushIdx);
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

/**
 * Check if mouse is near a path control point handle (ctrl1 or ctrl2) of the edited point.
 * Returns the DragTarget if hit, null otherwise.
 */
export function hitTestPathCtrlHandle(
  w: number, h: number,
  form: MaskForm, editedIdx: number,
  px: number, py: number, grid: DistortionGrid,
): DragTarget | null {
  const pts = form.points as MaskPointPath[];
  if (!pts || editedIdx < 0 || editedIdx >= pts.length) return null;
  const tPts = transformPathPoints(grid, pts);
  const p = tPts[editedIdx];
  if (isNearHandle(p.ctrl1[0] * w, p.ctrl1[1] * h, px, py))
    return { kind: "pathCtrl1", formid: form.formid, pointIndex: editedIdx };
  if (isNearHandle(p.ctrl2[0] * w, p.ctrl2[1] * h, px, py))
    return { kind: "pathCtrl2", formid: form.formid, pointIndex: editedIdx };
  return null;
}

/**
 * Check if mouse is near the feather or border handle of an edited brush point.
 * Returns the DragTarget if hit, null otherwise.
 */
export function hitTestBrushFeatherHandle(
  w: number, h: number,
  form: MaskForm, editedIdx: number,
  px: number, py: number, grid: DistortionGrid,
): DragTarget | null {
  const pts = form.points as MaskPointBrush[];
  if (!pts || editedIdx < 0 || editedIdx >= pts.length) return null;
  const tPts = transformBrushPoints(grid, pts);

  // Feather handle
  const featherAnchors = computeBrushFeatherAnchors(tPts, w, h);
  if (isNearHandle(featherAnchors[editedIdx].x, featherAnchors[editedIdx].y, px, py))
    return { kind: "brushFeather", formid: form.formid, pointIndex: editedIdx };

  // Border handle
  const borderAnchors = computeBrushBorderAnchors(tPts, w, h);
  if (isNearHandle(borderAnchors[editedIdx].x, borderAnchors[editedIdx].y, px, py))
    return { kind: "brushBorder", formid: form.formid, pointIndex: editedIdx };

  return null;
}

export function hitTestDragTarget(
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
    } else if (baseType === MASKS_TYPE.PATH) {
      const pts = form.points as MaskPointPath[];
      if (!pts || pts.length < 2) return null;
      const tPts = transformPathPoints(grid, pts);

      // Check corner handles
      for (let i = 0; i < tPts.length; i++) {
        if (isNearHandle(tPts[i].corner[0] * w, tPts[i].corner[1] * h, px, py))
          return { kind: "pathCorner", formid: form.formid, pointIndex: i };
      }

      // Check control point handles (only for edited point — need editedPoint context)
      // This is handled in MaskOverlay via editedPointRef

      // Check border handles
      const cw = pathWindingCW(tPts);
      const borderAnchors = computeBorderAnchors(tPts, w, h, cw);
      for (let i = 0; i < borderAnchors.length; i++) {
        if (isNearHandle(borderAnchors[i].x, borderAnchors[i].y, px, py))
          return { kind: "pathBorder", formid: form.formid, pointIndex: i };
      }

      // Grab anywhere inside the path to move
      const mainPath = buildPathBezier(tPts, w, h);
      if (ctx.isPointInPath(mainPath, px, py))
        return { kind: "center", formid: form.formid };

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
    } else if (baseType === MASKS_TYPE.BRUSH) {
      const pts = form.points as MaskPointBrush[];
      if (!pts || pts.length < 2) return null;
      const tPts = transformBrushPoints(grid, pts);

      // Check corner handles (point drag)
      for (let i = 0; i < tPts.length; i++) {
        if (isNearHandle(tPts[i].corner[0] * w, tPts[i].corner[1] * h, px, py))
          return { kind: "brushCorner", formid: form.formid, pointIndex: i };
      }

      // Check segment/form proximity — form drag
      const seg = hitTestBrushSegment(ctx, w, h, tPts, px, py);
      if (seg >= 0)
        return { kind: "center", formid: form.formid };
    }
    return null;
  }

  return null;
}
