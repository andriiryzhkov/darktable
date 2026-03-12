/**
 * Mask drag operations — extracted from MaskOverlay.tsx.
 *
 * Provides per-mask-type drag logic for center/radius/border/rotate interactions,
 * along with creation and commit parameter helpers.
 */

import { MASKS_TYPE } from "../types/protocol";
import type {
  MaskForm,
  MaskPointsCircle,
  MaskPointsEllipse,
  MaskPointsGradient,
  MaskPointPath,
  MaskPointBrush,
  DistortionGrid,
} from "../types/protocol";
import { forwardTransform, inverseTransform } from "./distortionGrid";

// ─── Bezier control point helpers (port of darktable path.c) ────────────────

/** Angle from reference to point, corrected for aspect ratio. */
function ctrlAngle2d(
  xRef: number, yRef: number,
  x: number, y: number,
  aspect: number,
): number {
  return Math.atan2(y - yRef, x * aspect - xRef * aspect);
}

/** Angle between ctrl1 and ctrl2 lines from corner, aspect-corrected. */
function getCtrlAngle(
  cx: number, cy: number,
  c1x: number, c1y: number,
  c2x: number, c2y: number,
  aspect: number,
): number {
  return ctrlAngle2d(cx, cy, c2x, c2y, aspect) - ctrlAngle2d(cx, cy, c1x, c1y, aspect);
}

/** Distance ratio |ctrl1-corner| / |ctrl2-corner|, aspect-corrected. */
function getCtrlScale(
  cx: number, cy: number,
  c1x: number, c1y: number,
  c2x: number, c2y: number,
  aspect: number,
): number {
  const cxa = cx * aspect, c1xa = c1x * aspect, c2xa = c2x * aspect;
  const l1 = Math.sqrt((c1xa - cxa) ** 2 + (c1y - cy) ** 2);
  const l2 = Math.sqrt((c2xa - cxa) ** 2 + (c2y - cy) ** 2);
  return l2 > 0 ? l1 / l2 : 1;
}

/** Move the opposite ctrl handle to preserve angle and scale ratio. */
function setCtrlAngleAndScale(
  cx: number, cy: number,
  angle: number, scale: number,
  movedIsCtrl1: boolean,
  c1: [number, number], c2: [number, number],
  aspect: number,
): void {
  const cxa = cx * aspect;
  if (movedIsCtrl1) {
    // ctrl1 was moved → adjust ctrl2
    const c1xa = c1[0] * aspect;
    const l1 = Math.sqrt((c1xa - cxa) ** 2 + (c1[1] - cy) ** 2);
    const a1 = Math.atan2(c1[1] - cy, c1xa - cxa);
    const a2 = a1 + angle;
    const l2 = scale > 0 ? l1 / scale : l1;
    c2[0] = (cxa + l2 * Math.cos(a2)) / aspect;
    c2[1] = cy + l2 * Math.sin(a2);
  } else {
    // ctrl2 was moved → adjust ctrl1
    const c2xa = c2[0] * aspect;
    const l2 = Math.sqrt((c2xa - cxa) ** 2 + (c2[1] - cy) ** 2);
    const a2 = Math.atan2(c2[1] - cy, c2xa - cxa);
    const a1 = a2 - angle;
    const l1 = l2 * scale;
    c1[0] = (cxa + l1 * Math.cos(a1)) / aspect;
    c1[1] = cy + l1 * Math.sin(a1);
  }
}

// ─── DragTarget ─────────────────────────────────────────────────────────────

/** Hit-test result identifying what part of a mask the user is dragging. */
export type DragTarget =
  | { kind: "center"; formid: number }
  | { kind: "radius"; formid: number; pointIndex?: number }
  | { kind: "border"; formid: number; pointIndex?: number }
  | { kind: "rotate"; formid: number }
  | { kind: "pathCorner"; formid: number; pointIndex: number }
  | { kind: "pathCtrl1"; formid: number; pointIndex: number }
  | { kind: "pathCtrl2"; formid: number; pointIndex: number }
  | { kind: "pathBorder"; formid: number; pointIndex: number }
  | { kind: "brushCorner"; formid: number; pointIndex: number }
  | { kind: "brushFeather"; formid: number; pointIndex: number }
  | { kind: "brushBorder"; formid: number; pointIndex: number }
  | { kind: "brushSegment"; formid: number; pointIndex: number };

// ─── DragState ──────────────────────────────────────────────────────────────

/** Persistent state captured at drag start and updated during dragging. */
export interface DragState {
  target: DragTarget;
  startX: number;
  startY: number;
  origCenter: [number, number];
  origRadius: number | [number, number];
  origBorder: number;
  dragAxis?: 0 | 1;
  lastAngle?: number;
  origCompression?: number;
  /** Deep copy of all path points at drag start (for path masks). */
  origPoints?: MaskPointPath[];
  /** Deep copy of all brush points at drag start (for brush masks). */
  origBrushPoints?: MaskPointBrush[];
  /** Whether the mouse actually moved during the drag. */
  moved?: boolean;
  /** Whether shift is held (updated each move). */
  shiftKey?: boolean;
  /** Ctrl handle angle at drag start (for coupled ctrl movement). */
  ctrlAngle?: number;
  /** Ctrl handle scale ratio at drag start (for coupled ctrl movement). */
  ctrlScale?: number;
}

// ─── MaskDragOps ────────────────────────────────────────────────────────────

/** Per-mask-type drag operations. */
export interface MaskDragOps {
  /** Get the mask's position (center for circle/ellipse, anchor for gradient). */
  getPosition(form: MaskForm): [number, number];
  /** Set the mask's position. */
  setPosition(form: MaskForm, pos: [number, number]): void;
  /** Build initial DragState when starting a drag. */
  initDrag(
    form: MaskForm,
    target: DragTarget,
    px: number,
    py: number,
    w: number,
    h: number,
    grid: DistortionGrid,
  ): DragState;
  /** Update form.points during drag based on mouse position. */
  updateDrag(
    form: MaskForm,
    drag: DragState,
    rawMouse: [number, number],
    px: number,
    py: number,
    w: number,
    h: number,
    grid: DistortionGrid,
  ): void;
  /** Get params to send to previewMaskParam during drag. */
  previewParams(form: MaskForm, dragKind: string): Record<string, unknown>;
  /** Get params to send to updateMask on drag commit. */
  commitParams(form: MaskForm): Record<string, unknown>;
}

// ─── Circle drag ops ────────────────────────────────────────────────────────

const circleDragOps: MaskDragOps = {
  getPosition(form: MaskForm): [number, number] {
    const pts = form.points as MaskPointsCircle;
    return pts.center;
  },

  setPosition(form: MaskForm, pos: [number, number]): void {
    const pts = form.points as MaskPointsCircle;
    pts.center[0] = pos[0];
    pts.center[1] = pos[1];
  },

  initDrag(
    form: MaskForm,
    target: DragTarget,
    px: number,
    py: number,
    _w: number,
    _h: number,
    _grid: DistortionGrid,
  ): DragState {
    const c = form.points as MaskPointsCircle;
    return {
      target,
      startX: px,
      startY: py,
      origCenter: [c.center[0], c.center[1]],
      origRadius: c.radius,
      origBorder: c.border,
    };
  },

  updateDrag(
    form: MaskForm,
    drag: DragState,
    rawMouse: [number, number],
    _px: number,
    _py: number,
    w: number,
    h: number,
    grid: DistortionGrid,
  ): void {
    const pts = form.points as MaskPointsCircle;
    const iw = grid.iwidth;
    const ih = grid.iheight;
    const dim = Math.min(iw, ih);

    if (drag.target.kind === "center") {
      const rawStart = inverseTransform(grid, drag.startX / w, drag.startY / h);
      const newX = drag.origCenter[0] + (rawMouse[0] - rawStart[0]);
      const newY = drag.origCenter[1] + (rawMouse[1] - rawStart[1]);
      pts.center[0] = newX;
      pts.center[1] = newY;
    } else if (drag.target.kind === "radius") {
      const dx = (rawMouse[0] - pts.center[0]) * iw;
      const dy = (rawMouse[1] - pts.center[1]) * ih;
      pts.radius = Math.max(0.001, Math.sqrt(dx * dx + dy * dy) / dim);
    } else if (drag.target.kind === "border") {
      const dx = (rawMouse[0] - pts.center[0]) * iw;
      const dy = (rawMouse[1] - pts.center[1]) * ih;
      pts.border = Math.max(0.001, Math.sqrt(dx * dx + dy * dy) / dim - pts.radius);
    }
  },

  previewParams(form: MaskForm, dragKind: string): Record<string, unknown> {
    const cp = form.points as MaskPointsCircle;
    if (dragKind === "center") return { center: [cp.center[0], cp.center[1]] };
    if (dragKind === "radius") return { radius: cp.radius };
    if (dragKind === "border") return { border: cp.border };
    return {};
  },

  commitParams(form: MaskForm): Record<string, unknown> {
    const pts = form.points as MaskPointsCircle;
    return {
      center: [pts.center[0], pts.center[1]],
      radius: pts.radius,
      border: pts.border,
    };
  },
};

// ─── Ellipse drag ops ───────────────────────────────────────────────────────

const ellipseDragOps: MaskDragOps = {
  getPosition(form: MaskForm): [number, number] {
    const pts = form.points as MaskPointsEllipse;
    return pts.center;
  },

  setPosition(form: MaskForm, pos: [number, number]): void {
    const pts = form.points as MaskPointsEllipse;
    pts.center[0] = pos[0];
    pts.center[1] = pos[1];
  },

  initDrag(
    form: MaskForm,
    target: DragTarget,
    px: number,
    py: number,
    _w: number,
    _h: number,
    _grid: DistortionGrid,
  ): DragState {
    const el = form.points as MaskPointsEllipse;
    // Lock axis at drag start: polyline q=0,2 -> longer axis, q=1,3 -> shorter
    const pi = (target as { pointIndex?: number }).pointIndex ?? 0;
    const isLongerAxis = pi === 0 || pi === 2;
    const swapped = el.radius[0] < el.radius[1];
    const dragAxis: 0 | 1 = (isLongerAxis !== swapped) ? 0 : 1;
    return {
      target,
      startX: px,
      startY: py,
      origCenter: [el.center[0], el.center[1]],
      origRadius: [el.radius[0], el.radius[1]],
      origBorder: el.border,
      dragAxis,
    };
  },

  updateDrag(
    form: MaskForm,
    drag: DragState,
    rawMouse: [number, number],
    px: number,
    py: number,
    w: number,
    h: number,
    grid: DistortionGrid,
  ): void {
    const ep = form.points as MaskPointsEllipse;
    const iw = grid.iwidth;
    const ih = grid.iheight;
    const dim = Math.min(iw, ih);

    if (drag.target.kind === "center") {
      const rawStart = inverseTransform(grid, drag.startX / w, drag.startY / h);
      const newX = drag.origCenter[0] + (rawMouse[0] - rawStart[0]);
      const newY = drag.origCenter[1] + (rawMouse[1] - rawStart[1]);
      ep.center[0] = newX;
      ep.center[1] = newY;
    } else if (drag.target.kind === "radius") {
      const dx = (rawMouse[0] - ep.center[0]) * iw;
      const dy = (rawMouse[1] - ep.center[1]) * ih;
      const rot = (ep.rotation ?? 0) * Math.PI / 180;
      if (drag.dragAxis === 0) {
        const proj = Math.abs(dx * Math.cos(rot) + dy * Math.sin(rot)) / dim;
        ep.radius[0] = Math.max(0.002, proj);
      } else {
        const proj = Math.abs(-dx * Math.sin(rot) + dy * Math.cos(rot)) / dim;
        ep.radius[1] = Math.max(0.002, proj);
      }
    } else if (drag.target.kind === "border") {
      const dx = (rawMouse[0] - ep.center[0]) * iw;
      const dy = (rawMouse[1] - ep.center[1]) * ih;
      const rot = (ep.rotation ?? 0) * Math.PI / 180;
      if (drag.dragAxis === 0) {
        const proj = Math.abs(dx * Math.cos(rot) + dy * Math.sin(rot)) / dim;
        ep.border = Math.max(0.001, proj - ep.radius[0]);
      } else {
        const proj = Math.abs(-dx * Math.sin(rot) + dy * Math.cos(rot)) / dim;
        ep.border = Math.max(0.001, proj - ep.radius[1]);
      }
    } else if (drag.target.kind === "rotate") {
      const tc = forwardTransform(grid, ep.center[0], ep.center[1]);
      const curAngle = Math.atan2(py / h - tc[1], px / w - tc[0]);
      const delta = (curAngle - (drag.lastAngle ?? curAngle)) * (180 / Math.PI);
      ep.rotation = ((ep.rotation ?? 0) + delta) % 360;
      if (ep.rotation < 0) ep.rotation += 360;
      drag.lastAngle = curAngle;
    }
  },

  previewParams(form: MaskForm, dragKind: string): Record<string, unknown> {
    const ep = form.points as MaskPointsEllipse;
    if (dragKind === "center") return { center: [ep.center[0], ep.center[1]] };
    if (dragKind === "radius") return { radius: [ep.radius[0], ep.radius[1]] };
    if (dragKind === "border") return { border: ep.border };
    if (dragKind === "rotate") return { rotation: ep.rotation };
    return {};
  },

  commitParams(form: MaskForm): Record<string, unknown> {
    const pts = form.points as MaskPointsEllipse;
    return {
      center: [pts.center[0], pts.center[1]],
      radius: [pts.radius[0], pts.radius[1]],
      border: pts.border,
      rotation: pts.rotation,
    };
  },
};

// ─── Gradient drag ops ──────────────────────────────────────────────────────

const gradientDragOps: MaskDragOps = {
  getPosition(form: MaskForm): [number, number] {
    const pts = form.points as MaskPointsGradient;
    return pts.anchor;
  },

  setPosition(form: MaskForm, pos: [number, number]): void {
    const pts = form.points as MaskPointsGradient;
    pts.anchor[0] = pos[0];
    pts.anchor[1] = pos[1];
  },

  initDrag(
    form: MaskForm,
    target: DragTarget,
    px: number,
    py: number,
    w: number,
    h: number,
    grid: DistortionGrid,
  ): DragState {
    const g = form.points as MaskPointsGradient;
    let startAngle: number | undefined;
    if (target.kind === "rotate") {
      const tc = forwardTransform(grid, g.anchor[0], g.anchor[1]);
      startAngle = Math.atan2(py / h - tc[1], px / w - tc[0]);
    }
    return {
      target,
      startX: px,
      startY: py,
      origCenter: [g.anchor[0], g.anchor[1]],
      origRadius: 0,
      origBorder: 0,
      origCompression: g.compression,
      lastAngle: startAngle,
    };
  },

  updateDrag(
    form: MaskForm,
    drag: DragState,
    rawMouse: [number, number],
    px: number,
    py: number,
    w: number,
    h: number,
    grid: DistortionGrid,
  ): void {
    const gp = form.points as MaskPointsGradient;
    const iw = grid.iwidth;
    const ih = grid.iheight;

    if (drag.target.kind === "center") {
      const rawStart = inverseTransform(grid, drag.startX / w, drag.startY / h);
      const newX = drag.origCenter[0] + (rawMouse[0] - rawStart[0]);
      const newY = drag.origCenter[1] + (rawMouse[1] - rawStart[1]);
      gp.anchor[0] = newX;
      gp.anchor[1] = newY;
    } else if (drag.target.kind === "border") {
      // Compression drag: distance from mouse to anchor along gradient direction
      const rot = (gp.rotation * Math.PI) / 180;
      const gdx = Math.sin(rot);
      const gdy = Math.cos(rot);
      const dx = (rawMouse[0] - gp.anchor[0]) * iw;
      const dy = (rawMouse[1] - gp.anchor[1]) * ih;
      const diag = Math.sqrt(iw * iw + ih * ih);
      const proj = Math.abs(dx * gdx + dy * gdy);
      gp.compression = Math.max(0.001, Math.min(1, proj / diag));
    } else if (drag.target.kind === "rotate") {
      const tc = forwardTransform(grid, gp.anchor[0], gp.anchor[1]);
      const curAngle = Math.atan2(py / h - tc[1], px / w - tc[0]);
      const delta = (curAngle - (drag.lastAngle ?? curAngle)) * (180 / Math.PI);
      // Gradient rotation is negated relative to ellipse
      gp.rotation = ((gp.rotation ?? 0) - delta) % 360;
      if (gp.rotation < 0) gp.rotation += 360;
      drag.lastAngle = curAngle;
    }
  },

  previewParams(form: MaskForm, dragKind: string): Record<string, unknown> {
    const gp = form.points as MaskPointsGradient;
    if (dragKind === "center") return { anchor: [gp.anchor[0], gp.anchor[1]] };
    if (dragKind === "border") return { compression: gp.compression };
    if (dragKind === "rotate") return { rotation: gp.rotation };
    return {};
  },

  commitParams(form: MaskForm): Record<string, unknown> {
    const gp = form.points as MaskPointsGradient;
    return {
      anchor: [gp.anchor[0], gp.anchor[1]],
      rotation: gp.rotation,
      compression: gp.compression,
      steepness: gp.steepness,
      curvature: gp.curvature,
      state: gp.state,
    };
  },
};

// ─── Path drag ops ───────────────────────────────────────────────────────────

/** Deep-copy a path point. */
function clonePathPoint(p: MaskPointPath): MaskPointPath {
  return {
    corner: [p.corner[0], p.corner[1]],
    ctrl1: [p.ctrl1[0], p.ctrl1[1]],
    ctrl2: [p.ctrl2[0], p.ctrl2[1]],
    border: [p.border[0], p.border[1]],
    state: p.state,
  };
}

/**
 * Auto-compute catmull-rom control points for path nodes with state NORMAL (1).
 * Mirrors darktable's _path_init_ctrl_points() from path.c.
 */
export function initPathCtrlPoints(points: MaskPointPath[]): void {
  const n = points.length;
  if (n < 2) return;
  for (let k = 0; k < n; k++) {
    const point3 = points[k];
    // Only recompute for NORMAL state points (bit 0 set)
    if (!(point3.state & 1)) continue;
    const km2 = ((k - 2) % n + n) % n;
    const km1 = ((k - 1) % n + n) % n;
    const kp1 = (k + 1) % n;
    const kp2 = (k + 2) % n;
    const p1 = points[km2];
    const p2 = points[km1];
    const p4 = points[kp1];
    const p5 = points[kp2];
    // Catmull-rom → bezier for segment (p2 → point3)
    let bx1 = (-p1.corner[0] + 6 * p2.corner[0] + point3.corner[0]) / 6;
    let by1 = (-p1.corner[1] + 6 * p2.corner[1] + point3.corner[1]) / 6;
    let bx2 = (p2.corner[0] + 6 * point3.corner[0] - p4.corner[0]) / 6;
    let by2 = (p2.corner[1] + 6 * point3.corner[1] - p4.corner[1]) / 6;
    if (p2.ctrl2[0] === -1) p2.ctrl2[0] = bx1;
    if (p2.ctrl2[1] === -1) p2.ctrl2[1] = by1;
    point3.ctrl1[0] = bx2;
    point3.ctrl1[1] = by2;
    // Catmull-rom → bezier for segment (point3 → p4)
    bx1 = (-p2.corner[0] + 6 * point3.corner[0] + p4.corner[0]) / 6;
    by1 = (-p2.corner[1] + 6 * point3.corner[1] + p4.corner[1]) / 6;
    bx2 = (point3.corner[0] + 6 * p4.corner[0] - p5.corner[0]) / 6;
    by2 = (point3.corner[1] + 6 * p4.corner[1] - p5.corner[1]) / 6;
    if (p4.ctrl1[0] === -1) p4.ctrl1[0] = bx2;
    if (p4.ctrl1[1] === -1) p4.ctrl1[1] = by2;
    point3.ctrl2[0] = bx1;
    point3.ctrl2[1] = by1;
  }
}

const pathDragOps: MaskDragOps = {
  getPosition(form: MaskForm): [number, number] {
    const pts = form.points as MaskPointPath[];
    if (!pts || pts.length === 0) return [0.5, 0.5];
    let cx = 0, cy = 0;
    for (const p of pts) { cx += p.corner[0]; cy += p.corner[1]; }
    return [cx / pts.length, cy / pts.length];
  },

  setPosition(form: MaskForm, pos: [number, number]): void {
    const pts = form.points as MaskPointPath[];
    if (!pts || pts.length === 0) return;
    const [ocx, ocy] = this.getPosition(form);
    const dx = pos[0] - ocx;
    const dy = pos[1] - ocy;
    for (const p of pts) {
      p.corner[0] += dx; p.corner[1] += dy;
      p.ctrl1[0] += dx; p.ctrl1[1] += dy;
      p.ctrl2[0] += dx; p.ctrl2[1] += dy;
    }
  },

  initDrag(
    form: MaskForm,
    target: DragTarget,
    px: number,
    py: number,
    _w: number,
    _h: number,
    grid: DistortionGrid,
  ): DragState {
    const pts = form.points as MaskPointPath[];
    const center = this.getPosition(form);
    const state: DragState = {
      target,
      startX: px,
      startY: py,
      origCenter: center,
      origRadius: 0,
      origBorder: 0,
      origPoints: pts.map(clonePathPoint),
    };
    // Capture angle/scale for coupled ctrl handle movement
    if ((target.kind === "pathCtrl1" || target.kind === "pathCtrl2") && "pointIndex" in target) {
      const p = pts[target.pointIndex];
      const aspect = grid.iwidth / grid.iheight;
      state.ctrlAngle = getCtrlAngle(
        p.corner[0], p.corner[1],
        p.ctrl1[0], p.ctrl1[1],
        p.ctrl2[0], p.ctrl2[1],
        aspect,
      );
      state.ctrlScale = getCtrlScale(
        p.corner[0], p.corner[1],
        p.ctrl1[0], p.ctrl1[1],
        p.ctrl2[0], p.ctrl2[1],
        aspect,
      );
    }
    return state;
  },

  updateDrag(
    form: MaskForm,
    drag: DragState,
    rawMouse: [number, number],
    _px: number,
    _py: number,
    w: number,
    h: number,
    grid: DistortionGrid,
  ): void {
    const pts = form.points as MaskPointPath[];
    const orig = drag.origPoints!;
    const kind = drag.target.kind;
    const idx = (drag.target as { pointIndex?: number }).pointIndex ?? 0;

    if (kind === "center") {
      // Move all points by delta from drag start
      const rawStart = inverseTransform(grid, drag.startX / w, drag.startY / h);
      const dx = rawMouse[0] - rawStart[0];
      const dy = rawMouse[1] - rawStart[1];
      for (let i = 0; i < pts.length; i++) {
        pts[i].corner[0] = orig[i].corner[0] + dx;
        pts[i].corner[1] = orig[i].corner[1] + dy;
        pts[i].ctrl1[0] = orig[i].ctrl1[0] + dx;
        pts[i].ctrl1[1] = orig[i].ctrl1[1] + dy;
        pts[i].ctrl2[0] = orig[i].ctrl2[0] + dx;
        pts[i].ctrl2[1] = orig[i].ctrl2[1] + dy;
      }
    } else if (kind === "pathCorner") {
      // Move a single corner + its control points
      const dx = rawMouse[0] - orig[idx].corner[0];
      const dy = rawMouse[1] - orig[idx].corner[1];
      pts[idx].corner[0] = rawMouse[0];
      pts[idx].corner[1] = rawMouse[1];
      pts[idx].ctrl1[0] = orig[idx].ctrl1[0] + dx;
      pts[idx].ctrl1[1] = orig[idx].ctrl1[1] + dy;
      pts[idx].ctrl2[0] = orig[idx].ctrl2[0] + dx;
      pts[idx].ctrl2[1] = orig[idx].ctrl2[1] + dy;
    } else if (kind === "pathCtrl1") {
      pts[idx].ctrl1[0] = rawMouse[0];
      pts[idx].ctrl1[1] = rawMouse[1];
      if (!drag.shiftKey && drag.ctrlAngle !== undefined && drag.ctrlScale !== undefined) {
        const aspect = grid.iwidth / grid.iheight;
        setCtrlAngleAndScale(
          pts[idx].corner[0], pts[idx].corner[1],
          drag.ctrlAngle, drag.ctrlScale,
          true, pts[idx].ctrl1, pts[idx].ctrl2, aspect,
        );
      }
    } else if (kind === "pathCtrl2") {
      pts[idx].ctrl2[0] = rawMouse[0];
      pts[idx].ctrl2[1] = rawMouse[1];
      if (!drag.shiftKey && drag.ctrlAngle !== undefined && drag.ctrlScale !== undefined) {
        const aspect = grid.iwidth / grid.iheight;
        setCtrlAngleAndScale(
          pts[idx].corner[0], pts[idx].corner[1],
          drag.ctrlAngle, drag.ctrlScale,
          false, pts[idx].ctrl1, pts[idx].ctrl2, aspect,
        );
      }
    } else if (kind === "pathBorder") {
      // Project mouse onto the corner→border direction line (perpendicular to ctrl2 tangent).
      // Matches DT: border anchor is at t=0 of outgoing segment, perpendicular to ctrl2.
      const iw = grid.iwidth;
      const ih = grid.iheight;
      const dim = Math.min(iw, ih);
      const cx = pts[idx].corner[0] * iw;
      const cy = pts[idx].corner[1] * ih;
      const c2x = pts[idx].ctrl2[0] * iw;
      const c2y = pts[idx].ctrl2[1] * ih;
      // Tangent direction at t=0 of outgoing segment is (ctrl2 - corner)
      const tx = c2x - cx;
      const ty = c2y - cy;
      const tlen = Math.sqrt(tx * tx + ty * ty);
      if (tlen > 1e-10) {
        // Perpendicular direction (border direction): rotate tangent 90°
        const nx = ty / tlen;
        const ny = -tx / tlen;
        // Project mouse position onto perpendicular direction from corner
        const mx = rawMouse[0] * iw - cx;
        const my = rawMouse[1] * ih - cy;
        const proj = mx * nx + my * ny;
        const newBorder = Math.max(0.001, Math.abs(proj) / dim);
        pts[idx].border[0] = newBorder;
        pts[idx].border[1] = newBorder;
      }
    }
  },

  previewParams(form: MaskForm, _dragKind: string): Record<string, unknown> {
    const pts = form.points as MaskPointPath[];
    return { points: pts.map(clonePathPoint) };
  },

  commitParams(form: MaskForm): Record<string, unknown> {
    const pts = form.points as MaskPointPath[];
    return { points: pts.map(clonePathPoint) };
  },
};

// ─── Brush drag ops ─────────────────────────────────────────────────────────

/** Deep-copy a brush point. */
function cloneBrushPoint(p: MaskPointBrush): MaskPointBrush {
  return {
    corner: [p.corner[0], p.corner[1]],
    ctrl1: [p.ctrl1[0], p.ctrl1[1]],
    ctrl2: [p.ctrl2[0], p.ctrl2[1]],
    border: [p.border[0], p.border[1]],
    density: p.density,
    hardness: p.hardness,
    state: p.state,
  };
}

/**
 * Convert ctrl2 to feather extremity position.
 * Matches DT's _brush_ctrl2_to_feather().
 * Works in any coordinate space (caller converts).
 */
export function brushCtrl2ToFeather(
  ptx: number, pty: number,
  ctrlx: number, ctrly: number,
  clockwise: boolean,
): [number, number] {
  if (clockwise) {
    return [ptx + ctrly - pty, pty + ptx - ctrlx];
  } else {
    return [ptx - ctrly + pty, pty - ptx + ctrlx];
  }
}

/**
 * Convert feather extremity back to ctrl1 and ctrl2.
 * Matches DT's _brush_feather_to_ctrl().
 */
export function brushFeatherToCtrl(
  ptx: number, pty: number,
  fx: number, fy: number,
  clockwise: boolean,
): { ctrl1: [number, number]; ctrl2: [number, number] } {
  if (clockwise) {
    return {
      ctrl2: [ptx + pty - fy, pty + fx - ptx],
      ctrl1: [ptx - pty + fy, pty - fx + ptx],
    };
  } else {
    return {
      ctrl1: [ptx + pty - fy, pty + fx - ptx],
      ctrl2: [ptx - pty + fy, pty - fx + ptx],
    };
  }
}

const brushDragOps: MaskDragOps = {
  getPosition(form: MaskForm): [number, number] {
    const pts = form.points as MaskPointBrush[];
    if (!pts || pts.length === 0) return [0.5, 0.5];
    let cx = 0, cy = 0;
    for (const p of pts) { cx += p.corner[0]; cy += p.corner[1]; }
    return [cx / pts.length, cy / pts.length];
  },

  setPosition(form: MaskForm, pos: [number, number]): void {
    const pts = form.points as MaskPointBrush[];
    if (!pts || pts.length === 0) return;
    const [ocx, ocy] = this.getPosition(form);
    const dx = pos[0] - ocx;
    const dy = pos[1] - ocy;
    for (const p of pts) {
      p.corner[0] += dx; p.corner[1] += dy;
      p.ctrl1[0] += dx; p.ctrl1[1] += dy;
      p.ctrl2[0] += dx; p.ctrl2[1] += dy;
    }
  },

  initDrag(
    form: MaskForm,
    target: DragTarget,
    px: number,
    py: number,
    _w: number,
    _h: number,
    _grid: DistortionGrid,
  ): DragState {
    const pts = form.points as MaskPointBrush[];
    const center = this.getPosition(form);
    return {
      target,
      startX: px,
      startY: py,
      origCenter: center,
      origRadius: 0,
      origBorder: 0,
      origBrushPoints: pts.map(cloneBrushPoint),
    };
  },

  updateDrag(
    form: MaskForm,
    drag: DragState,
    rawMouse: [number, number],
    _px: number,
    _py: number,
    w: number,
    h: number,
    grid: DistortionGrid,
  ): void {
    const pts = form.points as MaskPointBrush[];
    const orig = drag.origBrushPoints!;
    const kind = drag.target.kind;
    const idx = (drag.target as { pointIndex?: number }).pointIndex ?? 0;

    if (kind === "center") {
      // Move all points by delta from drag start
      const rawStart = inverseTransform(grid, drag.startX / w, drag.startY / h);
      const dx = rawMouse[0] - rawStart[0];
      const dy = rawMouse[1] - rawStart[1];
      for (let i = 0; i < pts.length; i++) {
        pts[i].corner[0] = orig[i].corner[0] + dx;
        pts[i].corner[1] = orig[i].corner[1] + dy;
        pts[i].ctrl1[0] = orig[i].ctrl1[0] + dx;
        pts[i].ctrl1[1] = orig[i].ctrl1[1] + dy;
        pts[i].ctrl2[0] = orig[i].ctrl2[0] + dx;
        pts[i].ctrl2[1] = orig[i].ctrl2[1] + dy;
      }
    } else if (kind === "brushCorner") {
      // Move single point + its control points
      const dx = rawMouse[0] - orig[idx].corner[0];
      const dy = rawMouse[1] - orig[idx].corner[1];
      pts[idx].corner[0] = rawMouse[0];
      pts[idx].corner[1] = rawMouse[1];
      pts[idx].ctrl1[0] = orig[idx].ctrl1[0] + dx;
      pts[idx].ctrl1[1] = orig[idx].ctrl1[1] + dy;
      pts[idx].ctrl2[0] = orig[idx].ctrl2[0] + dx;
      pts[idx].ctrl2[1] = orig[idx].ctrl2[1] + dy;
      // Recompute catmull-rom for affected NORMAL points
      initBrushCtrlPoints(pts);
    } else if (kind === "brushSegment") {
      // Move both segment endpoints + controls by same delta
      const rawStart = inverseTransform(grid, drag.startX / w, drag.startY / h);
      const dx = rawMouse[0] - rawStart[0];
      const dy = rawMouse[1] - rawStart[1];
      const i1 = idx;
      const i2 = Math.min(idx + 1, pts.length - 1);
      for (const i of [i1, i2]) {
        pts[i].corner[0] = orig[i].corner[0] + dx;
        pts[i].corner[1] = orig[i].corner[1] + dy;
        pts[i].ctrl1[0] = orig[i].ctrl1[0] + dx;
        pts[i].ctrl1[1] = orig[i].ctrl1[1] + dy;
        pts[i].ctrl2[0] = orig[i].ctrl2[0] + dx;
        pts[i].ctrl2[1] = orig[i].ctrl2[1] + dy;
      }
    } else if (kind === "brushFeather") {
      // Convert mouse (feather position) back to ctrl1/ctrl2
      // DT works in pixel space for the feather math
      const iw = grid.iwidth;
      const ih = grid.iheight;
      const { ctrl1, ctrl2 } = brushFeatherToCtrl(
        pts[idx].corner[0] * iw, pts[idx].corner[1] * ih,
        rawMouse[0] * iw, rawMouse[1] * ih,
        true, // clockwise
      );
      pts[idx].ctrl1[0] = ctrl1[0] / iw;
      pts[idx].ctrl1[1] = ctrl1[1] / ih;
      pts[idx].ctrl2[0] = ctrl2[0] / iw;
      pts[idx].ctrl2[1] = ctrl2[1] / ih;
      pts[idx].state = 2; // USER — no longer auto catmull-rom
    } else if (kind === "brushBorder") {
      // Project mouse onto corner→border direction line
      const iw = grid.iwidth;
      const ih = grid.iheight;
      const dim = Math.min(iw, ih);
      const cx = pts[idx].corner[0] * iw;
      const cy = pts[idx].corner[1] * ih;
      const c2x = pts[idx].ctrl2[0] * iw;
      const c2y = pts[idx].ctrl2[1] * ih;
      // Tangent direction at t=0 of outgoing segment is (ctrl2 - corner)
      const tx = c2x - cx;
      const ty = c2y - cy;
      const tlen = Math.sqrt(tx * tx + ty * ty);
      if (tlen > 1e-10) {
        // Perpendicular direction (border direction): rotate tangent 90°
        const nx = ty / tlen;
        const ny = -tx / tlen;
        // Project mouse onto perpendicular direction from corner
        const mx = rawMouse[0] * iw - cx;
        const my = rawMouse[1] * ih - cy;
        const proj = mx * nx + my * ny;
        const newBorder = Math.max(0.001, Math.abs(proj) / dim);
        pts[idx].border[0] = newBorder;
        pts[idx].border[1] = newBorder;
      }
    }
  },

  previewParams(form: MaskForm, _dragKind: string): Record<string, unknown> {
    const pts = form.points as MaskPointBrush[];
    return { points: pts.map(cloneBrushPoint) };
  },

  commitParams(form: MaskForm): Record<string, unknown> {
    const pts = form.points as MaskPointBrush[];
    return { points: pts.map(cloneBrushPoint) };
  },
};

// ─── getDragOps ─────────────────────────────────────────────────────────────

/**
 * Return the drag operations object for a given mask form, or null if the
 * mask type does not support dragging.
 */
export function getDragOps(form: MaskForm): MaskDragOps | null {
  const baseType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);
  switch (baseType) {
    case MASKS_TYPE.CIRCLE:
      return circleDragOps;
    case MASKS_TYPE.ELLIPSE:
      return ellipseDragOps;
    case MASKS_TYPE.GRADIENT:
      return gradientDragOps;
    case MASKS_TYPE.PATH:
      return pathDragOps;
    case MASKS_TYPE.BRUSH:
      return brushDragOps;
    default:
      return null;
  }
}

// ─── getCreationParams ──────────────────────────────────────────────────────

/**
 * Return default creation parameters for a new mask of the given tool type,
 * positioned at the specified raw-space coordinate.
 */
export function getCreationParams(
  tool: string,
  position: [number, number],
): Record<string, unknown> {
  if (tool === "gradient") {
    return {
      anchor: position,
      rotation: 0,
      compression: 0.05,
      steepness: 0,
      curvature: 0,
      state: 2, // DT_MASKS_GRADIENT_STATE_SIGMOIDAL
    };
  }
  if (tool === "path") {
    // Default 4-point diamond centered at the clicked position
    const [cx, cy] = position;
    const r = 0.08; // initial radius in normalized coords
    const cr = r * 0.55; // control point offset (~circular approximation)
    const border: [number, number] = [0.05, 0.05];
    return {
      points: [
        { corner: [cx, cy - r], ctrl1: [cx - cr, cy - r], ctrl2: [cx + cr, cy - r], border, state: 0 },
        { corner: [cx + r, cy], ctrl1: [cx + r, cy - cr], ctrl2: [cx + r, cy + cr], border, state: 0 },
        { corner: [cx, cy + r], ctrl1: [cx + cr, cy + r], ctrl2: [cx - cr, cy + r], border, state: 0 },
        { corner: [cx - r, cy], ctrl1: [cx - r, cy + cr], ctrl2: [cx - r, cy - cr], border, state: 0 },
      ],
    };
  }
  const params: Record<string, unknown> = { center: position };
  if (tool === "circle") {
    params.radius = 0.05;
    params.border = 0.025;
  } else {
    // ellipse (and any other shape with dual-axis radius)
    params.radius = [0.05, 0.05];
    params.border = 0.025;
  }
  return params;
}

// ─── Brush stroke utilities ─────────────────────────────────────────────────

/** Raw sampled point during brush drawing (canvas pixels + payload). */
export interface BrushRawPoint {
  x: number;  // normalized raw-space x
  y: number;  // normalized raw-space y
  border: number;
  hardness: number;
  density: number;
}

/** Brush settings for creation. */
export interface BrushSettings {
  border: number;    // 0.0005 – 0.5, default 0.05
  hardness: number;  // 0.0005 – 1.0, default 0.5
  density: number;   // 0 – 1.0, default 1.0
}

/**
 * Ramer-Douglas-Peucker simplification for brush strokes.
 * Ported from darktable's _brush_ramer_douglas_peucker().
 * Uses weighted distance including border/hardness/density payload.
 */
/**
 * Ramer-Douglas-Peucker simplification for brush strokes.
 * Ported from darktable's _brush_ramer_douglas_peucker().
 * Uses geometric projection for distance (matching DT's approach)
 * and weighted payload distance (border² + 0.01*hardness² + 0.01*density²).
 *
 * @param epsilon2 Squared epsilon threshold = factor * max(BORDER_MIN, border)²
 *   where factor depends on smoothing level (low=0.0025, medium=0.01, high=0.04).
 */
export function simplifyBrushStroke(
  raw: BrushRawPoint[],
  epsilon2: number,
): BrushRawPoint[] {
  if (raw.length <= 2) return [...raw];
  const keep = new Uint8Array(raw.length);
  keep[0] = 1;
  keep[raw.length - 1] = 1;
  _rdpRecurse(raw, 0, raw.length - 1, epsilon2, keep);
  return raw.filter((_, i) => keep[i]);
}

function _rdpRecurse(
  pts: BrushRawPoint[], start: number, end: number,
  epsilon2: number, keep: Uint8Array,
): void {
  if (end - start < 2) return;
  let maxDist = 0;
  let maxIdx = start;

  const sx = pts[start].x, sy = pts[start].y;
  const sb = pts[start].border, sh = pts[start].hardness, sd = pts[start].density;
  const ex = pts[end].x, ey = pts[end].y;
  const eb = pts[end].border, eh = pts[end].hardness, ed = pts[end].density;
  const r3 = ex - sx, r4 = ey - sy;
  const r5 = eb - sb, r6 = eh - sh, r7 = ed - sd;
  const l = r3 * r3 + r4 * r4;

  for (let i = start + 1; i < end; i++) {
    const x = pts[i].x, y = pts[i].y;
    const b = pts[i].border, h = pts[i].hardness, d = pts[i].density;

    // Projection parameter onto the line segment (same as DT)
    const r = (x - sx) * r3 + (y - sy) * r4;
    const p = l > 0 ? r / l : 0;

    let dx: number, dy: number, db: number, dh: number, dd: number;
    if (l === 0 || p < 0) {
      dx = x - sx; dy = y - sy;
      db = b - sb; dh = h - sh; dd = d - sd;
    } else if (p > 1) {
      dx = x - ex; dy = y - ey;
      db = b - eb; dh = h - eh; dd = d - ed;
    } else {
      dx = x - (sx + p * r3);
      dy = y - (sy + p * r4);
      db = b - (sb + p * r5);
      dh = h - (sh + p * r6);
      dd = d - (sd + p * r7);
    }

    const dist = dx * dx + dy * dy + db * db + 0.01 * dh * dh + 0.01 * dd * dd;
    if (dist > maxDist) {
      maxDist = dist;
      maxIdx = i;
    }
  }

  if (maxDist >= epsilon2) {
    keep[maxIdx] = 1;
    _rdpRecurse(pts, start, maxIdx, epsilon2, keep);
    _rdpRecurse(pts, maxIdx, end, epsilon2, keep);
  }
}

/**
 * Convert raw brush points to MaskPointBrush[] with Catmull-Rom control points.
 * Ported from darktable's brush creation finalization.
 * Handles single-point strokes by adding a helper point (matching DT).
 */
export function buildBrushPoints(raw: BrushRawPoint[]): MaskPointBrush[] {
  if (raw.length === 0) return [];

  // DT: if only one point, add a helper node very close to avoid degenerate case
  let pts = raw;
  if (raw.length === 1) {
    pts = [
      raw[0],
      { x: raw[0].x + 0.01, y: raw[0].y - 0.01,
        border: raw[0].border, hardness: raw[0].hardness, density: raw[0].density },
    ];
  }

  // Create brush points with -1 ctrl (to be filled by catmull-rom)
  const brushPts: MaskPointBrush[] = pts.map((r) => ({
    corner: [r.x, r.y] as [number, number],
    ctrl1: [-1, -1] as [number, number],
    ctrl2: [-1, -1] as [number, number],
    border: [r.border, r.border] as [number, number],
    density: r.density,
    hardness: r.hardness,
    state: 1, // NORMAL
  }));

  initBrushCtrlPoints(brushPts);
  return brushPts;
}

/**
 * Auto-compute Catmull-Rom control points for brush nodes (open spline).
 * Matches DT's _brush_init_ctrl_points() exactly: uses mirror phantom points
 * at boundaries instead of index clamping, so endpoint curvature extends naturally.
 */
export function initBrushCtrlPoints(points: MaskPointBrush[]): void {
  const n = points.length;
  if (n < 2) return;

  for (let k = 0; k < n; k++) {
    const p3 = points[k];
    if (!(p3.state & 1)) continue; // only NORMAL

    // Get neighboring points (may be null at boundaries)
    const prev = k > 0 ? points[k - 1] : null;
    const prevprev = k > 1 ? points[k - 2] : null;
    const next = k < n - 1 ? points[k + 1] : null;
    const nextnext = k < n - 2 ? points[k + 2] : null;

    // Mirror phantom points at boundaries (matching DT exactly)
    let p1c: [number, number], p2c: [number, number];
    let p4c: [number, number], p5c: [number, number];

    if (!prev && !prevprev) {
      // First point: both extending points mirror neighborhood
      const mirror: [number, number] = [
        2 * p3.corner[0] - next!.corner[0],
        2 * p3.corner[1] - next!.corner[1],
      ];
      p1c = mirror;
      p2c = mirror;
    } else if (!prevprev) {
      p1c = [
        2 * prev!.corner[0] - p3.corner[0],
        2 * prev!.corner[1] - p3.corner[1],
      ];
      p2c = prev!.corner;
    } else {
      p1c = prevprev.corner;
      p2c = prev!.corner;
    }

    if (!next && !nextnext) {
      // Last point: both extending points mirror neighborhood
      const mirror: [number, number] = [
        2 * p3.corner[0] - prev!.corner[0],
        2 * p3.corner[1] - prev!.corner[1],
      ];
      p4c = mirror;
      p5c = mirror;
    } else if (!nextnext) {
      p4c = next!.corner;
      p5c = [
        2 * next!.corner[0] - p3.corner[0],
        2 * next!.corner[1] - p3.corner[1],
      ];
    } else {
      p4c = next!.corner;
      p5c = nextnext.corner;
    }

    // Catmull-rom → bezier for segment (p2 → p3)
    const bx1 = (-p1c[0] + 6 * p2c[0] + p3.corner[0]) / 6;
    const by1 = (-p1c[1] + 6 * p2c[1] + p3.corner[1]) / 6;
    const bx2 = (p2c[0] + 6 * p3.corner[0] - p4c[0]) / 6;
    const by2 = (p2c[1] + 6 * p3.corner[1] - p4c[1]) / 6;

    if (prev && prev.ctrl2[0] === -1) prev.ctrl2[0] = bx1;
    if (prev && prev.ctrl2[1] === -1) prev.ctrl2[1] = by1;
    p3.ctrl1[0] = bx2;
    p3.ctrl1[1] = by2;

    // Catmull-rom → bezier for segment (p3 → p4)
    const cx1 = (-p2c[0] + 6 * p3.corner[0] + p4c[0]) / 6;
    const cy1 = (-p2c[1] + 6 * p3.corner[1] + p4c[1]) / 6;
    const cx2 = (p3.corner[0] + 6 * p4c[0] - p5c[0]) / 6;
    const cy2 = (p3.corner[1] + 6 * p4c[1] - p5c[1]) / 6;

    if (next && next.ctrl1[0] === -1) next.ctrl1[0] = cx2;
    if (next && next.ctrl1[1] === -1) next.ctrl1[1] = cy2;
    p3.ctrl2[0] = cx1;
    p3.ctrl2[1] = cy1;
  }

  // Endpoints fallback: ctrl1 of first = corner, ctrl2 of last = corner
  if (points[0].ctrl1[0] === -1) {
    points[0].ctrl1[0] = points[0].corner[0];
    points[0].ctrl1[1] = points[0].corner[1];
  }
  if (points[n - 1].ctrl2[0] === -1) {
    points[n - 1].ctrl2[0] = points[n - 1].corner[0];
    points[n - 1].ctrl2[1] = points[n - 1].corner[1];
  }
}
