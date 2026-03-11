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
  DistortionGrid,
} from "../types/protocol";
import { forwardTransform, inverseTransform } from "./distortionGrid";

// ─── DragTarget ─────────────────────────────────────────────────────────────

/** Hit-test result identifying what part of a mask the user is dragging. */
export type DragTarget =
  | { kind: "center"; formid: number }
  | { kind: "radius"; formid: number; pointIndex?: number }
  | { kind: "border"; formid: number; pointIndex?: number }
  | { kind: "rotate"; formid: number };

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

// ─── getDragOps ─────────────────────────────────────────────────────────────

/**
 * Return the drag operations object for a given mask form, or null if the
 * mask type does not support dragging (e.g. path, brush).
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
