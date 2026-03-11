/**
 * Client-side distortion grid interpolation and mask polyline generation.
 *
 * The grid maps coordinates between raw image space and output/screen space
 * using bilinear interpolation of a precomputed transform grid from the server.
 * This enables instant mask shape computation without server round-trips.
 */

import type { DistortionGrid, MaskPointPath, MaskPointBrush } from "../types/protocol";

// ─── Bilinear interpolation ─────────────────────────────────────────────────

/**
 * Transform a normalized coordinate through the distortion grid using
 * bilinear interpolation. Works for both forward (raw→output) and
 * inverse (output→raw) grids.
 *
 * @param grid - Flat [x,y,...] array from DistortionGrid.forward or .inverse
 * @param gw - Grid width
 * @param gh - Grid height
 * @param x - Input x in normalized [0,1] space
 * @param y - Input y in normalized [0,1] space
 * @returns [outX, outY] in the target normalized [0,1] space
 */
export function gridTransform(
  grid: number[],
  gw: number,
  gh: number,
  x: number,
  y: number,
): [number, number] {
  // Scale to grid space
  const gx = x * (gw - 1);
  const gy = y * (gh - 1);

  // Clamp to valid grid range
  const i = Math.max(0, Math.min(gw - 2, Math.floor(gx)));
  const j = Math.max(0, Math.min(gh - 2, Math.floor(gy)));
  const fx = gx - i;
  const fy = gy - j;

  // Four corner indices (each point is 2 floats: x, y)
  const idx00 = (j * gw + i) * 2;
  const idx10 = (j * gw + i + 1) * 2;
  const idx01 = ((j + 1) * gw + i) * 2;
  const idx11 = ((j + 1) * gw + i + 1) * 2;

  // Bilinear interpolation for x component
  const topX = grid[idx00] + (grid[idx10] - grid[idx00]) * fx;
  const botX = grid[idx01] + (grid[idx11] - grid[idx01]) * fx;
  const outX = topX + (botX - topX) * fy;

  // Bilinear interpolation for y component
  const topY = grid[idx00 + 1] + (grid[idx10 + 1] - grid[idx00 + 1]) * fx;
  const botY = grid[idx01 + 1] + (grid[idx11 + 1] - grid[idx01 + 1]) * fx;
  const outY = topY + (botY - topY) * fy;

  return [outX, outY];
}

/**
 * Transform a point from raw normalized space to output normalized space.
 */
export function forwardTransform(
  grid: DistortionGrid,
  x: number,
  y: number,
): [number, number] {
  return gridTransform(grid.forward, grid.width, grid.height, x, y);
}

/**
 * Transform a point from output normalized space to raw normalized space.
 */
export function inverseTransform(
  grid: DistortionGrid,
  x: number,
  y: number,
): [number, number] {
  return gridTransform(grid.inverse, grid.width, grid.height, x, y);
}

// ─── Circle polyline generation ─────────────────────────────────────────────

/**
 * Generate a circle mask polyline using the distortion grid.
 * Matches the server-side algorithm in server_develop.c.
 *
 * @returns { center, main_polyline, border_polyline } in output normalized [0,1] space
 */
export function generateCirclePolyline(
  grid: DistortionGrid,
  center: [number, number],
  radius: number,
  border: number,
): {
  center: [number, number];
  main_polyline: number[];
  border_polyline: number[];
} {
  const iw = grid.iwidth;
  const ih = grid.iheight;
  const dim = Math.min(iw, ih);

  // Sample count matches server: max(10, circumference in pixels)
  const r = radius * dim;
  const nSamples = Math.max(10, Math.round(2 * Math.PI * r));

  const [cx, cy] = center;
  const rb = (radius + border) * dim;

  // Transform center
  const tCenter = forwardTransform(grid, cx, cy);

  // Generate and transform main circle + border circle points
  const mainPoly: number[] = [];
  const borderPoly: number[] = [];

  for (let i = 0; i < nSamples; i++) {
    const angle = (2 * Math.PI * i) / nSamples;
    const ca = Math.cos(angle);
    const sa = Math.sin(angle);

    // Main circle point in raw normalized space
    const mx = cx + (r * ca) / iw;
    const my = cy + (r * sa) / ih;
    const [tmx, tmy] = forwardTransform(grid, mx, my);
    mainPoly.push(tmx, tmy);

    // Border circle point in raw normalized space
    const bx = cx + (rb * ca) / iw;
    const by = cy + (rb * sa) / ih;
    const [tbx, tby] = forwardTransform(grid, bx, by);
    borderPoly.push(tbx, tby);
  }

  return { center: tCenter, main_polyline: mainPoly, border_polyline: borderPoly };
}

// ─── Ellipse polyline generation ────────────────────────────────────────────

/**
 * Generate an ellipse mask polyline using the distortion grid.
 * Matches the server-side algorithm in server_develop.c.
 *
 * @returns { center, main_polyline, border_polyline } in output normalized [0,1] space
 */
export function generateEllipsePolyline(
  grid: DistortionGrid,
  center: [number, number],
  radiusArr: [number, number],
  rotation: number,
  border: number,
  flags: number,
): {
  center: [number, number];
  main_polyline: number[];
  border_polyline: number[];
} {
  const iw = grid.iwidth;
  const ih = grid.iheight;
  const dim = Math.min(iw, ih);

  // Match DT: swap axes so a >= b
  const v1 = rotation * (Math.PI / 180);
  const v2 = v1 - Math.PI / 2;
  let a: number, bAx: number, v: number;
  if (radiusArr[0] >= radiusArr[1]) {
    a = radiusArr[0] * dim;
    bAx = radiusArr[1] * dim;
    v = v1;
  } else {
    a = radiusArr[1] * dim;
    bAx = radiusArr[0] * dim;
    v = v2;
  }

  const sinv = Math.sin(v);
  const cosv = Math.cos(v);

  // Border radii (match DT's proportional vs absolute)
  const prop = (flags & 1) !== 0; // DT_MASKS_ELLIPSE_PROPORTIONAL
  let ab: number, bb: number;
  if (prop) {
    ab = a * (1 + border);
    bb = bAx * (1 + border);
  } else {
    ab = a + border * dim;
    bb = bAx + border * dim;
  }

  // Sample count: Ramanujan approximation (match DT)
  const lambda = (a - bAx) / (a + bAx + 1e-10);
  const nEl = Math.max(
    100,
    Math.round(
      (Math.PI *
        (a + bAx) *
        (1 + (3 * lambda * lambda) / (10 + Math.sqrt(4 - 3 * lambda * lambda)))) /
        10,
    ),
  );

  const [cx, cy] = center;

  // Transform center
  const tCenter = forwardTransform(grid, cx, cy);

  // Generate and transform main ellipse + border ellipse points
  const mainPoly: number[] = [];
  const borderPoly: number[] = [];

  for (let i = 0; i < nEl; i++) {
    const alpha = (i * 2 * Math.PI) / nEl;
    const cosA = Math.cos(alpha);
    const sinA = Math.sin(alpha);

    // Main ellipse point in raw pixel space, then to normalized
    const mx = cx + (a * cosA * cosv - bAx * sinA * sinv) / iw;
    const my = cy + (a * cosA * sinv + bAx * sinA * cosv) / ih;
    const [tmx, tmy] = forwardTransform(grid, mx, my);
    mainPoly.push(tmx, tmy);

    // Border ellipse point
    const bpx = cx + (ab * cosA * cosv - bb * sinA * sinv) / iw;
    const bpy = cy + (ab * cosA * sinv + bb * sinA * cosv) / ih;
    const [tbx, tby] = forwardTransform(grid, bpx, bpy);
    borderPoly.push(tbx, tby);
  }

  return { center: tCenter, main_polyline: mainPoly, border_polyline: borderPoly };
}

// ─── Path control point transformation ──────────────────────────────────────

/**
 * Transform path mask control points from raw normalized space to output space.
 * Corner, ctrl1, ctrl2 are each transformed through the grid.
 * Border values are preserved (border computation happens in screen space).
 */
export function transformPathPoints(
  grid: DistortionGrid,
  pts: MaskPointPath[],
): MaskPointPath[] {
  return pts.map((p) => ({
    corner: forwardTransform(grid, p.corner[0], p.corner[1]),
    ctrl1: forwardTransform(grid, p.ctrl1[0], p.ctrl1[1]),
    ctrl2: forwardTransform(grid, p.ctrl2[0], p.ctrl2[1]),
    border: p.border,
    state: p.state,
  }));
}

// ─── Brush control point transformation ─────────────────────────────────────

/**
 * Transform brush mask control points from raw normalized space to output space.
 */
export function transformBrushPoints(
  grid: DistortionGrid,
  pts: MaskPointBrush[],
): MaskPointBrush[] {
  return pts.map((p) => ({
    corner: forwardTransform(grid, p.corner[0], p.corner[1]),
    ctrl1: forwardTransform(grid, p.ctrl1[0], p.ctrl1[1]),
    ctrl2: forwardTransform(grid, p.ctrl2[0], p.ctrl2[1]),
    border: p.border,
    density: p.density,
    hardness: p.hardness,
    state: p.state,
  }));
}

// ─── Gradient polyline generation ───────────────────────────────────────────

/**
 * Generate gradient mask polylines using the distortion grid.
 * Samples points along the main line and two border lines in raw space,
 * transforms each through the grid.
 */
export function generateGradientPolylines(
  grid: DistortionGrid,
  anchor: [number, number],
  rotation: number,
  compression: number,
): {
  anchor: [number, number];
  rotation: number;
  main_polyline: number[];
  border_polyline1: number[];
  border_polyline2: number[];
} {
  const iw = grid.iwidth;
  const ih = grid.iheight;
  const rot = (rotation * Math.PI) / 180;

  // Gradient direction in raw pixel space: (sin(rot), cos(rot))
  const gdx = Math.sin(rot);
  const gdy = Math.cos(rot);

  // Line direction (perpendicular to gradient) in raw pixel space
  const ldx = Math.cos(rot);
  const ldy = -Math.sin(rot);

  // Border offset in raw pixels: compression * diagonal
  const diag = Math.sqrt(iw * iw + ih * ih);
  const borderOffset = compression * diag;

  // Anchor in raw pixels
  const ax_px = anchor[0] * iw;
  const ay_px = anchor[1] * ih;

  // Transform anchor
  const tAnchor = forwardTransform(grid, anchor[0], anchor[1]);

  // Estimate transformed rotation from local gradient direction
  const delta = 0.001;
  const [gx1, gy1] = forwardTransform(
    grid,
    anchor[0] + (gdx * delta) / iw,
    anchor[1] + (gdy * delta) / ih,
  );
  const tRot = Math.atan2(gx1 - tAnchor[0], gy1 - tAnchor[1]) * (180 / Math.PI);

  // Sample points along all 3 lines in raw pixel space, transform to output
  const nSamples = 50;
  const lineLen = diag * 1.5;

  const mainPoly: number[] = [];
  const border1Poly: number[] = [];
  const border2Poly: number[] = [];

  for (let i = 0; i <= nSamples; i++) {
    const t = -lineLen + (2 * lineLen * i) / nSamples;

    // Main line point (raw pixels → raw normalized → grid transform)
    const mx = (ax_px + ldx * t) / iw;
    const my = (ay_px + ldy * t) / ih;
    const [tmx, tmy] = forwardTransform(grid, mx, my);
    mainPoly.push(tmx, tmy);

    // Border line 1 (positive offset along gradient direction)
    const b1x = (ax_px + gdx * borderOffset + ldx * t) / iw;
    const b1y = (ay_px + gdy * borderOffset + ldy * t) / ih;
    const [tb1x, tb1y] = forwardTransform(grid, b1x, b1y);
    border1Poly.push(tb1x, tb1y);

    // Border line 2 (negative offset)
    const b2x = (ax_px - gdx * borderOffset + ldx * t) / iw;
    const b2y = (ay_px - gdy * borderOffset + ldy * t) / ih;
    const [tb2x, tb2y] = forwardTransform(grid, b2x, b2y);
    border2Poly.push(tb2x, tb2y);
  }

  return {
    anchor: tAnchor,
    rotation: tRot,
    main_polyline: mainPoly,
    border_polyline1: border1Poly,
    border_polyline2: border2Poly,
  };
}
