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
  curvature = 0,
): {
  anchor: [number, number];
  rotation: number;
  main_polyline: number[];
  border_polyline1: number[];
  border_polyline2: number[];
} {
  const iw = grid.iwidth;
  const ih = grid.iheight;
  const scale = Math.sqrt(iw * iw + ih * ih);

  // Match DT's _gradient_get_points: v = -(rotation) * PI/180
  const v = -(rotation * Math.PI) / 180;
  const cosv = Math.cos(v);
  const sinv = Math.sin(v);

  // Border offset: compression * diagonal, perpendicular to gradient line
  const borderOffset = compression * scale;
  const v1 = -(rotation - 90) * (Math.PI / 180);
  const v2 = -(rotation + 90) * (Math.PI / 180);

  // Anchor in raw pixels
  const ax_px = anchor[0] * iw;
  const ay_px = anchor[1] * ih;

  // Border anchor centers (offset perpendicular to gradient direction)
  const b1cx = (ax_px + borderOffset * Math.cos(v1)) / iw;
  const b1cy = (ay_px + borderOffset * Math.sin(v1)) / ih;
  const b2cx = (ax_px + borderOffset * Math.cos(v2)) / iw;
  const b2cy = (ay_px + borderOffset * Math.sin(v2)) / ih;

  // Transform anchor
  const tAnchor = forwardTransform(grid, anchor[0], anchor[1]);

  // Estimate transformed rotation from reference point (matches server approach)
  // Server: atan2(pixel_dx / odim, pixel_dy / odim) — both divided by same value, so equivalent to pixel-space atan2
  const refDist = 0.1 * Math.min(iw, ih);
  const rotRad = rotation * (Math.PI / 180);
  const refX = (ax_px + refDist * Math.sin(rotRad)) / iw;
  const refY = (ay_px + refDist * Math.cos(rotRad)) / ih;
  const [trx, try_] = forwardTransform(grid, refX, refY);
  const pw = grid.processed_width;
  const ph = grid.processed_height;
  const tRot = Math.atan2((trx - tAnchor[0]) * pw, (try_ - tAnchor[1]) * ph) * (180 / Math.PI);

  // Parametric x range — matches DT: if |curvature| > 1, limit range
  const xstart = Math.abs(curvature) > 1 ? -Math.sqrt(1 / Math.abs(curvature)) : -1;
  const nSamples = Math.max(50, Math.round(scale));
  const xdelta = -2 * xstart / (nSamples > 1 ? nSamples - 1 : 1);

  const mainPoly: number[] = [];
  const border1Poly: number[] = [];
  const border2Poly: number[] = [];

  // Centers for 3 lines: main, border1, border2
  const centers: [number, number][] = [
    [anchor[0], anchor[1]],
    [b1cx, b1cy],
    [b2cx, b2cy],
  ];
  const polys = [mainPoly, border1Poly, border2Poly];

  for (let line = 0; line < 3; line++) {
    const cx = centers[line][0];
    const cy = centers[line][1];
    for (let i = 0; i < nSamples; i++) {
      const xi = xstart + i * xdelta;
      const yi = curvature * xi * xi;
      const xii = (cosv * xi + sinv * yi) * scale;
      const yii = (sinv * xi - cosv * yi) * scale;
      const rawX = cx + xii / iw;
      const rawY = cy + yii / ih;
      const [tx, ty] = forwardTransform(grid, rawX, rawY);
      polys[line].push(tx, ty);
    }
  }

  return {
    anchor: tAnchor,
    rotation: tRot,
    main_polyline: mainPoly,
    border_polyline1: border1Poly,
    border_polyline2: border2Poly,
  };
}
