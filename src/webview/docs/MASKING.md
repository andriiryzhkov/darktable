# Nova Masking

## 1. How DT Masks Work

### Coordinate System

All mask coordinates are stored as **normalized fractions** of the input image dimensions:
- `points[0]` = x / iwidth  (0.0 to 1.0)
- `points[1]` = y / iheight (0.0 to 1.0)

This is the "raw" coordinate space — independent of any geometric transformations applied by the pixel pipeline.

### Mask Types

| Type | Storage | Key Fields |
| ---- | ------- | ---------- |
| Circle | center + radius + border | `points[0..1]` = center, `points[2]` = radius, `points[3]` = border |
| Ellipse | center + radii + rotation + border | Similar to circle with extra dims |
| Path | array of bezier control points | Each point: `corner[2], ctrl1[2], ctrl2[2], border[2], state` |
| Brush | array of brush strokes | Points with pressure/hardness |
| Gradient | position + rotation + compression | Linear gradient mask |
| Group | container of sub-masks | Boolean operations (union/intersect/exclude) |

### The Distortion Pipeline

When DT renders masks on screen, raw coordinates are transformed through all geometric modules in the pixel pipeline (lens correction, crop, rotation, perspective, rawprepare, etc.) via `dt_dev_distort_transform_plus()`. This maps raw normalized coordinates to output/screen coordinates.

Every RAW image has at least `rawprepare` distortion (sensor border crop), so the transform is **always** needed — there is no subset of images where it can be skipped.

## 2. Distortion Grid Approach

Nova uses **Option C: precomputed distortion grid** — a 64×64 grid of coordinate transforms sent from server to client.

### Why This Approach

The server has `dt_dev_distort_transform_plus(dev, pipe, ...)` which already accepts explicit `dev` and `pipe` parameters — no core DT modifications needed. The grid is computed once per pipeline completion (~32KB transfer) and enables the client to transform any coordinate instantly via bilinear interpolation.

This avoids:

- Forking core mask functions (Options A/F)
- Swapping globals (Option B — thread-unsafe)
- Per-coordinate server round-trips (Option D — latency)
- Double pipeline processing (Option E — expensive)

### Server Side

`dt_server_develop_get_distortion_grid()` in `src/server/server_develop.c`:

1. Generates a 64×64 regular grid in raw normalized space, scales to pixel coords
2. Calls `dt_dev_distort_transform_plus()` with `session->dev.full.pipe` → forward grid (raw → output)
3. Generates a 64×64 regular grid in output normalized space, scales to pixel coords
4. Calls `dt_dev_distort_backtransform_plus()` → inverse grid (output → raw)
5. Normalizes both grids to [0,1] output space and returns as JSON

The grid is fetched by the client on every `preview_ready` event (pipeline reprocessed).

### Client Side

`ui/src/lib/distortionGrid.ts` provides:

- **`gridTransform(grid, gw, gh, x, y)`** — bilinear interpolation in a grid
- **`forwardTransform(grid, x, y)`** — raw normalized → output normalized
- **`inverseTransform(grid, x, y)`** — output normalized → raw normalized
- **`generateCirclePolyline(grid, center, radius, border)`** — generates circle polyline from raw-space params
- **`generateEllipsePolyline(grid, center, radius, rotation, border, flags)`** — ellipse polyline
- **`transformPathPoints(grid, points)`** — transforms path bezier control points
- **`transformBrushPoints(grid, points)`** — transforms brush stroke points
- **`generateGradientPolylines(grid, anchor, rotation, compression)`** — gradient lines

All polyline generators take raw-space mask parameters, sample points in raw space, and transform each point through the forward grid to output space. The result is a dense polyline in output normalized [0,1] coordinates ready for canvas rendering.

## 3. Mask Overlay Architecture

### File: `ui/src/components/Darkroom/MaskOverlay.tsx`

Canvas overlay that renders mask shapes and handles interactive editing.

### Rendering Pipeline

```
Server: form.points (raw normalized [0,1])
    │
    ▼
Client: distortionGrid.forward (64×64 bilinear interpolation)
    │
    ▼
Output normalized polylines [0,1]
    │
    ▼
Canvas pixel coordinates (× canvas width/height)
    │
    ▼
dualStroke rendering (dark bg + bright fg, matching DT style)
```

All mask types require the distortion grid to render. If no grid is available, masks are not drawn.

### Key Functions

| Function | Purpose |
| -------- | ------- |
| `drawForm()` | Dispatches to per-type polyline generators + canvas rendering |
| `hitTestForm()` | Dispatches to per-type hit testing using generated polylines |
| `hitTestDragTarget()` | Tests proximity to center/radius/border handles |
| `drawCirclePolyline()` | Renders pre-generated circle polyline with dual-stroke |
| `drawEllipsePolyline()` | Renders pre-generated ellipse polyline |
| `drawPath()` | Renders transformed path bezier with border |
| `drawBrush()` | Renders transformed brush strokes |
| `drawGradientPolyline()` | Renders gradient lines with arrow |

## 4. Creation Flow

Matches DT's `gui->creation` lifecycle.

### States

1. **Tool selected** (`creationTool` + `creationModule`): User clicked "add circle/ellipse" in blending toolbar. Canvas shows crosshair cursor. No form exists yet on server.

2. **Form created, following cursor** (`creatingMaskId`): User clicked on canvas (or toolbar button called `createMask` with `_creation: true`). Server creates the form with initial params. The form follows the cursor — on each `mousemove`, `inverseTransform` converts output-space cursor to raw-space center, and `previewMaskParam` sends coalesced updates to server.

3. **Cursor outside image during creation**: Mask centers at (0.5, 0.5) in output space. Slider adjustments (radius, border) use this centered position until cursor returns to the image.

4. **Click to commit** (`saveCreation`): Final raw-space center is sent to server as a history-writing update. Creation mode exits, form becomes selected for editing.

5. **Escape to cancel** (`cancelCreation`): Form is deleted from server, creation mode exits.

### Key Insight: Nova vs DT Creation

DT creates the form in-memory during `gui->creation` and only writes to history on `save_creation`. Nova must create the form on the server immediately (for live preview with the pixel pipeline), then commits the final position. The `_creation` flag on `createMask` signals that the form enters creation mode (`creatingMaskId`) rather than being immediately committed.

## 5. Drag Editing

### Handle Types

| DragTarget.kind | DT equivalent | What it controls |
| --------------- | ------------- | ---------------- |
| `"center"` | `gui->form_dragging` | Entire mask position |
| `"radius"` | `gui->point_dragging` | Mask size (radius for circle, axis radii for ellipse) |
| `"border"` | `gui->point_border_dragging` | Feather/border width |

### Drag Flow

1. **mousedown**: `hitTestDragTarget` generates polylines from grid to find handle positions in output space. If a handle is near the click, captures `origCenter`/`origRadius`/`origBorder` from `form.points` (raw space).

2. **mousemove**: Converts mouse position to raw space via `inverseTransform`. Computes delta from drag start (also in raw space). Updates `form.points` directly for instant visual feedback (grid regenerates polylines from updated raw-space params).

3. **mouseup**: Commits final `form.points` values to server via `updateMask`.

All drag math operates in raw space. The grid handles the visual mapping to screen.

## 6. Coordinate Space Summary

| Space | Range | Used by |
| ----- | ----- | ------- |
| Raw normalized | [0,1] × [0,1] over input image | `form.points`, server storage, `inverseTransform` output |
| Output normalized | [0,1] × [0,1] over processed image | Polylines, canvas rendering, `forwardTransform` output |
| Canvas pixels | [0,w] × [0,h] | Mouse events, `drawForm`, `hitTestForm` |

Conversions:

- **Raw → Output**: `forwardTransform(grid, x, y)` — for rendering
- **Output → Raw**: `inverseTransform(grid, x, y)` — for mouse interaction
- **Output → Canvas**: multiply by canvas width/height
- **Canvas → Output**: divide by canvas width/height

## 7. Server API

| Endpoint | Purpose |
| -------- | ------- |
| `develop.get_distortion_grid` | Returns 64×64 forward + inverse grids |
| `develop.get_masks` | Returns all mask forms with raw-space `points` |
| `develop.create_mask` | Creates a new mask form (circle/ellipse) |
| `develop.update_mask` | Updates mask params; `preview_only` flag skips history |
| `develop.assign_mask` | Assigns a form to a module's blend group |

### Key Files

| File | Role |
| ---- | ---- |
| `src/server/server_develop.c` | Server-side mask CRUD + distortion grid computation |
| `ui/src/lib/distortionGrid.ts` | Client-side grid interpolation + polyline generation |
| `ui/src/components/Darkroom/MaskOverlay.tsx` | Canvas rendering, hit testing, creation + drag editing |
| `ui/src/stores/developStore.ts` | State management (grid, creation state, mask forms) |
| `ui/src/components/modules/BlendingToolbar.tsx` | Mask type buttons, blend mode controls |

## 8. Terminology

Nova masking code aligns with the GTK darktable C codebase (`src/develop/masks/`).

| Nova term | DT C equivalent | Description |
| --------- | -------------- | ----------- |
| `creatingMaskId` | `gui->creation` | A mask form is being created — follows cursor, not yet committed to history |
| `creationTool` | `gui->creation` (type) | Which mask type is selected for creation (`"circle"`, `"ellipse"`) |
| `creationModule` | `gui->creation_module` | The IOP module that initiated mask creation |
| `saveCreation()` | `dt_masks_gui_form_save_creation()` | Commit the mask being created (write to history, exit creation mode) |
| `cancelCreation()` | Right-click during `gui->creation` | Cancel mask creation and delete the uncommitted form |
| `resetCreation()` | Deselect creation tool | Clear tool selection without deleting any form (pre-creation state) |
| `creationCursorRef` | `gui->posx` / `gui->posy` | Cursor position tracked during creation for center placement |
| `DragTarget.kind: "center"` | `gui->form_dragging` | Entire mask is being dragged (center handle) |
| `DragTarget.kind: "radius"` | `gui->point_dragging` | Radius handle is being dragged |
| `DragTarget.kind: "border"` | `gui->point_border_dragging` / `gui->feather_dragging` | Border/feather handle is being dragged |
| `distortionGrid` | `dt_dev_distort_transform_plus()` | Precomputed 64×64 coordinate transform grid |
| `forwardTransform()` | `DT_DEV_TRANSFORM_DIR_ALL` | Raw normalized coords → output normalized coords |
| `inverseTransform()` | `DT_DEV_TRANSFORM_DIR_BACK_ALL` | Output normalized coords → raw normalized coords |

## 9. Historical Analysis

The earlier analysis of approaches (Options A–F) and the rawprepare discovery are preserved in git history. Key conclusions that informed the current design:

- `dt_dev_distort_transform_plus()` already accepts explicit `dev` and `pipe` — no core DT modifications needed
- Every RAW image has `rawprepare` distortion — transforms cannot be skipped
- Server-side polyline generation (Option F) was rejected in favor of client-side generation with grid (Option C) for instant interactive feedback
- The grid approach solves both rendering and interactive editing uniformly — forward transform for display, inverse transform for mouse-to-raw conversion
