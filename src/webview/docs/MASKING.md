# Masking API Analysis & Nova Integration Plan

## 1. How DT Masks Work

### Coordinate System

All mask coordinates are stored as **normalized fractions** of the input image dimensions:
- `points[0]` = x / iwidth  (0.0 to 1.0)
- `points[1]` = y / iheight (0.0 to 1.0)

This is the "raw" coordinate space — independent of any geometric transformations applied by the pixel pipeline.

### Mask Types

| Type | Storage | Key Fields |
|------|---------|------------|
| Circle | center + radius + border | `points[0..1]` = center, `points[2]` = radius, `points[3]` = border |
| Ellipse | center + radii + rotation + border | Similar to circle with extra dims |
| Path | array of bezier control points | Each point: `corner[2], ctrl1[2], ctrl2[2], border[2], state` |
| Brush | array of brush strokes | Points with pressure/hardness |
| Gradient | position + rotation + compression | Linear gradient mask |
| Group | container of sub-masks | Boolean operations (union/intersect/exclude) |
| Object | SAM/SAM2 AI segmentation | Prompt points + generated mask |

### GUI Point Generation Pipeline

When DT renders masks on screen, it goes through this pipeline:

```
Raw coordinates (normalized)
    │
    ▼
dt_masks_get_points_border()     ← per-type function generates dense polylines
    │                               (e.g., circle → 360 points, path → sampled bezier)
    ▼
dt_masks_get_image_size()        ← gets iwidth/iheight from preview_pipe
    │
    ▼
dt_dev_distort_transform_plus()  ← applies all geometric module transforms
    │                               (lens correction, crop, rotation, perspective, etc.)
    ▼
Screen coordinates (preview_pipe output space)
    │
    ▼
Cairo rendering                  ← drawn on GTK overlay
```

### The Distortion Pipeline

`dt_dev_distort_transform_plus(dev, pipe, iop_order, direction, points, count)`:

1. Iterates through pipe nodes from `iop_order` to the end
2. For each module that has `distort_transform` (or `distort_backtransform`):
   - Calls the module's transform function on the point array
   - This modifies coordinates **in-place**
3. Modules that distort: `lens`, `clipping`, `crop`, `rotatepixels`, `scalepixels`, `flip`, `ashift`, `retouch`

Key detail: the function signature uses the **global** `darktable.develop` and `dev->preview_pipe` — there's no way to pass a different pipe.

### Critical Global Dependencies

```c
// masks.h — hardcoded to global state
static inline void dt_masks_get_image_size(float *width, float *height, ...)
{
    *width = darktable.develop->preview_pipe->backbuf_width;   // ← global
    *height = darktable.develop->preview_pipe->backbuf_height; // ← global
}

// develop.c — uses dev->preview_pipe
int dt_dev_distort_transform(dt_develop_t *dev, ...)
{
    return dt_dev_distort_transform_plus(dev, dev->preview_pipe, ...);  // ← hardcoded pipe
}
```

## 2. Why Nova Masks Are Misaligned

Nova renders masks using **raw normalized coordinates** directly mapped to the displayed image. But the displayed image has already been through the full pixel pipeline, which includes geometric transforms.

Example: An image with lens correction active:
- Input: 6384×4182 (ratio 1.526)
- Output: 980×653 (ratio 1.501)
- A mask at `(0.5, 0.5)` in raw coords maps to the center of the **input** image
- But after lens correction warps the geometry, that point may render at a different pixel in the **output** image
- Nova places the mask at the center of the output image — wrong position

**This affects ALL mask types, not just circles.** Any image with active geometric modules (lens correction, crop, rotation, perspective correction, etc.) will show misalignment.

## 3. Server Architecture Constraints

The server has its own `dt_develop_t` per session (`session->dev`), with:
- `session->dev.full.pipe` — fully processed pipeline (used for rendering)
- `session->dev.preview_pipe` — exists but is **never processed** (dimensions always 0)

The mask API functions (`dt_masks_get_points_border`, `dt_dev_distort_transform`) use:
- `darktable.develop` — the **GUI singleton**, not the server's dev
- `dev->preview_pipe` — which in the server context has zero dimensions

Directly calling these functions from the server crashes or produces garbage.

## 4. Possible Approaches

### Option A: Fork Mask Functions With Explicit Pipe Parameter

Create server-specific versions of the mask point generation functions that accept an explicit pipe parameter instead of using globals.

```c
// New API
int dt_masks_get_points_border_ext(dt_develop_t *dev, dt_dev_pixelpipe_t *pipe,
                                    dt_masks_form_t *form, float **points, int *count,
                                    float **border, int *border_count);
```

**Pros:**
- Clean, correct approach
- Full accuracy — identical to DT's own rendering
- Works for all mask types including path borders

**Cons:**
- Requires modifying core mask code (every mask type's `get_points` function)
- Large surface area of changes across `circle.c`, `path.c`, `ellipse.c`, `brush.c`, `gradient.c`
- Each per-type function also calls `dt_masks_get_image_size()` internally — all call sites need updating
- Risk of divergence if upstream DT updates these functions
- Distortion transform also needs a pipe-explicit version

**Verdict:** Correct but invasive. High maintenance burden.

### Option B: Generate GUI Points After Pipeline Completion

After `session->dev.full.pipe` finishes processing, temporarily swap globals to point at the server's dev/pipe, call the standard mask functions, then restore.

```c
// In server's pipeline completion callback
dt_develop_t *saved = darktable.develop;
darktable.develop = &session->dev;
// ... generate gui_points using standard API ...
darktable.develop = saved;
```

**Pros:**
- No changes to core mask code
- Full accuracy
- Leverages existing, tested code paths

**Cons:**
- **Thread safety nightmare** — `darktable.develop` is a global singleton; swapping it while other threads may read it is a data race
- `preview_pipe` dimensions still need to be set (server doesn't process preview_pipe)
- Pipeline node list (`pipe->nodes`) may differ between full and preview pipes
- Requires careful timing — must happen after pipe completion

**Verdict:** Fragile and unsafe. Only viable with a mutex around all `darktable.develop` access (impractical).

### Option C: Compute Distortion Map and Send to Client

Generate a distortion lookup table (grid of transformed points) on the server and send it to the client. The client applies the distortion to raw mask coordinates before rendering.

```c
// Server generates a grid, e.g., 64×64 control points
for (int y = 0; y < grid_h; y++)
    for (int x = 0; x < grid_w; x++) {
        points[i*2]   = (float)x / (grid_w - 1) * iwidth;
        points[i*2+1] = (float)y / (grid_h - 1) * iheight;
    }
dt_dev_distort_transform_plus(dev, pipe, 0.0, DT_DEV_TRANSFORM_DIR_ALL, points, n);
// Send grid + transformed points to client
```

Client-side: bilinear interpolation in the grid to transform any mask coordinate.

**Pros:**
- One-time computation per image/history change
- Small data transfer (~32KB for 64×64 grid)
- Client can transform any coordinate instantly
- No per-mask-type code changes needed
- Works for all mask types uniformly

**Cons:**
- Same global state problem for `dt_dev_distort_transform_plus`
- Grid resolution limits accuracy (may be noticeable near strong distortions)
- Complex client-side interpolation code
- Need to regenerate grid when pipeline changes
- Doesn't solve border generation (only point positions)

**Verdict:** Elegant for coordinate transforms but doesn't solve the full problem (border computation still needs DT-side code). Also still hits the global state issue.

### Option D: Add `develop.distort_points` API Endpoint

Create a new server endpoint that accepts an array of points and returns them transformed through the distortion pipeline.

```
POST /develop/distort_points
{
  "session_id": "dev-001",
  "points": [[0.5, 0.5], [0.3, 0.7], ...],
  "direction": "forward",
  "iop_order": 0.0
}
→ { "points": [[0.52, 0.49], [0.31, 0.69], ...] }
```

**Pros:**
- Clean API boundary — client sends raw coords, gets back transformed coords
- Can be used for any mask type
- No changes to core mask code (just wraps existing function)
- Client remains in control of rendering

**Cons:**
- **Same global state problem** — `dt_dev_distort_transform_plus` uses globals
- Round-trip latency for every mask update (hover, drag, create)
- Doesn't help with border generation
- Need to call for every mask form's points individually

**Verdict:** Good API design but doesn't solve the fundamental global state problem, and the latency makes interactive editing sluggish.

### Option E: Process preview_pipe in Server

Set up and process `session->dev.preview_pipe` in the server, making the standard mask API work directly.

**Pros:**
- Everything just works — standard API, no modifications needed

**Cons:**
- **Double the processing cost** — full pipe + preview pipe per image
- preview_pipe setup code is deeply entangled with GTK GUI code
- Significant memory overhead
- Server architecture explicitly avoided this for good reasons

**Verdict:** Rejected. Too expensive, too complex.

### Option F: Hybrid — Server-Side GUI Point Generation with Isolated Dev Context (Recommended)

Create a dedicated function that generates mask GUI points using the server's own pipe, by carefully providing the needed context without touching globals.

Key insight: We don't need to swap globals. We need to:
1. Fork only `dt_masks_get_image_size()` to accept explicit dimensions
2. Fork only `dt_dev_distort_transform_plus()` to accept explicit pipe
3. Call per-type `get_points` functions which internally use #1
4. Call `distort_transform` which internally uses #2

This is a smaller fork than Option A — we only need to modify the **leaf functions** that access globals, not every mask type.

```c
// New thread-safe wrapper
int dt_masks_generate_gui_points(dt_develop_t *dev,
                                  dt_dev_pixelpipe_t *pipe,
                                  dt_masks_form_t *form,
                                  float **points, int *points_count,
                                  float **border, int *border_count)
{
    // Uses pipe->iwidth/iheight instead of preview_pipe
    // Calls per-type get_points with explicit dimensions
    // Transforms through pipe's distortion chain
}
```

**Implementation steps:**
1. Add `_ext` variants of `dt_masks_get_image_size` and `dt_dev_distort_transform_plus` that take explicit pipe parameter
2. Add `_ext` variants of per-type `get_points` that use the explicit versions
3. Create `dt_masks_generate_gui_points()` that orchestrates everything
4. Call from server after pipe completion, include results in `get_masks` response
5. Client renders pre-transformed points directly (code already exists in MaskOverlay.tsx)

**Pros:**
- Thread-safe — no global state mutation
- Full accuracy — identical math to DT
- Moderate code changes — only touching leaf functions + adding wrappers
- Clean separation — server generates, client renders
- No latency issues — points computed once after pipe completion

**Cons:**
- Still requires forking some core functions
- Must be kept in sync with upstream changes to mask math
- Per-type `get_points` functions have internal `dt_masks_get_image_size` calls that need updating

**Verdict:** Best balance of correctness, safety, and implementation effort.

## 5. Recommended Implementation Plan

### Phase 1: Infrastructure (Foundation)

1. **Add `dt_dev_distort_transform_plus_ext()`** in `src/develop/develop.c`:
   - Same as `dt_dev_distort_transform_plus()` but takes explicit `dt_dev_pixelpipe_t *pipe`
   - The original function becomes a thin wrapper calling `_ext` with `dev->preview_pipe`

2. **Add `dt_masks_get_image_size_ext()`** in `src/develop/masks/masks.c`:
   - Takes explicit `float iw, float ih` instead of reading globals
   - Original becomes wrapper

### Phase 2: Per-Type Point Generation

3. **Add `_get_points_ext` for each mask type** that accepts explicit dimensions:
   - `_circle_get_points_ext()` — simplest, start here
   - `_ellipse_get_points_ext()`
   - `_path_get_points_border_ext()` — most complex (bezier + border)
   - `_brush_get_points_border_ext()`
   - `_gradient_get_points_ext()`

4. **Create `dt_masks_generate_gui_points()`** — the orchestrator function

### Phase 3: Server Integration

5. **Call from server** in `_server_cmd_get_masks()` after pipe completion:
   ```c
   dt_masks_generate_gui_points(&session->dev, session->dev.full.pipe,
                                 form, &gui_pts, &gui_count,
                                 &gui_border, &gui_border_count);
   ```

6. **Serialize gui_points/gui_border** in JSON response

### Phase 4: Client Rendering

7. **MaskOverlay.tsx already has `drawGuiForm()` and `hitTestGuiForm()`** — just needs data
8. Verify coordinate mapping: gui_points will be in pipe output space, need to map to canvas

### Phase 5: Interactive Editing

9. For mask creation/editing, use the `distort_points` endpoint approach (Option D) for real-time coordinate transform during mouse interaction
10. On mouse-up / edit complete, re-generate gui_points via pipe completion

## 6. Risk Assessment

| Risk | Mitigation |
|------|------------|
| Upstream mask code changes break `_ext` forks | Keep forks minimal; `_ext` functions should delegate to shared math |
| Performance of gui_points generation | Circle/ellipse are trivial; path/brush may need profiling |
| Pipe not ready when masks requested | Only generate gui_points after pipe completion; flag in response |
| Border computation differences | Validate against DT's own rendering with screenshot comparison |
| Object masks (SAM) | These use bitmap masks, not point-based — need separate handling |

## 7. Open Questions

1. **Should gui_points be cached?** If the history doesn't change, they don't need regeneration. Could cache per `history_hash + form_id`.

2. **What about mask editing?** During interactive editing (dragging a control point), we need real-time coordinate transforms. The `distort_points` endpoint could serve this, or we could send the distortion grid (Option C) for client-side transforms during editing, then snap to server-computed points on release.

3. **Gradient masks** use a different rendering approach (fill with gradient, not polyline). Need to verify how gui_points work for gradients in DT.

4. **Group masks** — the boolean operations happen at the bitmap level. gui_points are per-sub-form, which should work fine for rendering individual form outlines.

5. **Object masks** store a bitmap mask, not geometric coordinates. These need a completely different approach — likely sending the mask bitmap as an image overlay.

## 8. Empirical Findings: rawprepare Is the Universal Distortion

### Discovery

Testing with an image that has **no geometric modules** (no lens correction, no crop, no rotation — only sigmoid, exposure, color calibration, and similar tonal modules), a circle mask still showed a small but consistent vertical shift between DT and Nova.

### Root Cause: rawprepare Crop

Every RAW image goes through `rawprepare`, which crops the sensor's black border pixels. This module has a `distort_transform` that shifts all coordinates:

```c
// src/iop/rawprepare.c:203
gboolean distort_transform(dt_iop_module_t *self,
                           dt_dev_pixelpipe_iop_t *piece,
                           float *const restrict points,
                           size_t points_count)
{
    dt_iop_rawprepare_data_t *d = piece->data;
    if(d->left == 0 && d->top == 0) return TRUE;

    const float scale = piece->buf_in.scale / piece->iscale;
    const float x = (float)d->left * scale;
    const float y = (float)d->top * scale;

    for(size_t i = 0; i < points_count * 2; i += 2)
    {
        points[i] -= x;
        points[i + 1] -= y;
    }
    return TRUE;
}
```

### Measured Dimensions

For the test image (Nikon Z8 RAW):
- **Input (sensor)**: 6880×4544 — ratio **1.5141**
- **Output (pipe)**: 1029×686 — ratio **1.5000** (exactly 3:2)

The ratio change from 1.5141 to 1.5000 (~0.94%) is caused by `rawprepare` cropping asymmetric sensor borders. The `d->left` and `d->top` values are small (a few pixels) but at normalized-coordinate scale they produce a visible shift.

### Key Implication

**There is no "simple" subset of images where distortion can be skipped.** Every RAW image has `rawprepare` distortion. This means:

1. The distortion pipeline is **always** needed for correct mask placement
2. Even the most basic history stack (auto-applied modules only) requires coordinate transforms
3. Nova's current approach of rendering raw normalized coordinates will **always** be wrong for RAW files
4. Only JPEG/TIFF inputs (which skip `rawprepare`) would render correctly without transforms

### Reinforced Recommendation

This finding strengthens the case for server-side distortion via `dt_dev_distort_transform_plus()`. Since we need it for every image anyway, there's no incremental approach that works — the transform must be implemented as a prerequisite for correct masking.

The simplest first step: add a `develop.distort_points` endpoint that accepts normalized coordinates, scales them to pipe input space, calls `dt_dev_distort_transform_plus(&session->dev, session->dev.full.pipe, ...)`, scales back to normalized output space, and returns the transformed coordinates. One endpoint, uniform handling for rawprepare, lens, crop, and all other geometric modules.

## 9. Revised Implementation Plan

Sections 4-5 above were written before the rawprepare discovery and before verifying that `dt_dev_distort_transform_plus()` already accepts explicit `dev` and `pipe` parameters. This section supersedes the earlier recommendations.

### Key Insight

`dt_dev_distort_transform_plus(dev, pipe, iop_order, direction, points, count)` **already takes explicit dev and pipe**. The "global state problem" only affects:
- `dt_masks_get_image_size()` — reads `darktable.develop->preview_pipe` (we don't need this)
- `dt_dev_distort_transform()` — wrapper that hardcodes `dev->preview_pipe` (we call `_plus` directly)

We can call `dt_dev_distort_transform_plus(&session->dev, session->dev.full.pipe, ...)` from the server with **zero core DT modifications**.

### Approach: Transform Control Points in get_masks

Instead of generating dense gui_points server-side (Option F), transform only the mask control points. Nova already generates geometry client-side (circle arcs, bezier curves, border polylines) — it just needs corrected anchor positions.

### Server Side (server_develop.c)

In `_server_cmd_get_masks()`, after collecting each form's raw points:

1. Scale normalized control points to input pixel space: `x * iwidth, y * iheight`
2. Call `dt_dev_distort_transform_plus(&session->dev, session->dev.full.pipe, 0.0, DT_DEV_TRANSFORM_DIR_ALL, points, count)`
3. Scale result back to normalized output space: `x / pipe_width, y / pipe_height`
4. Include `transformed_points` alongside raw `points` in the JSON response

Per mask type, the points to transform:

| Type | Points to Transform |
|------|-------------------|
| Circle | center + point at center+radius + point at center+radius+border |
| Ellipse | center + axis endpoints (for distorted radii + rotation) |
| Path | each corner + ctrl1 + ctrl2 (6 floats per path point) + border offsets |
| Brush | each stroke point (2 floats each) |
| Gradient | anchor point + rotation reference point |

### Circle Example

```c
dt_dev_pixelpipe_t *pipe = session->dev.full.pipe;
const float iw = pipe->iwidth;
const float ih = pipe->iheight;
const float pw = pipe->backbuf_width;
const float ph = pipe->backbuf_height;
const float dim = MIN(iw, ih);

// Build points array: center, center+radius, center+radius+border
float pts[6];
pts[0] = form->points[0] * iw;                          // center x
pts[1] = form->points[1] * ih;                          // center y
pts[2] = pts[0] + form->points[2] * dim;                // center_x + radius
pts[3] = pts[1];                                         // same y
pts[4] = pts[0] + (form->points[2] + form->points[3]) * dim;  // center_x + radius + border
pts[5] = pts[1];                                         // same y

dt_dev_distort_transform_plus(&session->dev, pipe,
                               0.0, DT_DEV_TRANSFORM_DIR_ALL, pts, 3);

// Back to normalized output space
float t_center_x = pts[0] / pw;
float t_center_y = pts[1] / ph;
float t_radius = sqrtf((pts[2]-pts[0])*(pts[2]-pts[0]) +
                        (pts[3]-pts[1])*(pts[3]-pts[1])) / MIN(pw, ph);
float t_border = sqrtf((pts[4]-pts[0])*(pts[4]-pts[0]) +
                        (pts[5]-pts[1])*(pts[5]-pts[1])) / MIN(pw, ph) - t_radius;
```

### Path Example

```c
// For each path point: transform corner, ctrl1, ctrl2
// border is handled as offset from transformed corner
int n = g_list_length(form->points);
float *pts = malloc(n * 6 * sizeof(float));  // 3 points per path point, 2 floats each
int idx = 0;
for(GList *l = form->points; l; l = l->next) {
    dt_masks_point_path_t *pt = l->data;
    pts[idx++] = pt->corner[0] * iw;
    pts[idx++] = pt->corner[1] * ih;
    pts[idx++] = pt->ctrl1[0] * iw;
    pts[idx++] = pt->ctrl1[1] * ih;
    pts[idx++] = pt->ctrl2[0] * iw;
    pts[idx++] = pt->ctrl2[1] * ih;
}
dt_dev_distort_transform_plus(&session->dev, pipe,
                               0.0, DT_DEV_TRANSFORM_DIR_ALL, pts, n * 3);
// Scale back to normalized output space...
```

### Client Side (MaskOverlay.tsx)

- Add `transformed_points` field to `MaskForm` TypeScript type (per mask type)
- In `drawCircle`, `drawPath`, etc.: use `transformed_points` when available, fall back to raw `points`
- The geometry generation (circle arcs, bezier sampling, border polylines) stays the same — just with corrected anchor positions

### For Interactive Editing (Future)

When dragging a mask control point:
- Client works in raw normalized coordinates during drag
- On mouse-up, sends updated raw coords to server (existing `develop.update_mask`)
- Server re-renders pipe, `get_masks` returns new transformed points
- Small visual jump on release is acceptable initially

Better future option: send distortion grid (Option C) alongside masks for client-side real-time transforms during drag.

### Implementation Steps

1. **Verify `dt_dev_distort_transform_plus` internals** — confirm it doesn't touch globals beyond iterating `pipe->nodes`
2. **Add transform logic in `_server_cmd_get_masks`** — start with circle only
3. **Add `transformed_points` to JSON response** — alongside existing raw points
4. **Update TypeScript types** — add transformed fields to `MaskForm` subtypes
5. **Update `drawCircle` in MaskOverlay.tsx** — use transformed center/radius
6. **Test with rawprepare-only image** — verify vertical shift is fixed
7. **Test with lens correction image** — verify complex distortion works
8. **Extend to path, ellipse, gradient, brush** — same pattern per type
9. **Handle border radius** — transform border offset points too

### Estimated Effort

- Server: ~50-80 lines of C in `server_develop.c`
- Client: ~20 lines of TypeScript changes
- No core DT code modifications needed
- No new files needed

