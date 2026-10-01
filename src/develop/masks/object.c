/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifdef HAVE_AI
#include "common/ai/segmentation.h"
#include "common/ai_models.h"
#endif
#include "common/dtdata.h"
#include "common/colorspaces.h"
#include "common/debug.h"
#include "common/densecrf.h"
#include "common/distance_transform.h"
#include "common/mipmap_cache.h"
#include "common/ras2vect.h"
#include "control/conf.h"
#include "control/control.h"
#include "develop/blend.h"
#include "develop/imageop.h"
#include "develop/masks.h"
#include "develop/openmp_maths.h"
#include "develop/pixelpipe_hb.h"
#include "dtgtk/paint.h"
#include "gui/draw.h"
#include "gui/gtk.h"
#include "imageio/imageio_common.h"
#include "views/view.h"

#include <limits.h>
#include <math.h>
#include <string.h>

/* --- the stored mask: compiled in every build, AI or not --- */

/* a committed object's shape is pixels over the whole frame, not points, so
   _raster_get_mask_roi maps the roi back into it, on gradient's grid */

/* decoded entries, named by the sha1 of their bytes, so a decoded one never
   goes stale. the cap is in bytes and drops the least recently used entry
   first, where the masks an edit replaced end up. it holds about 40 masks
   at the stored size; an image with more decodes them all on every rerun */
#define RASTER_CACHE_BYTES ((size_t)64 << 20)

// one decoded entry. mask and its size never change, so a reference reads
// them unlocked; refs, evicted and located take the lock, and the anchor,
// written once before located is set, is read unlocked after
typedef struct _raster_cache_t
{
  char entry[DT_DTDATA_ENTRY_LEN];
  dt_imgid_t imgid;
  uint8_t *mask;     // as stored, 8 bits. NULL: unreadable, don't try again
  int width, height;
  int refs;          // callers still reading mask
  gboolean evicted;  // out of the list, freed by the last release
  // the icon's place, the point deepest inside the mask as a fraction of the
  // frame, NAN without an interior. found by the gui on first draw
  gboolean located;
  float anchor[2];
} _raster_cache_t;

// most recently used first
static GList *_cache = NULL;
static size_t _cache_bytes = 0;
// bumped by _raster_forget_failure, so a failed read racing it is not cached
static guint _cache_forgets = 0;
G_LOCK_DEFINE_STATIC(raster_cache);

// the stored mask's reference, the form's first point. NULL for a new object
static const dt_dtdata_ref_t *_raster_ref(const dt_masks_form_t *form)
{
  return form->points ? &((const dt_masks_point_object_t *)form->points->data)->ref : NULL;
}

static void _raster_free(_raster_cache_t *c)
{
  dt_free_align(c->mask);
  g_free(c);
}

// a failed read is charged too, or images whose sidecars are missing would
// add entries without end
static size_t _raster_bytes(const _raster_cache_t *c)
{
  return c->mask ? (size_t)c->width * c->height : (size_t)64 << 10;
}

// unlist an entry, freed now or by its last release. called with the lock held
static void _raster_unlist(GList *l)
{
  _raster_cache_t *c = l->data;
  _cache = g_list_delete_link(_cache, l);
  _cache_bytes -= _raster_bytes(c);
  if(c->refs)
    c->evicted = TRUE;
  else
    _raster_free(c);
}

// least recently used first, down to the cap, never the newest entry
static void _raster_evict(void)
{
  GList *l = g_list_last(_cache);
  while(l && l != _cache && _cache_bytes > RASTER_CACHE_BYTES)
  {
    GList *prev = l->prev;
    _raster_unlist(l);
    l = prev;
  }
}

// called with the lock held
static _raster_cache_t *_raster_lookup(const dt_imgid_t imgid, const char *entry)
{
  for(GList *l = _cache; l; l = g_list_next(l))
  {
    _raster_cache_t *c = l->data;
    if(c->imgid == imgid && !strcmp(c->entry, entry))
    {
      _cache = g_list_remove_link(_cache, l);
      _cache = g_list_concat(l, _cache);
      c->refs++;
      return c;
    }
  }
  return NULL;
}

static void _raster_release(_raster_cache_t *c);

/* a referenced entry, to hand back with _raster_release. NULL when the
   entry is missing or damaged, which leaves the object out of the mask:
   the sidecar was lost, or an xmp travelled without it. the failure is
   cached like a success, or every pipe run and every redraw would reopen
   the sidecar and log it again, until a commit writes the entry */
static _raster_cache_t *_raster_get(const dt_imgid_t imgid,
                                    const dt_dtdata_ref_t *ref)
{
  if(!ref || !ref->entry[0] || !dt_is_valid_imgid(imgid)) return NULL;

  G_LOCK(raster_cache);
  _raster_cache_t *c = _raster_lookup(imgid, ref->entry);
  const guint forgets = _cache_forgets;
  G_UNLOCK(raster_cache);
  if(c && !c->mask)
  {
    _raster_release(c);
    return NULL;
  }
  if(c) return c;

  // decode unlocked, so a miss does not stall the other pipes. back to the
  // 8 bits _finalize_raster stored, a quarter of the floats' memory
  int w = 0, h = 0;
  float *pixels = dt_dtdata_read_gray(imgid, ref, &w, &h);
  uint8_t *mask = pixels ? dt_alloc_aligned((size_t)w * h) : NULL;
  if(mask)
    for(size_t k = 0; k < (size_t)w * h; k++)
      mask[k] = (uint8_t)(CLIP(pixels[k]) * 255.0f + 0.5f);
  dt_free_align(pixels);

  G_LOCK(raster_cache);
  // another pipe may have decoded the same entry meanwhile
  c = _raster_lookup(imgid, ref->entry);
  if(c)
    dt_free_align(mask);
  // a failure read before a commit wrote the entry is not cached
  else if(mask || forgets == _cache_forgets)
  {
    c = g_malloc0(sizeof(_raster_cache_t));
    g_strlcpy(c->entry, ref->entry, sizeof(c->entry));
    c->imgid = imgid;
    c->mask = mask;
    c->width = w;
    c->height = h;
    c->refs = 1;
    _cache = g_list_prepend(_cache, c);
    _cache_bytes += _raster_bytes(c);
    _raster_evict();
  }
  G_UNLOCK(raster_cache);
  if(c && !c->mask)
  {
    _raster_release(c);
    return NULL;
  }
  return c;
}

static void _raster_release(_raster_cache_t *c)
{
  if(!c) return;
  G_LOCK(raster_cache);
  const gboolean dead = --c->refs == 0 && c->evicted;
  G_UNLOCK(raster_cache);
  if(dead) _raster_free(c);
}

#ifdef HAVE_AI
/* a regenerated mask can come out byte for byte the same as the one that
   was lost, and so under the same name, which the cache remembers as a
   failed read */
static void _raster_forget_failure(const dt_imgid_t imgid, const char *entry)
{
  G_LOCK(raster_cache);
  _cache_forgets++;
  for(GList *l = _cache; l; l = g_list_next(l))
  {
    const _raster_cache_t *c = l->data;
    if(c->imgid == imgid && !c->mask && !strcmp(c->entry, entry))
    {
      _raster_unlist(l);
      break;
    }
  }
  G_UNLOCK(raster_cache);
}
#endif

// an icon needs no finer place than this many mask pixels, and the distance
// transform runs on the gui thread
#define RASTER_LOCATE_STEP 4

/* the frame edge counts as outside, so the anchor of a mask that runs off
   it, a sky, sits inside the image rather than on its border */
static void _raster_locate(_raster_cache_t *c)
{
  G_LOCK(raster_cache);
  const gboolean located = c->located;
  G_UNLOCK(raster_cache);
  if(located) return;

  // the block maximum, so a part thinner than the step still counts
  const int f = MIN(c->width, c->height) >= 3 * RASTER_LOCATE_STEP ? RASTER_LOCATE_STEP : 1;
  const int w = c->width / f, h = c->height / f;
  const size_t n = (size_t)w * h;
  float *m = dt_alloc_align_float(n);
  float *dist = dt_alloc_align_float(n);
  float anchor[2] = { NAN, NAN };
  if(m && dist)
  {
    DT_OMP_FOR()
    for(int y = 0; y < h; y++)
      for(int x = 0; x < w; x++)
      {
        uint8_t max = 0;
        for(int j = 0; j < f; j++)
          for(int i = 0; i < f; i++)
            max = MAX(max, c->mask[(size_t)(y * f + j) * c->width + x * f + i]);
        m[(size_t)y * w + x] = max * (1.0f / 255.0f);
      }
    for(int x = 0; x < w; x++) m[x] = m[(size_t)(h - 1) * w + x] = 0.0f;
    for(int y = 0; y < h; y++) m[(size_t)y * w] = m[(size_t)y * w + w - 1] = 0.0f;

    if(dt_image_distance_transform(m, dist, w, h, 0.5f, DT_DISTANCE_TRANSFORM_MASK) > 0.0f)
    {
      size_t best = 0;
      for(size_t k = 1; k < n; k++)
        if(dist[k] > dist[best]) best = k;
      anchor[0] = ((best % w) + 0.5f) * f / c->width;
      anchor[1] = ((best / w) + 0.5f) * f / c->height;
    }
  }
  dt_free_align(m);
  dt_free_align(dist);

  G_LOCK(raster_cache);
  if(!c->located)
  {
    c->anchor[0] = anchor[0];
    c->anchor[1] = anchor[1];
    c->located = TRUE;
  }
  G_UNLOCK(raster_cache);
}

/* an object whose mask is missing is left out, never silently: a pipe of
   export type says so for its image, also on stderr for darktable-cli; the
   darkroom says it once per mask and session */
static GHashTable *_missing_seen = NULL;     // "imgid:entry" said in the darkroom
// the object tool's encoder renders, which would report the very object
// being regenerated
static GList *_quiet_pipes = NULL;
G_LOCK_DEFINE_STATIC(raster_missing);

static void _raster_note_missing(const dt_dev_pixelpipe_t *pipe,
                                 const dt_masks_form_t *form,
                                 const dt_dtdata_ref_t *ref)
{
  const dt_imgid_t imgid = pipe->image.id;
  if(pipe->type & DT_DEV_PIXELPIPE_EXPORT)
  {
    G_LOCK(raster_missing);
    const gboolean quiet = g_list_find(_quiet_pipes, pipe) != NULL;
    G_UNLOCK(raster_missing);
    if(quiet) return;
    dt_print(DT_DEBUG_ALWAYS, "[object mask] %s: the mask of '%s' is missing"
             " and was left out", pipe->image.filename, form->name);
    // worded for any export pipe: neural restore uses them too
    dt_control_log(_("%s: the mask of '%s' is missing and was left out"),
                   pipe->image.filename, form->name);
  }
  else if(pipe->type & DT_DEV_PIXELPIPE_SCREEN)
  {
    gchar *key = g_strdup_printf("%d:%s", imgid, ref->entry);
    G_LOCK(raster_missing);
    if(!_missing_seen)
      _missing_seen = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
    const gboolean first = g_hash_table_add(_missing_seen, key);
    G_UNLOCK(raster_missing);
#ifdef HAVE_AI
    if(first)
      dt_control_log(_("the mask of '%s' is missing, click its icon to regenerate it"),
                     form->name);
#else
    (void)first;
#endif
  }
}

/* the hovered icon's mask, tinted over the preview. one icon is hovered at a
   time, so one surface, rebuilt when the entry or the preview changes. gui
   thread only */
static struct
{
  char entry[DT_DTDATA_ENTRY_LEN];
  dt_hash_t hash;
  cairo_surface_t *surface;
} _hover;

void dt_masks_object_cache_cleanup(void)
{
  G_LOCK(raster_missing);
  if(_missing_seen) g_hash_table_destroy(_missing_seen);
  _missing_seen = NULL;
  G_UNLOCK(raster_missing);

  G_LOCK(raster_cache);
  g_list_free_full(_cache, (GDestroyNotify)_raster_free);
  _cache = NULL;
  _cache_bytes = 0;
  G_UNLOCK(raster_cache);

  if(_hover.surface) cairo_surface_destroy(_hover.surface);
  _hover.surface = NULL;
}

/* bilinear read of m at pixel coordinates, which must lie within the
   buffer: x in [0, mw - 1], y in [0, mh - 1] */
static inline float _bilinear(const float *const restrict m,
                              const int mw,
                              const int mh,
                              const float x,
                              const float y)
{
  const int x0 = (int)x, y0 = (int)y;
  const int x1 = MIN(x0 + 1, mw - 1), y1 = MIN(y0 + 1, mh - 1);
  const float fx = x - x0, fy = y - y0;
  return (m[(size_t)y0 * mw + x0] * (1.0f - fx) + m[(size_t)y0 * mw + x1] * fx)
             * (1.0f - fy)
         + (m[(size_t)y1 * mw + x0] * (1.0f - fx) + m[(size_t)y1 * mw + x1] * fx)
             * fy;
}

/* bilinear sample of the stored mask at a point given as a fraction of the
   pipe's input frame. stored pixel k was taken at the center of its span,
   (k + 0.5) / mw of the frame (_finalize_raster), so reading
   back is the inverse of that */
static inline float _sample(const uint8_t *const restrict m,
                            const int mw,
                            const int mh,
                            const float u,
                            const float v)
{
  const float x = CLAMPF(u * mw - 0.5f, 0.0f, mw - 1.0f);
  const float y = CLAMPF(v * mh - 0.5f, 0.0f, mh - 1.0f);
  const int x0 = (int)x, y0 = (int)y;
  const int x1 = MIN(x0 + 1, mw - 1), y1 = MIN(y0 + 1, mh - 1);
  // the four neighbors scaled as the png decoder does, so a value reads
  // back exactly as decoded
  const float n = 1.0f / 255.0f;
  const float q[4] = { m[(size_t)y0 * mw + x0] * n, m[(size_t)y0 * mw + x1] * n,
                       m[(size_t)y1 * mw + x0] * n, m[(size_t)y1 * mw + x1] * n };
  return _bilinear(q, 2, 2, x - x0, y - y0);
}

static int _raster_get_mask_roi(const dt_iop_module_t *const module,
                                const dt_dev_pixelpipe_iop_t *const piece,
                                dt_masks_form_t *const form,
                                const dt_iop_roi_t *roi,
                                float *buffer)
{
  const dt_dtdata_ref_t *ref = _raster_ref(form);
  if(!ref) return 0;

  /* a missing or damaged entry leaves the object out of its group, which
     is not the same as an empty mask: an empty mask still takes part, so
     an inverted object would then apply the module everywhere */
  _raster_cache_t *c = _raster_get(piece->pipe->image.id, ref);
  if(!c || c->width <= 0 || c->height <= 0)
  {
    _raster_release(c);
    _raster_note_missing(piece->pipe, form, ref);
    return 0;
  }
  const uint8_t *const restrict mask = c->mask;
  const int mw = c->width;
  const int mh = c->height;

  const int w = roi->width;
  const int h = roi->height;
  const int px = roi->x;
  const int py = roi->y;
  const float iscale = 1.0f / roi->scale;
  const int grid = CLAMP((10.0f * roi->scale + 2.0f) / 3.0f, 1, 4);
  const int gw = (w + grid - 1) / grid + 1;
  const int gh = (h + grid - 1) / grid + 1;

  float *points = dt_alloc_align_float((size_t)2 * gw * gh);
  if(points == NULL)
  {
    _raster_release(c);
    return 0;
  }

  DT_OMP_FOR(collapse(2))
  for(int j = 0; j < gh; j++)
    for(int i = 0; i < gw; i++)
    {
      const size_t index = (size_t)j * gw + i;
      points[index * 2] = (grid * i + px) * iscale;
      points[index * 2 + 1] = (grid * j + py) * iscale;
    }

  if(!dt_dev_distort_backtransform_plus(module->dev, piece->pipe,
                                        module->iop_order,
                                        DT_DEV_TRANSFORM_DIR_BACK_INCL, points,
                                        (size_t)gw * gh))
  {
    _raster_release(c);
    dt_free_align(points);
    return 0;
  }

  /* the stored mask spans the whole frame whatever its own size, so the
     backtransformed point normalises against the pipe input rather than
     against the entry's dimensions */
  const float wd = piece->pipe->iwidth;
  const float ht = piece->pipe->iheight;

  DT_OMP_FOR()
  for(int k = 0; k < gw * gh; k++)
    points[k * 2] = _sample(mask, mw, mh, points[k * 2] / wd, points[k * 2 + 1] / ht);
  _raster_release(c);

  DT_OMP_FOR()
  for(int j = 0; j < h; j++)
  {
    const int jj = j % grid;
    const int mj = j / grid;
    const int grid_jj = grid - jj;
    for(int i = 0; i < w; i++)
    {
      const int ii = i % grid;
      const int mi = i / grid;
      const int grid_ii = grid - ii;
      const size_t mindex = (size_t)mj * gw + mi;
      buffer[(size_t)j * w + i]
        = (points[mindex * 2] * grid_ii * grid_jj
           + points[(mindex + 1) * 2] * ii * grid_jj
           + points[(mindex + gw) * 2] * grid_ii * jj
           + points[(mindex + gw + 1) * 2] * ii * jj)
          / (grid * grid);
    }
  }

  dt_free_align(points);
  return 1;
}

static void _raster_duplicate_points(dt_develop_t *const dev,
                                     dt_masks_form_t *const base,
                                     dt_masks_form_t *const dest)
{
  for(GList *pts = base->points; pts; pts = g_list_next(pts))
  {
    dt_masks_point_object_t *pt = pts->data;
    dt_masks_point_object_t *npt = malloc(sizeof(dt_masks_point_object_t));
    if(!npt) return;
    memcpy(npt, pt, sizeof(dt_masks_point_object_t));
    dest->points = g_list_append(dest->points, npt);
  }
}


/* on the canvas a committed object is an icon at its anchor, and its mask
   shows, tinted, only while the icon is hovered. an outline would need a
   threshold, and the soft, scattered masks of skies or subjects have no
   edge worth tracing */
#define RASTER_ICON_RADIUS 10.0f // unscaled pixels
// the tint of a mask over the image, premultiplied red at this alpha
#define RASTER_OVERLAY_ALPHA 80.0f
// preview pixels per overlay pixel
#define RASTER_OVERLAY_STEP 2.0f

/* the form drawn at a position of the visible group. gui->points is built
   from that group's direct members in the same order (see
   dt_masks_gui_form_test_create), so the index means the same here */
static dt_masks_form_t *_raster_form_at(const int index)
{
  dt_masks_form_t *grp = darktable.develop->form_visible;
  if(!grp) return NULL;
  if(!(grp->type & DT_MASKS_GROUP)) return grp;
  const dt_masks_point_group_t *pt = g_list_nth_data(grp->points, index);
  return pt ? dt_masks_get_from_id(darktable.develop, pt->formid) : NULL;
}

// whether a preview point is over the object's own pixels, at 0.5, the
// default of the configurable threshold the selection is drawn with
static gboolean _raster_over_mask(const _raster_cache_t *c, const float x, const float y)
{
  float iwidth, iheight;
  dt_masks_get_image_size(NULL, NULL, &iwidth, &iheight);
  float pt[2] = { x, y };
  if(!dt_dev_distort_backtransform(darktable.develop, pt, 1)) return FALSE;
  const float u = pt[0] / iwidth, v = pt[1] / iheight;
  return u >= 0.0f && v >= 0.0f && u <= 1.0f && v <= 1.0f
         && _sample(c->mask, c->width, c->height, u, v) >= 0.5f;
}

// as is DT_PIXEL_APPLY_DPI(7) / zoom_scale (dt_masks_sensitive_dist), so
// this is the icon's radius at the current zoom
static gboolean _raster_over_icon(const dt_masks_form_gui_points_t *gpt,
                                  const float x,
                                  const float y,
                                  const float as)
{
  const float r = as * RASTER_ICON_RADIUS / 7.0f;
  return sqf(gpt->points[0] - x) + sqf(gpt->points[1] - y) < sqf(r);
}

static void _raster_get_distance(const float x,
                                 const float y,
                                 const float as,
                                 dt_masks_form_gui_t *gui,
                                 const int index,
                                 const int num_points,
                                 gboolean *inside,
                                 gboolean *inside_border,
                                 int *near,
                                 gboolean *inside_source,
                                 float *dist)
{
  *inside = FALSE;
  *inside_border = FALSE;
  *inside_source = FALSE;
  *near = -1;
  *dist = FLT_MAX;

  if(!gui) return;
  const dt_masks_form_gui_points_t *gpt = g_list_nth_data(gui->points, index);
  if(!gpt || gpt->points_count < 1) return;

  /* only the icon: the pixels of a sky or a background cover most of the
     frame, and treating them as the shape would select and tint it from
     almost anywhere. over its pixels the icon is only lit (post_expose) */
  *dist = sqf(gpt->points[0] - x) + sqf(gpt->points[1] - y);
  *inside = _raster_over_icon(gpt, x, y, as);
}

// the anchor in image space, carried forward to the preview
static int _raster_get_points_border(dt_develop_t *dev,
                                     dt_masks_form_t *form,
                                     float **points,
                                     int *points_count,
                                     float **border,
                                     int *border_count,
                                     const int source,
                                     const dt_iop_module_t *const module)
{
  if(border) *border = NULL;
  if(border_count) *border_count = 0;
  const dt_dtdata_ref_t *ref = _raster_ref(form);
  if(source || !ref) return 0;

  _raster_cache_t *c = _raster_get(dev->image_storage.id, ref);
  float ax = NAN, ay = NAN;
  if(c)
  {
    _raster_locate(c);
    ax = c->anchor[0];
    ay = c->anchor[1];
    _raster_release(c);
  }
  /* the mask is missing, or has no pixel at 0.5 or above off the frame
     edge: the icon still has to be there, as the way to edit or regenerate
     it, so it goes to the first foreground click instead */
  for(const GList *l = g_list_next(form->points); l && isnan(ax); l = g_list_next(l))
  {
    const dt_masks_point_object_t *pt = l->data;
    if(pt->prompt.label > 0.5f)
    {
      ax = pt->prompt.pos[0];
      ay = pt->prompt.pos[1];
    }
  }
  if(isnan(ax)) return 0;

  float iwidth, iheight;
  dt_masks_get_image_size(NULL, NULL, &iwidth, &iheight);
  float *pts = dt_alloc_align_float(2);
  if(!pts) return 0;
  pts[0] = ax * iwidth;
  pts[1] = ay * iheight;
  if(!dt_dev_distort_transform(dev, pts, 1))
  {
    dt_free_align(pts);
    return 0;
  }
  *points = pts;
  *points_count = 1;
  return 1;
}

static int _raster_mouse_moved(dt_iop_module_t *module,
                               float pzx,
                               float pzy,
                               const double pressure,
                               const int which,
                               const float zoom_scale,
                               dt_masks_form_t *form,
                               const dt_imgid_t parentid,
                               dt_masks_form_gui_t *gui,
                               const int index)
{
  if(gui->creation) return 0;

  float wd, ht;
  dt_masks_get_image_size(&wd, &ht, NULL, NULL);
  gboolean in, inb, ins;
  int near;
  float dist;
  _raster_get_distance(pzx * wd, pzy * ht, dt_masks_sensitive_dist(zoom_scale),
                       gui, index, 0, &in, &inb, &near, &ins, &dist);
  gui->form_selected = in;
  gui->border_selected = FALSE;
  gui->source_selected = FALSE;
  // the icon is the object's one handle, what a click on it opens the edit from
  gui->point_selected = in ? 0 : -1;
  gui->point_border_selected = -1;

  dt_control_queue_redraw_center();
  return in;
}

/* the stored mask resampled into out over the whole preview frame on a
   w x h grid, the same way _raster_get_mask_roi reads it for the pipe. used
   for the hover overlay and, at the encoded size, to resume editing */
static gboolean _raster_to_preview(const _raster_cache_t *c,
                                   const int w,
                                   const int h,
                                   float *const out)
{
  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);
  const size_t n = (size_t)w * h;
  float *pts = dt_alloc_align_float(2 * n);
  if(!pts) return FALSE;

  DT_OMP_FOR(collapse(2))
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
    {
      const size_t k = (size_t)y * w + x;
      pts[k * 2] = (x + 0.5f) * wd / w;
      pts[k * 2 + 1] = (y + 0.5f) * ht / h;
    }
  if(!dt_dev_distort_backtransform(darktable.develop, pts, n))
  {
    dt_free_align(pts);
    return FALSE;
  }

  const uint8_t *const mask = c->mask;
  const int mw = c->width, mh = c->height;
  DT_OMP_FOR()
  for(size_t k = 0; k < n; k++)
  {
    const float u = pts[k * 2] / iwidth;
    const float v = pts[k * 2 + 1] / iheight;
    out[k] = (u < 0.0f || v < 0.0f || u > 1.0f || v > 1.0f)
      ? 0.0f
      : _sample(mask, mw, mh, u, v);
  }
  dt_free_align(pts);
  return TRUE;
}

/* a mask as a red tint, premultiplied: in proportion to the mask, or at
   full strength above threshold. NULL on error */
static cairo_surface_t *_tint(const float *const mask,
                              const int w,
                              const int h,
                              const gboolean proportional,
                              const float threshold)
{
  cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
  if(cairo_surface_status(surface) != CAIRO_STATUS_SUCCESS)
  {
    cairo_surface_destroy(surface);
    return NULL;
  }
  cairo_surface_flush(surface);
  unsigned char *data = cairo_image_surface_get_data(surface);
  const int stride = cairo_image_surface_get_stride(surface);

  DT_OMP_FOR()
  for(int y = 0; y < h; y++)
  {
    unsigned char *row = data + (size_t)y * stride;
    for(int x = 0; x < w; x++)
    {
      const float v = mask[(size_t)y * w + x];
      const float a = proportional
        ? CLAMPF(v, 0.0f, 1.0f) * RASTER_OVERLAY_ALPHA
        : (v > threshold ? RASTER_OVERLAY_ALPHA : 0.0f);
      row[x * 4 + 0] = 0;                  // B
      row[x * 4 + 1] = 0;                  // G
      row[x * 4 + 2] = (unsigned char)a;   // R, premultiplied
      row[x * 4 + 3] = (unsigned char)a;   // A
    }
  }
  cairo_surface_mark_dirty(surface);
  return surface;
}

static cairo_surface_t *_raster_overlay(const _raster_cache_t *c)
{
  const dt_hash_t hash = darktable.develop->preview_pipe->backbuf_hash;
  if(_hover.surface && _hover.hash == hash && !strcmp(_hover.entry, c->entry))
    return _hover.surface;

  float wd, ht;
  dt_masks_get_image_size(&wd, &ht, NULL, NULL);
  const int ow = MAX(1, (int)ceilf(wd / RASTER_OVERLAY_STEP));
  const int oh = MAX(1, (int)ceilf(ht / RASTER_OVERLAY_STEP));
  float *val = dt_alloc_align_float((size_t)ow * oh);
  cairo_surface_t *surface =
    val && _raster_to_preview(c, ow, oh, val) ? _tint(val, ow, oh, TRUE, 0.0f) : NULL;
  dt_free_align(val);
  if(!surface) return NULL;

  if(_hover.surface) cairo_surface_destroy(_hover.surface);
  _hover.surface = surface;
  _hover.hash = hash;
  g_strlcpy(_hover.entry, c->entry, sizeof(_hover.entry));
  return surface;
}

static void _raster_post_expose(cairo_t *cr,
                                const float zoom_scale,
                                dt_masks_form_gui_t *gui,
                                const int index,
                                const int num_points)
{
  if(!gui) return;
  const dt_masks_form_gui_points_t *gpt = g_list_nth_data(gui->points, index);
  if(!gpt || gpt->points_count < 1) return;

  const dt_masks_form_t *form = _raster_form_at(index);
  // hovered is the icon itself, the only part that selects (_raster_get_distance)
  const gboolean hovered = gui->group_selected == index && gui->form_selected;
  const dt_dtdata_ref_t *ref = form ? _raster_ref(form) : NULL;
  _raster_cache_t *c = _raster_get(darktable.develop->image_storage.id, ref);
  const gboolean missing = ref && !c;
  // over the object's pixels the icon lights up, to show which one they are
  const gboolean over_mask =
    !hovered && c && darktable.develop->darkroom_mouse_in_center_area
    && _raster_over_mask(c, gui->posx, gui->posy);
  const gboolean active =
    hovered || over_mask
    || (form && form->formid == darktable.develop->mask_form_selected_id);

  // the tint follows the icon only; the pixels or a selection just light it
  if(hovered && c)
  {
    cairo_surface_t *overlay = _raster_overlay(c);
    if(overlay)
    {
      float wd, ht;
      dt_masks_get_image_size(&wd, &ht, NULL, NULL);
      cairo_save(cr);
      cairo_scale(cr, wd / cairo_image_surface_get_width(overlay),
                  ht / cairo_image_surface_get_height(overlay));
      cairo_set_source_surface(cr, overlay, 0, 0);
      cairo_paint(cr);
      cairo_restore(cr);
    }
  }
  _raster_release(c);

  const float x = gpt->points[0], y = gpt->points[1];
  const float r = DT_PIXEL_APPLY_DPI(RASTER_ICON_RADIUS) / zoom_scale;
  cairo_save(cr);
  cairo_set_dash(cr, NULL, 0, 0);
  cairo_arc(cr, x, y, r, 0.0, 2.0 * M_PI);
  dt_draw_set_color_overlay(cr, FALSE, active ? 0.8 : 0.5);
  cairo_fill(cr);
  // the paint function takes integer coordinates, which a zoomed-in canvas
  // would round away: draw it at a fixed size and scale that instead
  const float s = 1.4f * r;
  cairo_translate(cr, x - 0.5f * s, y - 0.5f * s);
  cairo_scale(cr, s / 100.0f, s / 100.0f);
  dt_draw_set_color_overlay(cr, TRUE, active ? 1.0 : 0.7);
  dtgtk_cairo_paint_masks_object(cr, 0, 0, 100, 100, 0, NULL);
  if(missing)
  {
    // struck through: the object is left out until it is regenerated
    cairo_set_line_width(cr, 12.0);
    cairo_move_to(cr, 15.0, 85.0);
    cairo_line_to(cr, 85.0, 15.0);
    cairo_stroke(cr);
  }
  cairo_restore(cr);
}


/* --- creation: the AI tool, only where AI is compiled in --- */

#ifdef HAVE_AI

/* a committed object has points and a new one none until the commit. not a
   lookup in dev->forms: a history change mid-edit replaces it with copies */
static gboolean _is_edit(const dt_masks_form_t *form)
{
  return form && form->points;
}

#define CONF_OBJECT_THRESHOLD_KEY "plugins/darkroom/masks/object/threshold"
#define CONF_OBJECT_REFINE_PASSES_KEY "plugins/darkroom/masks/object/refine_passes"
#define CONF_OBJECT_CLEANUP_KEY "plugins/darkroom/masks/object/cleanup"
#define CONF_OBJECT_SMOOTHING_KEY "plugins/darkroom/masks/object/smoothing"
#define CONF_OBJECT_FEATHER_KEY "plugins/darkroom/masks/object/feather"
#define CONF_OBJECT_PERSIST_KEY "plugins/darkroom/masks/object/persist_model"
#define CONF_OBJECT_VECTORIZE_KEY "plugins/darkroom/masks/object/vectorize"
#define CONF_OBJECT_REFINE_BOUNDARY_KEY "plugins/darkroom/masks/object/refine_boundary"
#define CONF_OBJECT_REFINE_BOUNDARY_ITER_KEY "plugins/darkroom/masks/object/refine_boundary_iterations"
#define CONF_OBJECT_REFINE_BOUNDARY_SIGMA_COLOR_KEY "plugins/darkroom/masks/object/refine_boundary_sigma_color"
#define CONF_OBJECT_REFINE_BOUNDARY_W_BILATERAL_KEY "plugins/darkroom/masks/object/refine_boundary_weight_bilateral"

// default render target (longest side in pixels).
// the SAM encoder internally downscales to 1024 so encoding quality
// is the same, but higher render resolution gives the guided filter
// and vectorizer more detail for edge refinement.
// configurable via plugins/darkroom/masks/object/render_size
#define SEG_RENDER_DEFAULT 1536
#define CONF_OBJECT_RENDER_SIZE_KEY "plugins/darkroom/masks/object/render_size"

// longest side of a stored mask, whatever the render target
#define RASTER_STORE_SIZE 1536

// --- per-session segmentation state (stored in gui->scratchpad) ---

typedef enum _encode_state_t
{
  ENCODE_ERROR = -1,
  ENCODE_IDLE = 0,
  ENCODE_MSG_SHOWN = 1, // busy message queued, waiting for next expose
  ENCODE_READY = 2,     // encoding complete, results available
  ENCODE_RUNNING = 3,   // background thread in progress
} _encode_state_t;

// minimum drag distance (preview pipe pixels) to distinguish click from drag
#define DRAG_THRESHOLD 5.0f

// how long input has to settle before the outline is traced again
#define OUTLINE_TRACE_DELAY_MS 150

typedef struct _object_data_t
{
  dt_ai_environment_t *env; // AI environment for model registry
  dt_seg_context_t *seg;    // SAM context (encoder+decoder)
  float *mask;              // the selection at the encoded size, g_free'd
  int mask_w, mask_h;       // mask dimensions
  gboolean model_loaded;    // whether the model was loaded
  int encode_state;         // uses _encode_state_t values (atomic access)
  dt_imgid_t encoded_imgid; // image ID that was encoded
  dt_hash_t encoded_distort_hash; // distort hash at encode time (detects crop/rotate)
  int encode_w, encode_h;   // encoding resolution (for coordinate mapping)
  guint modifier_poll_id;   // timer to detect shift key changes
  GThread *encode_thread;   // background encoding thread
  gboolean dragging;        // TRUE between press and release during click drag
  float drag_start_x;       // press position (preview pipe pixel space)
  float drag_start_y;
  gboolean has_selection;   // TRUE after first click, enables refinement mode
  // the prompts, dt_masks_point_object_t input-image normalized, in click order
  GList *prompts;
  // outline of the paths a commit would trace, kept while applying as paths
  GList *outline_forms;             // GList of dt_masks_form_t* (mask-space pixel coords)
  GList *outline_signs;             // parallel GList of sign values ('+' or '-')
  guint outline_trace_id;           // pending trace of the outline, 0 if none
  // editing a committed object: its stored mask and prompts were restored
  gboolean restored;
  guint resume_id;          // pending restore of the edit, 0 if none
  // a decode is running, and pumping the main loop from inside it
  gboolean decoding;
  // the selection changed since then, so a commit has something to store
  gboolean changed;
} _object_data_t;

static _object_data_t *_get_data(dt_masks_form_gui_t *gui)
{
  return (gui && gui->scratchpad) ? (_object_data_t *)gui->scratchpad : NULL;
}

// compute a hash of all distortion module parameters
// from a develop history — changes on crop/rotate/perspective/lens
// but NOT on exposure/color/masks
static dt_hash_t _compute_distort_hash(dt_develop_t *dev)
{
  dt_hash_t hash = DT_INITHASH;
  for(GList *l = dev->history; l; l = g_list_next(l))
  {
    const dt_dev_history_item_t *item = l->data;
    if(item->module
       && item->module->enabled
       && (item->module->operation_tags() & IOP_TAG_DISTORT))
    {
      hash = dt_hash(hash, item->params, item->module->params_size);
    }
  }
  return hash;
}

static void _on_view_changed(gpointer instance,
                             dt_view_t *old_view,
                             dt_view_t *new_view,
                             gpointer user_data)
{
  (void)instance;
  (void)new_view;
  (void)user_data;

  // free persistent model when leaving darkroom
  if(old_view && old_view->view(old_view) == DT_VIEW_DARKROOM)
  {
    dt_ai_seg_t *seg = &darktable.ai_seg;
    if(seg->ctx)
    {
      dt_print(DT_DEBUG_AI,
               "[object mask] freeing persistent model");
      dt_seg_free(seg->ctx);
      seg->ctx = NULL;
    }
    if(seg->env)
    {
      dt_ai_env_destroy(seg->env);
      seg->env = NULL;
    }
    seg->model_loaded = FALSE;

    DT_CONTROL_SIGNAL_DISCONNECT(_on_view_changed, NULL);
    seg->signal_connected = FALSE;
  }
}

// the outline's forms are never registered in dev->forms
static void _free_outline(_object_data_t *d)
{
  if(!d) return;
  for(GList *l = d->outline_forms; l; l = g_list_next(l))
    dt_masks_free_form(l->data);
  g_list_free(d->outline_forms);
  d->outline_forms = NULL;
  g_list_free(d->outline_signs);
  d->outline_signs = NULL;
}

static void _clear_prompts(_object_data_t *d)
{
  g_list_free_full(d->prompts, free);
  d->prompts = NULL;
}

// a deep copy of a list of prompts, NULL when out of memory
static GList *_copy_prompts(const GList *prompts)
{
  GList *copy = NULL;
  for(const GList *l = prompts; l; l = g_list_next(l))
  {
    dt_masks_point_object_t *pt = malloc(sizeof(dt_masks_point_object_t));
    if(!pt)
    {
      g_list_free_full(copy, free);
      return NULL;
    }
    memcpy(pt, l->data, sizeof(dt_masks_point_object_t));
    copy = g_list_prepend(copy, pt);
  }
  return g_list_reverse(copy);
}

/* the "apply as paths" switch. with sidecar files disabled there is nowhere
   to keep pixels, so it is on whatever it says */
static gboolean _as_paths(void)
{
  return !dt_dtdata_enabled() || dt_conf_get_bool(CONF_OBJECT_VECTORIZE_KEY);
}

// whether a right-click traces paths. an edit stays pixels: it updates the
// object in place
static gboolean _commits_paths(const dt_masks_form_t *form)
{
  return !_is_edit(form) && _as_paths();
}

/* the mask traced into path forms in mask pixels, on the commit's settings
   so the outline matches the paths. FALSE only when out of memory */
static gboolean _trace(const _object_data_t *d, GList **forms, GList **signs)
{
  *forms = NULL;
  *signs = NULL;

  // potrace traces dark ink on white, the mask is high inside the object:
  // invert both the mask and the threshold
  const size_t n = (size_t)d->mask_w * d->mask_h;
  float *inv_mask = g_try_malloc(n * sizeof(float));
  if(!inv_mask)
    return FALSE;

  for(size_t i = 0; i < n; i++)
    inv_mask[i] = 1.0f - d->mask[i];

  const int cleanup = dt_conf_get_int(CONF_OBJECT_CLEANUP_KEY);
  const float smoothing = dt_conf_get_float(CONF_OBJECT_SMOOTHING_KEY);
  const float thresh = 1.0f - CLAMP(dt_conf_get_float(CONF_OBJECT_THRESHOLD_KEY),
                                    0.3f, 0.9f);
  *forms = ras2forms(inv_mask, d->mask_w, d->mask_h, NULL,
                     thresh, cleanup, (double)smoothing, signs);
  g_free(inv_mask);

  const float feather = dt_conf_get_float(CONF_OBJECT_FEATHER_KEY);
  for(GList *fl = *forms; fl; fl = g_list_next(fl))
  {
    dt_masks_form_t *f = fl->data;
    for(GList *pt = f->points; pt; pt = g_list_next(pt))
    {
      dt_masks_point_path_t *p = pt->data;
      p->border[0] = p->border[1] = feather;
    }
  }
  return TRUE;
}

/* an edit stays pixels, so it has no outline to show. what the right-click
   will do is the test, not whether the edit restored: a restore that failed
   is still an edit */
static gboolean _wants_outline(const _object_data_t *d)
{
  return d->mask && d->mask_w > 0 && d->mask_h > 0
    && _commits_paths(darktable.develop->form_visible);
}

static gboolean _outline_trace_cb(gpointer data)
{
  _object_data_t *d = data;
  d->outline_trace_id = 0;
  _free_outline(d);
  if(_wants_outline(d))
    _trace(d, &d->outline_forms, &d->outline_signs);
  dt_control_queue_redraw_center();
  return G_SOURCE_REMOVE;
}

/* a trace is a potrace pass, too slow for every scroll step, so it waits for
   input to settle and the last outline stays up meanwhile. with nothing to
   trace, the outline goes at once */
static void _schedule_outline(_object_data_t *d)
{
  if(d->outline_trace_id)
  {
    g_source_remove(d->outline_trace_id);
    d->outline_trace_id = 0;
  }
  if(!_wants_outline(d))
  {
    _free_outline(d);
    dt_control_queue_redraw_center();
    return;
  }
  d->outline_trace_id = g_timeout_add(OUTLINE_TRACE_DELAY_MS, _outline_trace_cb, d);
}

// free all resources in _object_data_t (must be called after thread has joined),
// preserves seg+env in persistent statics so the model stays loaded
static void _destroy_data(_object_data_t *d)
{
  if(!d)
    return;
  if(d->modifier_poll_id)
    g_source_remove(d->modifier_poll_id);
  if(d->outline_trace_id)
    g_source_remove(d->outline_trace_id);
  if(d->resume_id)
    g_source_remove(d->resume_id);
  if(d->encode_thread)
    g_thread_join(d->encode_thread);

  // save model to persistent storage - keeps it loaded across
  // mask sessions, disk cache handles embedding persistence.
  // only persist if nobody already claimed the slot (guards
  // against deferred cleanup racing with a new session)
  dt_ai_seg_t *ps = &darktable.ai_seg;
  const gboolean persist = dt_conf_get_bool(CONF_OBJECT_PERSIST_KEY);
  if(persist && !ps->ctx && d->seg)
  {
    dt_seg_reset_encoding(d->seg);
    ps->env = d->env;
    ps->ctx = d->seg;
    ps->model_loaded = d->model_loaded;
    d->env = NULL;
    d->seg = NULL;
  }
  else
  {
    if(d->seg) dt_seg_free(d->seg);
    if(d->env) dt_ai_env_destroy(d->env);
    d->seg = NULL;
    d->env = NULL;
  }

  g_free(d->mask);
  _free_outline(d);
  _clear_prompts(d);
  g_free(d);
}

// idle callback for deferred cleanup when background thread was still running
static gboolean _deferred_cleanup(gpointer data)
{
  _object_data_t *d = data;
  const int state = g_atomic_int_get(&d->encode_state);
  if(state == ENCODE_RUNNING || d->decoding)
    return G_SOURCE_CONTINUE;
  _destroy_data(d);
  return G_SOURCE_REMOVE;
}

static void _free_data(dt_masks_form_gui_t *gui)
{
  _object_data_t *d = _get_data(gui);
  if(!d)
    return;
  gui->scratchpad = NULL;

  const int state = g_atomic_int_get(&d->encode_state);
  if(state == ENCODE_RUNNING || d->decoding)
  {
    /* the encode thread still holds d, or a decode is running on this very
       stack: dt_gui_cursor_set_busy pumps the main loop, so anything that
       leaves the tool can be dispatched from inside the decode, which writes
       d when it returns. the scratchpad is cleared above either way, so no
       handler reaches d through it meanwhile */
    g_timeout_add(200, _deferred_cleanup, d);
    return;
  }
  _destroy_data(d);
}

// data passed to the background encoding thread
typedef struct _encode_thread_data_t
{
  _object_data_t *d;
  dt_imgid_t imgid;        // image to encode (thread renders via export pipe)
  int32_t history_end;     // darkroom history_end (may be ahead of database)
  dt_hash_t distort_hash;  // hash from live darkroom state (for disk cache key)
} _encode_thread_data_t;

// background thread: loads model, renders image via export pipe, and encodes,
// does ZERO GLib/GTK calls - only computation + atomic state set,
// the poll timer on the main thread detects completion
static gpointer _encode_thread_func(gpointer data)
{
  _encode_thread_data_t *td = data;
  _object_data_t *d = td->d;
  const dt_imgid_t imgid = td->imgid;
  const int32_t td_history_end = td->history_end;
  const dt_hash_t distort_hash = td->distort_hash;
  g_free(td);

  // load model if needed
  if(!d->model_loaded)
  {
    if(!d->env)
      d->env = dt_ai_env_init(NULL);

    char *model_id = dt_ai_models_get_active_for_task("mask");
    d->seg = dt_seg_load(d->env, model_id);
    g_free(model_id);

    if(!d->seg)
    {
      g_atomic_int_set(&d->encode_state, ENCODE_ERROR);
      return NULL;
    }
    d->model_loaded = TRUE;
  }

  // render image at high resolution via temporary export pipeline
  dt_develop_t dev;
  dt_dev_init(&dev, FALSE);
  dt_dev_load_image(&dev, imgid);

  // the database's history_end may lag behind the darkroom's
  // in-memory state (crop/rotate not flushed yet), override
  // so synch_all applies all current edits
  if(td_history_end > 0 && td_history_end > dev.history_end)
    dev.history_end = td_history_end;

  dt_mipmap_buffer_t buf;
  dt_mipmap_cache_get(&buf, imgid, DT_MIPMAP_FULL, DT_MIPMAP_BLOCKING, 'r');

  if(!buf.buf || !buf.width || !buf.height)
  {
    dt_print(DT_DEBUG_AI,
             "[object mask] failed to get image buffer for encoding");
    dt_mipmap_cache_release(&buf);
    dt_dev_cleanup(&dev);
    g_atomic_int_set(&d->encode_state, ENCODE_ERROR);
    return NULL;
  }

  const int wd = dev.image_storage.width;
  const int ht = dev.image_storage.height;

  dt_dev_pixelpipe_t pipe;
  if(!dt_dev_pixelpipe_init_export(&pipe, wd, ht, IMAGEIO_RGB | IMAGEIO_INT8,
                                   FALSE))
  {
    dt_print(DT_DEBUG_AI,
             "[object mask] failed to init export pipe for encoding");
    dt_mipmap_cache_release(&buf);
    dt_dev_cleanup(&dev);
    g_atomic_int_set(&d->encode_state, ENCODE_ERROR);
    return NULL;
  }

  dt_dev_pixelpipe_set_icc(&pipe, DT_COLORSPACE_SRGB, NULL,
                           DT_INTENT_PERCEPTUAL);
  dt_dev_pixelpipe_set_input(&pipe, &dev, (float *)buf.buf,
                             buf.width, buf.height, buf.iscale);
  dt_dev_pixelpipe_create_nodes(&pipe, &dev);
  dt_dev_pixelpipe_synch_all(&pipe, &dev);

  dt_dev_pixelpipe_get_dimensions(&pipe, &dev, pipe.iwidth, pipe.iheight,
                                  &pipe.processed_width,
                                  &pipe.processed_height);

  const int render_target = dt_conf_key_exists(CONF_OBJECT_RENDER_SIZE_KEY)
    ? MAX(dt_conf_get_int(CONF_OBJECT_RENDER_SIZE_KEY), 1024)
    : SEG_RENDER_DEFAULT;
  const double scale = fmin((double)render_target / (double)pipe.processed_width,
                            (double)render_target / (double)pipe.processed_height);
  const double final_scale = fmin(scale, 1.0); // don't upscale
  const int out_w = (int)(final_scale * pipe.processed_width);
  const int out_h = (int)(final_scale * pipe.processed_height);

  // use distort hash from darkroom's live state (passed by caller)
  // instead of computing from the thread's dev, which may have
  // stale history (not yet flushed to database)
  if(dt_seg_disk_cache_load(d->seg, imgid, distort_hash))
  {
    dt_dev_pixelpipe_cleanup(&pipe);
    dt_mipmap_cache_release(&buf);
    dt_dev_cleanup(&dev);
    dt_seg_get_encoded_rgb(d->seg, &d->encode_w, &d->encode_h);
    g_atomic_int_set(&d->encode_state, ENCODE_READY);
    dt_seg_warmup_decoder(d->seg);
    return NULL;
  }

  dt_print(DT_DEBUG_AI,
           "[object mask] rendering %dx%d (scale=%.3f) for encoding...",
           out_w, out_h, final_scale);

  G_LOCK(raster_missing);
  _quiet_pipes = g_list_prepend(_quiet_pipes, &pipe);
  G_UNLOCK(raster_missing);
  dt_dev_pixelpipe_process_no_gamma(&pipe, &dev, 0, 0, out_w, out_h, final_scale);
  G_LOCK(raster_missing);
  _quiet_pipes = g_list_remove(_quiet_pipes, &pipe);
  G_UNLOCK(raster_missing);

  // backbuf is float RGBA after process_no_gamma, convert to uint8 RGB for SAM
  uint8_t *rgb = NULL;
  if(pipe.backbuf)
  {
    const float *outbuf = (const float *)pipe.backbuf;
    rgb = g_try_malloc((size_t)out_w * out_h * 3);
    if(rgb)
    {
      for(size_t i = 0; i < (size_t)out_w * out_h; i++)
      {
        rgb[i * 3 + 0] = (uint8_t)CLAMP(outbuf[i * 4 + 0] * 255.0f + 0.5f, 0, 255);
        rgb[i * 3 + 1] = (uint8_t)CLAMP(outbuf[i * 4 + 1] * 255.0f + 0.5f, 0, 255);
        rgb[i * 3 + 2] = (uint8_t)CLAMP(outbuf[i * 4 + 2] * 255.0f + 0.5f, 0, 255);
      }
    }
  }

  dt_dev_pixelpipe_cleanup(&pipe);
  dt_mipmap_cache_release(&buf);
  dt_dev_cleanup(&dev);

  if(!rgb)
  {
    dt_print(DT_DEBUG_AI, "[object mask] failed to render image for encoding");
    g_atomic_int_set(&d->encode_state, ENCODE_ERROR);
    return NULL;
  }

  // store encoding dimensions for coordinate mapping
  d->encode_w = out_w;
  d->encode_h = out_h;

  // encode the image
  gboolean ok = dt_seg_encode_image(d->seg, rgb, out_w, out_h);

  // if accelerated encoding failed, fall back to CPU
  if(!ok)
  {
    dt_print(DT_DEBUG_AI,
             "[object mask] encoding failed, retrying with CPU provider");
    dt_seg_free(d->seg);
    dt_ai_env_set_provider(d->env, DT_AI_PROVIDER_CPU);
    char *model_id = dt_ai_models_get_active_for_task("mask");
    d->seg = dt_seg_load(d->env, model_id);
    g_free(model_id);

    if(d->seg)
      ok = dt_seg_encode_image(d->seg, rgb, out_w, out_h);
    else
      d->model_loaded = FALSE;
  }

  // dt_seg_encode_image keeps its own copy of rgb for edge refinement
  if(ok)
    dt_seg_disk_cache_save(d->seg, imgid, distort_hash,
                           rgb, out_w, out_h);
  g_free(rgb);

  // signal ready so the user can start placing points; warmup continues
  // on this thread; _run_decoder joins the thread on the first click to
  // avoid a race with warmup on the shared segmentation context
  g_atomic_int_set(&d->encode_state, ok ? ENCODE_READY : ENCODE_ERROR);

  // warm up decoder with real encoder embeddings so the first user click
  // doesn't pay ORT's lazy-init + arena-sizing cost on the main thread
  if(ok)
    dt_seg_warmup_decoder(d->seg);

  return NULL;
}

// keep only the connected component containing the seed pixel
// (seed_x, seed_y), if the seed is outside any foreground region,
// keep the largest component instead, operates in-place: non-selected
// foreground pixels are zeroed
static void _keep_seed_component(float *mask,
                                 const int w,
                                 const int h,
                                 const float threshold,
                                 const int seed_x,
                                 const int seed_y)
{
  const int npix = w * h;
  int16_t *labels = g_try_malloc0((size_t)npix * sizeof(int16_t));
  if(!labels)
    return;
  int *stack = g_try_malloc((size_t)npix * sizeof(int));
  if(!stack)
  {
    g_free(labels);
    return;
  }

  int16_t n_labels = 0;
  int16_t best_label = 0;
  int best_area = 0;
  int16_t seed_label = 0;

  for(int i = 0; i < npix; i++)
  {
    if(mask[i] <= threshold || labels[i] != 0)
      continue;
    if(n_labels >= INT16_MAX)
      break;

    n_labels++;
    const int16_t label = n_labels;
    int area = 0;
    int sp = 0;
    stack[sp++] = i;
    labels[i] = label;

    while(sp > 0)
    {
      const int p = stack[--sp];
      area++;
      const int px = p % w;
      const int py = p / w;

      if(px == seed_x && py == seed_y)
        seed_label = label;

      // 4-connected neighbors
      if(py > 0 && labels[p - w] == 0 && mask[p - w] > threshold)
      {
        labels[p - w] = label;
        stack[sp++] = p - w;
      }
      if(py < h - 1 && labels[p + w] == 0 && mask[p + w] > threshold)
      {
        labels[p + w] = label;
        stack[sp++] = p + w;
      }
      if(px > 0 && labels[p - 1] == 0 && mask[p - 1] > threshold)
      {
        labels[p - 1] = label;
        stack[sp++] = p - 1;
      }
      if(px < w - 1 && labels[p + 1] == 0 && mask[p + 1] > threshold)
      {
        labels[p + 1] = label;
        stack[sp++] = p + 1;
      }
    }

    if(area > best_area)
    {
      best_area = area;
      best_label = label;
    }
  }

  // prefer component containing the seed point; fall back to largest
  const int16_t keep = (seed_label > 0) ? seed_label : best_label;

  if(keep > 0)
  {
    for(int i = 0; i < npix; i++)
    {
      if(mask[i] > threshold && labels[i] != keep)
        mask[i] = 0.0f;
    }
  }

  g_free(stack);
  g_free(labels);
}

static float _mask_iou(const float *const restrict a,
                       const float *const restrict b,
                       const size_t n,
                       const float threshold)
{
  size_t inter = 0, uni = 0;
  DT_OMP_FOR(reduction(+:inter, uni))
  for(size_t i = 0; i < n; i++)
  {
    const int A = a[i] > threshold;
    const int B = b[i] > threshold;
    inter += A & B;
    uni   += A | B;
  }
  return uni > 0 ? (float)inter / (float)uni : 0.0f;
}

// peak of the (exact-Euclidean) distance transform of mask>threshold,
// excluding pixels within min_separation of any positive prompt
static gboolean _find_peak_point(const float *const restrict mask,
                                 const size_t w,
                                 const size_t h,
                                 const float threshold,
                                 const dt_seg_point_t *const exclude,
                                 const int n_exclude,
                                 const float min_separation,
                                 dt_seg_point_t *const out)
{
  float *const restrict dist = dt_alloc_align_float(w * h);
  if(!dist) return FALSE;

  // exact-euclidean DT: dist[i] = distance to nearest pixel where mask<thr
  // require ~4 px interior depth — shallower peaks aren't informative
  const float min_depth = 4.0f;
  const float max_dist
    = dt_image_distance_transform(mask, dist, w, h,
                                  threshold, DT_DISTANCE_TRANSFORM_MASK);
  if(max_dist <= min_depth) { dt_free_align(dist); return FALSE; }

  // zero out pixels too close to existing positive prompts so the
  // subsequent argmax never picks them
  const float min_sep_sq = min_separation * min_separation;
  for(int k = 0; k < n_exclude; k++)
  {
    if(exclude[k].label != 1) continue;
    const float px = exclude[k].x;
    const float py = exclude[k].y;
    const int x0 = MAX(0, (int)(px - min_separation));
    const int x1 = MIN((int)w - 1, (int)(px + min_separation));
    const int y0 = MAX(0, (int)(py - min_separation));
    const int y1 = MIN((int)h - 1, (int)(py + min_separation));
    DT_OMP_FOR(collapse(2))
    for(int y = y0; y <= y1; y++)
      for(int x = x0; x <= x1; x++)
      {
        const float dx = (float)x - px;
        const float dy = (float)y - py;
        if(dx * dx + dy * dy < min_sep_sq) dist[(size_t)y * w + x] = 0.0f;
      }
  }

  // single-threaded combined max+argmax (exclusion may have lowered
  // the peak below max_dist, so we can't reuse that value here)
  size_t best_idx = (size_t)-1;
  float best = min_depth;
  for(size_t i = 0; i < w * h; i++)
    if(dist[i] > best) { best = dist[i]; best_idx = i; }
  dt_free_align(dist);
  if(best_idx == (size_t)-1) return FALSE;

  const size_t py = best_idx / w;
  const size_t px = best_idx % w;
  out->x = (float)px;
  out->y = (float)py;
  out->label = 1;
  return TRUE;
}

// tight bbox around mask>threshold, padded by `padding` (fraction of
// bbox extent); FALSE if mask is empty
static gboolean _compute_bbox(const float *const restrict mask,
                              const int w,
                              const int h,
                              const float threshold,
                              const float padding,
                              dt_seg_point_t *const tl,
                              dt_seg_point_t *const br)
{
  // single-threaded: cheap, and avoids OMP-reduction identity surprises
  int min_x = INT_MAX, min_y = INT_MAX, max_x = INT_MIN, max_y = INT_MIN;
  for(int y = 0; y < h; y++)
  {
    for(int x = 0; x < w; x++)
    {
      if(mask[(size_t)y * w + x] > threshold)
      {
        if(x < min_x) min_x = x;
        if(y < min_y) min_y = y;
        if(x > max_x) max_x = x;
        if(y > max_y) max_y = y;
      }
    }
  }
  if(max_x == INT_MIN) return FALSE;

  const int pad_x = (int)((max_x - min_x) * padding) + 1;
  const int pad_y = (int)((max_y - min_y) * padding) + 1;
  tl->x = (float)CLAMP(min_x - pad_x, 0, w - 1);
  tl->y = (float)CLAMP(min_y - pad_y, 0, h - 1);
  tl->label = 2;
  br->x = (float)CLAMP(max_x + pad_x, 0, w - 1);
  br->y = (float)CLAMP(max_y + pad_y, 0, h - 1);
  br->label = 3;
  return TRUE;
}

/* the n prompts forward to preview pixels, the view the encoded image shows
   at its own scale. NULL on error, free with dt_free_align */
static float *_prompts_to_preview(const GList *prompts, const int n)
{
  if(n <= 0) return NULL;
  float iwidth, iheight;
  dt_masks_get_image_size(NULL, NULL, &iwidth, &iheight);
  float *pts = dt_alloc_align_float((size_t)2 * n);
  if(!pts) return NULL;

  int k = 0;
  for(const GList *l = prompts; l; l = g_list_next(l), k++)
  {
    const dt_masks_point_object_t *pt = l->data;
    pts[k * 2] = pt->prompt.pos[0] * iwidth;
    pts[k * 2 + 1] = pt->prompt.pos[1] * iheight;
  }
  if(!dt_dev_distort_transform(darktable.develop, pts, n))
  {
    dt_free_align(pts);
    return NULL;
  }
  return pts;
}

/* whether a prompt in preview pixels is in view, which a crop made since can
   leave it out of. a pixel of slack keeps one on the edge, or rotated just
   past it, and clamps it in */
static gboolean _in_view(float *const p, const float wd, const float ht)
{
  if(p[0] < -1.0f || p[1] < -1.0f || p[0] > wd + 1.0f || p[1] > ht + 1.0f)
    return FALSE;
  p[0] = CLAMPF(p[0], 0.0f, wd);
  p[1] = CLAMPF(p[1], 0.0f, ht);
  return TRUE;
}

// the prompts in view, in encoded pixels, into out: the decoder has nothing
// to relate the others to
static int _encoded_prompts(const _object_data_t *d, dt_seg_point_t *out)
{
  float wd, ht;
  dt_masks_get_image_size(&wd, &ht, NULL, NULL);
  float *pts = _prompts_to_preview(d->prompts, g_list_length(d->prompts));
  if(!pts) return 0;

  const float sx = (wd > 0) ? (float)d->encode_w / wd : 1.0f;
  const float sy = (ht > 0) ? (float)d->encode_h / ht : 1.0f;
  int count = 0, k = 0;
  for(const GList *l = d->prompts; l; l = g_list_next(l), k++)
  {
    float *const p = pts + 2 * k;
    if(!_in_view(p, wd, ht)) continue;
    const dt_masks_point_object_t *pt = l->data;
    out[count].x = p[0] * sx;
    out[count].y = p[1] * sy;
    out[count].label = (int)pt->prompt.label;
    count++;
  }
  dt_free_align(pts);
  return count;
}

static void _run_decoder(_object_data_t *d)
{
  if(!d || !d->seg || !dt_seg_is_encoded(d->seg) || !d->prompts)
    return;

  /* dt_gui_cursor_set_busy below pumps the main loop, so an event queued
     before it is dispatched from inside this call. a second decode would
     race this one on the shared segmentation context, so it is refused;
     freeing d is deferred for as long as the flag is up (_free_data) */
  if(d->decoding)
    return;
  d->decoding = TRUE;

  // wait for encode thread: warmup may still be running after ENCODE_READY
  if(d->encode_thread)
  {
    g_thread_join(d->encode_thread);
    d->encode_thread = NULL;
  }

  // headroom: one peak point per pass + 2 box corners (SAM only)
  const int n_passes = CLAMP(dt_conf_get_int(CONF_OBJECT_REFINE_PASSES_KEY),
                             1, 3);
  dt_seg_point_t *points = g_new(dt_seg_point_t,
                                 g_list_length(d->prompts) + n_passes + 2);
  // every decode sends all the prompts in view at once
  int n_points = _encoded_prompts(d, points);
  if(n_points == 0)
  {
    g_free(points);
    d->decoding = FALSE;
    return;
  }

  dt_gui_cursor_set_busy();

  // the connected component filter keeps what the last foreground prompt is in
  int seed_x = -1, seed_y = -1;
  for(int i = n_points - 1; i >= 0; i--)
  {
    if(points[i].label == 1)
    {
      seed_x = (int)points[i].x;
      seed_y = (int)points[i].y;
      break;
    }
  }

  const float threshold
    = CLAMP(dt_conf_get_float(CONF_OBJECT_THRESHOLD_KEY), 0.3f, 0.9f);
  const gboolean supports_box = dt_seg_supports_box(d->seg);
  int mw = 0, mh = 0;
  float *mask = NULL;
  gboolean box_added = FALSE;

  for(int pass = 0; pass < n_passes; pass++)
  {
    float *new_mask = dt_seg_compute_mask(d->seg, points, n_points, &mw, &mh);
    if(!new_mask) break;

    if(mask && _mask_iou(mask, new_mask, (size_t)mw * mh, threshold) > 0.99f)
    {
      g_free(mask);
      mask = new_mask;
      dt_print(DT_DEBUG_AI,
               "[object mask] converged at pass %d/%d", pass + 1, n_passes);
      break;
    }
    g_free(mask);
    mask = new_mask;

    if(pass + 1 >= n_passes) break;

    gboolean any_added = FALSE;
    dt_seg_point_t peak;
    if(_find_peak_point(mask, mw, mh, threshold,
                        points, n_points, 8.0f, &peak))
    {
      points[n_points++] = peak;
      any_added = TRUE;
    }
    if(supports_box && !box_added)
    {
      dt_seg_point_t tl, br;
      if(_compute_bbox(mask, mw, mh, threshold, 0.05f, &tl, &br))
      {
        points[n_points++] = tl;
        points[n_points++] = br;
        box_added = TRUE;
        any_added = TRUE;
      }
    }
    if(!any_added) break;
  }
  g_free(points);

  if(mask)
  {
    // remove disconnected blobs: keep only the component at the seed point
    seed_x = CLAMP(seed_x, 0, mw - 1);
    seed_y = CLAMP(seed_y, 0, mh - 1);
    _keep_seed_component(mask, mw, mh, threshold, seed_x, seed_y);

    // optional DenseCRF edge refinement using the encoded RGB as guide
    if(dt_conf_get_bool(CONF_OBJECT_REFINE_BOUNDARY_KEY))
    {
      int rgb_w = 0, rgb_h = 0;
      const uint8_t *rgb = dt_seg_get_encoded_rgb(d->seg, &rgb_w, &rgb_h);
      if(rgb && rgb_w == mw && rgb_h == mh)
      {
        const int crf_iter
          = CLAMP(dt_conf_get_int(CONF_OBJECT_REFINE_BOUNDARY_ITER_KEY),
                  1, 10);
        const float crf_sigma_color
          = CLAMP(dt_conf_get_float(CONF_OBJECT_REFINE_BOUNDARY_SIGMA_COLOR_KEY),
                  1.0f, 50.0f);
        const float crf_w_bilateral
          = CLAMP(dt_conf_get_float(CONF_OBJECT_REFINE_BOUNDARY_W_BILATERAL_KEY),
                  0.5f, 30.0f);
        const double t0 = dt_get_wtime();
        dt_dense_crf_binary(mask, rgb, mw, mh,
                            5.0f, crf_sigma_color,
                            3.0f, crf_w_bilateral, crf_iter);
        dt_print(DT_DEBUG_AI,
                 "[object mask] CRF refinement: %dx%d (%.2fs)",
                 mw, mh, dt_get_wtime() - t0);
      }
    }

    g_free(d->mask);
    d->mask = mask;
    d->mask_w = mw;
    d->mask_h = mh;
    _schedule_outline(d);
  }
  d->decoding = FALSE;
  dt_gui_cursor_clear_busy();
}

// transform mask-space forms to input-normalized coords and register them,
// takes ownership of `forms` and `signs` lists (forms are appended to dev->forms)
static dt_masks_form_t *
_register_vectorized_forms(GList *forms,
                           GList *signs,
                           const int mask_w,
                           const int mask_h)
{
  // darktable mask coordinates are stored in input-image-normalized space:
  //   coord = backtransform(backbuf_pixel) / iwidth
  // this undoes all geometric pipeline transforms (crop, rotation, lens, etc.)
  // so that the mask can be applied at any point in the pipeline
  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);

  // vectorized coordinates are in mask space (encoding resolution),
  // dt_dev_distort_backtransform expects preview pipe pixel space
  const float msx = (mask_w > 0) ? wd / (float)mask_w : 1.0f;
  const float msy = (mask_h > 0) ? ht / (float)mask_h : 1.0f;

  for(GList *l = forms; l; l = g_list_next(l))
  {
    dt_masks_form_t *f = l->data;
    const int npts = g_list_length(f->points);
    if(npts == 0)
      continue;

    // collect all coordinates into a flat array for batch backtransform,
    // each path point has 3 coordinate pairs: corner, ctrl1, ctrl2
    float *pts = g_new(float, npts * 6);
    int i = 0;
    for(GList *p = f->points; p; p = g_list_next(p))
    {
      dt_masks_point_path_t *pt = p->data;
      pts[i++] = pt->corner[0];
      pts[i++] = pt->corner[1];
      pts[i++] = pt->ctrl1[0];
      pts[i++] = pt->ctrl1[1];
      pts[i++] = pt->ctrl2[0];
      pts[i++] = pt->ctrl2[1];
    }

    // scale from mask space (encoding resolution) to preview pipe space
    for(int j = 0; j < npts * 6; j += 2)
    {
      pts[j + 0] *= msx;
      pts[j + 1] *= msy;
    }

    dt_dev_distort_backtransform(darktable.develop, pts, npts * 3);

    // write back and normalize by input image dimensions
    i = 0;
    for(GList *p = f->points; p; p = g_list_next(p))
    {
      dt_masks_point_path_t *pt = p->data;
      pt->corner[0] = pts[i++] / iwidth;
      pt->corner[1] = pts[i++] / iheight;
      pt->ctrl1[0] = pts[i++] / iwidth;
      pt->ctrl1[1] = pts[i++] / iheight;
      pt->ctrl2[0] = pts[i++] / iwidth;
      pt->ctrl2[1] = pts[i++] / iheight;
    }
    g_free(pts);
  }

  const int nbform = g_list_length(forms);
  if(nbform == 0)
  {
    g_list_free_full(forms, (GDestroyNotify)dt_masks_free_form);
    g_list_free(signs);
    dt_control_log(_("no mask extracted from AI segmentation"));
    return NULL;
  }

  // always wrap paths in a group; holes use difference mode

  // count existing AI object groups/paths for numbering
  dt_develop_t *dev = darktable.develop;
  const char *group_prefix = _("ai object group");
  const char *path_prefix = _("ai object");

  guint grp_nb = 0;
  guint path_nb = 0;
  for(GList *l = dev->forms; l; l = g_list_next(l))
  {
    const dt_masks_form_t *f = l->data;
    if(strncmp(f->name, group_prefix, strlen(group_prefix)) == 0)
      grp_nb++;
    if(strncmp(f->name, path_prefix, strlen(path_prefix)) == 0)
      path_nb++;
  }
  grp_nb++;
  path_nb++;
  for(GList *l = forms; l; l = g_list_next(l))
  {
    dt_masks_form_t *f = l->data;
    snprintf(f->name, sizeof(f->name),
             "%s #%d", path_prefix, (int)path_nb++);
  }

  dt_masks_form_t *grp = dt_masks_create(DT_MASKS_GROUP);
  snprintf(grp->name, sizeof(grp->name), "%s #%d", group_prefix, (int)grp_nb);

  // register all path forms so they exist in dev->forms
  for(GList *l = forms; l; l = g_list_next(l))
  {
    dt_masks_form_t *f = l->data;
    dev->forms = g_list_append(dev->forms, f);
  }

  // add each path to the group; holes get difference mode
  GList *s = signs;
  for(GList *l = forms; l; l = g_list_next(l), s = s ? g_list_next(s) : NULL)
  {
    dt_masks_form_t *f = l->data;
    const int sign = s ? GPOINTER_TO_INT(s->data) : '+';
    dt_masks_point_group_t *grpt = dt_masks_group_add_form(grp, f);
    if(grpt && sign == '-')
    {
      grpt->state = (grpt->state & ~DT_MASKS_STATE_UNION) | DT_MASKS_STATE_DIFFERENCE;
    }
  }

  // register the group (history item added by caller after blend mask
  // assignment)
  dev->forms = g_list_append(dev->forms, grp);

  g_list_free(forms);
  g_list_free(signs);

  dt_print(DT_DEBUG_MASKS, "[object mask] created %d paths", nbform);
  return grp;
}

/* the selection stored in the sidecar: the form's new points, or NULL to
   leave it alone. outside the encoded view an edit keeps old's mask */
static GList *_finalize_raster(const _object_data_t *d, const dt_masks_form_t *old)
{
  if(!d || !d->mask || d->mask_w <= 0 || d->mask_h <= 0 || !d->prompts)
  {
    dt_print(DT_DEBUG_MASKS, "[object mask] raster: no mask buffer");
    return NULL;
  }

  // the mask lives after crop, rotate and lens, but is stored before them,
  // like every form's points, so it survives a later crop
  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);
  if(wd <= 0.0f || ht <= 0.0f || iwidth <= 0.0f || iheight <= 0.0f)
  {
    dt_print(DT_DEBUG_MASKS, "[object mask] raster: bad image size %fx%f / %fx%f",
             wd, ht, iwidth, iheight);
    return NULL;
  }

  const int tw = (iwidth >= iheight)
    ? RASTER_STORE_SIZE
    : (int)(RASTER_STORE_SIZE * iwidth / iheight);
  const int th = (iwidth >= iheight)
    ? (int)(RASTER_STORE_SIZE * iheight / iwidth)
    : RASTER_STORE_SIZE;
  if(tw <= 0 || th <= 0)
  {
    dt_print(DT_DEBUG_MASKS, "[object mask] raster: bad target size %dx%d", tw, th);
    return NULL;
  }

  float *pts = dt_alloc_align_float((size_t)2 * tw * th);
  float *out = dt_alloc_align_float((size_t)tw * th);
  if(!pts || !out)
  {
    dt_free_align(pts);
    dt_free_align(out);
    return NULL;
  }

  DT_OMP_FOR(collapse(2))
  for(int y = 0; y < th; y++)
    for(int x = 0; x < tw; x++)
    {
      const size_t k = (size_t)y * tw + x;
      pts[k * 2] = (x + 0.5f) * iwidth / tw;
      pts[k * 2 + 1] = (y + 0.5f) * iheight / th;
    }

  // input pixels forward to where the encode pipe saw them
  if(!dt_dev_distort_transform(darktable.develop, pts, (size_t)tw * th))
  {
    dt_print(DT_DEBUG_MASKS, "[object mask] raster: distort transform failed");
    dt_free_align(pts);
    dt_free_align(out);
    return NULL;
  }

  const float msx = wd / (float)d->mask_w;
  const float msy = ht / (float)d->mask_h;

  _raster_cache_t *oc = old
    ? _raster_get(darktable.develop->image_storage.id, _raster_ref(old))
    : NULL;
  const uint8_t *const om = oc ? oc->mask : NULL;
  const int ow = oc ? oc->width : 0, oh = oc ? oc->height : 0;

  DT_OMP_FOR()
  for(int k = 0; k < tw * th; k++)
  {
    const float mx = pts[k * 2] / msx;
    const float my = pts[k * 2 + 1] / msy;
    if(mx < 0.0f || my < 0.0f || mx > d->mask_w - 1 || my > d->mask_h - 1)
      out[k] = om ? _sample(om, ow, oh, (k % tw + 0.5f) / tw, (k / tw + 0.5f) / th) : 0.0f;
    else
      out[k] = _bilinear(d->mask, d->mask_w, d->mask_h, mx, my);
  }
  _raster_release(oc);
  dt_free_align(pts);

  const dt_imgid_t imgid = darktable.develop->image_storage.id;
  char *model = dt_ai_models_get_active_for_task("mask");
  dt_dtdata_ref_t ref = { 0 };
  const gboolean ok =
    dt_dtdata_write_gray(imgid, DT_DTDATA_KIND_MASK, DT_DTDATA_ORIGIN_REGENERABLE,
                         model ? model : "", out, tw, th, 8, &ref);
  g_free(model);
  dt_free_align(out);
  if(!ok) return NULL;
  _raster_forget_failure(imgid, ref.entry);

  dt_print(DT_DEBUG_MASKS, "[object mask] raster: stored '%s' %dx%d", ref.entry, tw, th);

  // the reference, then the prompts in click order
  GList *prompts = _copy_prompts(d->prompts);
  dt_masks_point_object_t *head = prompts ? calloc(1, sizeof(dt_masks_point_object_t)) : NULL;
  if(!head)
  {
    g_list_free_full(prompts, free);
    return NULL;
  }
  head->ref = ref;
  return g_list_prepend(prompts, head);
}

// --- mask event handlers ---

static int _object_events_mouse_scrolled(dt_iop_module_t *module,
                                         const float pzx,
                                         const float pzy,
                                         const gboolean up,
                                         const uint32_t state,
                                         dt_masks_form_t *form,
                                         const dt_imgid_t parentid,
                                         dt_masks_form_gui_t *gui,
                                         const int index)
{
  _object_data_t *d = _get_data(gui);

  // trace settings, only while applying as paths (after first click).
  // otherwise scroll is not taken, as before the first click
  if(d && d->has_selection && d->mask && _commits_paths(form))
  {
    if(dt_modifier_is(state, 0))
    {
      // plain scroll: adjust smoothing (potrace alphamax)
      const float smoothing =
        CLAMP(dt_conf_get_float(CONF_OBJECT_SMOOTHING_KEY) + (up ? 0.05f : -0.05f),
              0.0f, 1.3f);
      dt_conf_set_float(CONF_OBJECT_SMOOTHING_KEY, smoothing);
      dt_toast_log(_("smoothing: %3.2f"), smoothing);
      dt_dev_masks_list_change(darktable.develop);
      _schedule_outline(d);
      dt_control_queue_redraw_center();
      return 1;
    }
    if(dt_modifier_is(state, GDK_SHIFT_MASK))
    {
      // shift+scroll: adjust cleanup (potrace turdsize)
      const int cleanup =
        CLAMP(dt_conf_get_int(CONF_OBJECT_CLEANUP_KEY) + (up ? 5 : -5), 0, 100);
      dt_conf_set_int(CONF_OBJECT_CLEANUP_KEY, cleanup);
      dt_toast_log(_("cleanup: %d"), cleanup);
      dt_dev_masks_list_change(darktable.develop);
      _schedule_outline(d);
      dt_control_queue_redraw_center();
      return 1;
    }
  }

  // opacity control (ctrl+scroll)
  if(dt_modifier_is(state, GDK_CONTROL_MASK))
  {
    float opacity = dt_conf_get_float("plugins/darkroom/masks/opacity");
    opacity = CLAMP(opacity + (up ? 0.05f : -0.05f), 0.05f, 1.0f);
    dt_conf_set_float("plugins/darkroom/masks/opacity", opacity);
    dt_toast_log(_("opacity: %d%%"), (int)(opacity * 100.0f));
    dt_dev_masks_list_change(darktable.develop);
    dt_control_queue_redraw_center();
    return 1;
  }
  return 0;
}

/* clear the selection with its outline and the decoder's refinement state,
   and the prompts in view. the others stay, as a commit keeps the stored
   mask out of view */
static void _clear_selection(_object_data_t *d)
{
  float wd, ht;
  dt_masks_get_image_size(&wd, &ht, NULL, NULL);
  float *pts = _prompts_to_preview(d->prompts, g_list_length(d->prompts));
  int k = 0;
  for(GList *l = d->prompts; l; k++)
  {
    GList *next = g_list_next(l);
    if(!pts || _in_view(pts + 2 * k, wd, ht))
    {
      free(l->data);
      d->prompts = g_list_delete_link(d->prompts, l);
    }
    l = next;
  }
  dt_free_align(pts);

  g_free(d->mask);
  d->mask = NULL;
  d->mask_w = d->mask_h = 0;

  if(d->seg)
    dt_seg_reset_prev_mask(d->seg);

  // reset selection and outline state
  d->has_selection = FALSE;
  _free_outline(d);

  dt_control_queue_redraw_center();
}

/* an edit resumes from the stored mask, as the selection and the decoder's
   previous mask; with the entry lost, it is decoded again from the prompts.
   the decode pumps the main loop, so this must not run from a draw handler:
   it is driven from the click that adds to the edit and from an idle the
   redraw schedules, whichever comes first. `decode` is FALSE for the click,
   which runs the decoder itself once its own prompt is in */
static void _resume_edit(_object_data_t *d,
                         const dt_masks_form_t *form,
                         const gboolean decode)
{
  const dt_dtdata_ref_t *ref = _raster_ref(form);
  // nothing to resume from without the encoding the stored mask maps onto,
  // so the flag stays clear and the next click or redraw tries again
  if(!ref || !d->seg || d->encode_w <= 0 || d->encode_h <= 0) return;

  d->restored = TRUE;
  d->changed = FALSE;

  _raster_cache_t *c = _raster_get(darktable.develop->image_storage.id, ref);
  // g_malloc, as every other d->mask
  float *mask = c ? g_try_malloc(sizeof(float) * d->encode_w * d->encode_h) : NULL;
  if(mask && _raster_to_preview(c, d->encode_w, d->encode_h, mask))
  {
    g_free(d->mask);
    d->mask = mask;
    d->mask_w = d->encode_w;
    d->mask_h = d->encode_h;
    d->has_selection = TRUE;
    dt_seg_set_prev_mask(d->seg, mask, d->mask_w, d->mask_h);
  }
  else
    g_free(mask);
  _raster_release(c);

  // the stored prompts first: a click can land before the edit is restored
  d->prompts = g_list_concat(_copy_prompts(g_list_next(form->points)), d->prompts);
  if(!d->mask && d->prompts)
  {
    // the context outlives sessions, so no earlier mask may leak in
    dt_seg_reset_prev_mask(d->seg);
    d->has_selection = TRUE;
    if(decode)
    {
      _run_decoder(d);
      d->changed = d->mask != NULL;
    }
  }
}

/* the redraw after the image is encoded brings the stored mask back, but
   from here rather than from the draw handler itself */
static gboolean _resume_edit_cb(gpointer data)
{
  _object_data_t *d = data;
  d->resume_id = 0;
  const dt_masks_form_t *form = darktable.develop->form_visible;
  // the tool may have been left, or another form opened, since this was
  // queued, and the scratchpad is then no longer the edit's
  if(_get_data(darktable.develop->form_gui) == d
     && !d->restored
     && g_atomic_int_get(&d->encode_state) == ENCODE_READY
     && _is_edit(form))
  {
    _resume_edit(d, form, TRUE);
    // only when something came back: a redraw after a restore that failed
    // would arm this again from post_expose, and so on at frame rate
    if(d->restored)
      dt_control_queue_redraw_center();
  }
  return G_SOURCE_REMOVE;
}

/* leave the tool, whether something was committed or not, and select what
   it leaves behind, which also rebuilds the mask manager right away rather
   than on its next lazy redraw */
static void _leave_creation(dt_iop_module_t *module,
                            dt_masks_form_gui_t *gui,
                            const dt_mask_id_t select)
{
  gui->creation = FALSE;
  gui->creation_continuous = FALSE;
  gui->creation_continuous_module = NULL;
  gui->creation_module = NULL;

  _free_data(gui);

  dt_control_hinter_message("");

  // dt_masks_set_edit_mode requires a non-NULL module (it returns
  // immediately otherwise), so clear the form directly when module
  // is NULL (standalone mask creation)
  if(module)
  {
    dt_masks_set_edit_mode(module, DT_MASKS_EDIT_FULL);
    dt_masks_iop_update(module);
  }
  else
  {
    dt_masks_change_form_gui(NULL);
  }
  if(dt_is_valid_maskid(select))
    dt_dev_masks_selection_change(darktable.develop, module, select);
  dt_control_queue_redraw_center();
}

void dt_masks_object_cancel_edit(void)
{
  dt_develop_t *dev = darktable.develop;
  dt_masks_form_gui_t *gui = dev->form_gui;
  const dt_masks_form_t *form = dev->form_visible;
  // a new object has nothing to go back to, so escape leaves it alone
  if(!gui || !gui->creation || !form || !(form->type & DT_MASKS_OBJECT)
     || !_is_edit(form))
    return;

  // as for right-click: don't exit while background threads are running
  _object_data_t *d = _get_data(gui);
  if(d && g_atomic_int_get(&d->encode_state) == ENCODE_RUNNING)
    return;

  _leave_creation(gui->creation_module, gui, form->formid);
}

/* an edit goes to the form listed under its id now, since a history change
   replaces the edited one. unchanged or cleared, the object stays as it was */
static dt_mask_id_t _commit_edit(dt_iop_module_t *module,
                                 const _object_data_t *d,
                                 const dt_masks_form_t *form)
{
  dt_masks_form_t *live = dt_masks_get_from_id(darktable.develop, form->formid);
  if(!live)
  {
    dt_control_log(_("the object was removed, the edit is discarded"));
    return NO_MASKID;
  }
  if(d && d->has_selection && d->mask && d->changed)
  {
    GList *points = _finalize_raster(d, live);
    if(points)
    {
      g_list_free_full(live->points, free);
      live->points = points;
      dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
    }
    else
      dt_control_log(_("could not store the object in the sidecar, it is not changed"));
  }
  return live->formid;
}

// a new object traced into paths, grouped into the module's mask group
static dt_mask_id_t _commit_paths(dt_iop_module_t *module, const _object_data_t *d)
{
  GList *forms = NULL, *signs = NULL;
  if(!d || !d->mask || !_trace(d, &forms, &signs)) return NO_MASKID;
  dt_masks_form_t *grp = _register_vectorized_forms(forms, signs, d->mask_w, d->mask_h);
  if(!grp) return NO_MASKID;

  dt_develop_t *dev = darktable.develop;
  if(module)
  {
    dt_masks_form_t *mod_grp = dt_masks_get_from_id(dev, module->blend_params->mask_id);
    if(!mod_grp)
    {
      mod_grp = dt_masks_create(DT_MASKS_GROUP);
      gchar *module_label = dt_history_item_get_name(module);
      snprintf(mod_grp->name, sizeof(mod_grp->name),
               _("group '%s'"), module_label);
      g_free(module_label);
      dev->forms = g_list_append(dev->forms, mod_grp);
      module->blend_params->mask_id = mod_grp->formid;
    }
    dt_masks_point_group_t *grpt = dt_masks_group_add_form(mod_grp, grp);
    if(grpt)
      grpt->opacity = dt_conf_get_float("plugins/darkroom/masks/opacity");
  }
  dt_dev_add_masks_history_item(dev, module, TRUE);
  return grp->formid;
}

// a new object as pixels, or as paths when they cannot be stored
static dt_mask_id_t _commit_pixels(dt_iop_module_t *module,
                                   dt_masks_form_gui_t *gui,
                                   const _object_data_t *d,
                                   dt_masks_form_t *form)
{
  GList *points = _finalize_raster(d, NULL);
  if(!points)
  {
    dt_control_log(_("could not store the object in the sidecar, it is saved as paths"));
    return _commit_paths(module, d);
  }
  form->points = points;
  dt_masks_gui_form_save_creation(darktable.develop, module, form, gui);
  return form->formid;
}

static int _object_events_button_pressed(dt_iop_module_t *module,
                                         float pzx,
                                         float pzy,
                                         const double pressure,
                                         const int which,
                                         const int type,
                                         const uint32_t state,
                                         dt_masks_form_t *form,
                                         const dt_imgid_t parentid,
                                         dt_masks_form_gui_t *gui,
                                         const int index)
{
  (void)pressure;
  (void)parentid;
  (void)index;
  if(type == GDK_2BUTTON_PRESS || type == GDK_3BUTTON_PRESS)
    return 1;

  _object_data_t *d = _get_data(gui);

  if(which == 1 && dt_modifier_is(state, GDK_CONTROL_MASK | GDK_SHIFT_MASK))
  {
    // ctrl+shift+click: clear selection (only after first selection)
    if(d && d->has_selection && d->encode_state == ENCODE_READY)
    {
      _clear_selection(d);
      if(darktable.develop->proxy.masks.module)
        darktable.develop->proxy.masks.list_change(
          darktable.develop->proxy.masks.module);
    }
    return 1;
  }
  else if(which == 1)
  {
    // off the image, a click would be a prompt the decoder never sees
    if(pzx < 0.0f || pzy < 0.0f || pzx >= 1.0f || pzy >= 1.0f)
      return 1;

    // need valid scratchpad and completed encoding
    if(!d || d->encode_state != ENCODE_READY)
      return 1;

    // dismiss the "ready" hint now that the user is interacting
    dt_control_log_ack_all();

    // start drag tracking, resolved as click on button release
    float wd, ht, iwidth, iheight;
    dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);

    d->dragging = TRUE;
    d->drag_start_x = pzx * wd;
    d->drag_start_y = pzy * ht;
    return 1;
  }
  else if(which == 3)
  {
    // don't exit while background threads are running
    if(d && g_atomic_int_get(&d->encode_state) == ENCODE_RUNNING)
      return 1;

    // the module the tool started for, not the focused one. none from the mask
    // manager without a module's row, or for an object not among its masks
    dt_iop_module_t *crea_module = gui->creation_module;
    const gboolean has_mask = d && d->has_selection && d->mask;
    dt_mask_id_t select = NO_MASKID;
    if(_is_edit(form))
      select = _commit_edit(crea_module, d, form);
    else if(has_mask && _commits_paths(form))
      select = _commit_paths(crea_module, d);
    else if(has_mask)
      select = _commit_pixels(crea_module, gui, d, form);
    _leave_creation(crea_module, gui, select);
    return 1;
  }

  return 0;
}

static int _object_events_button_released(dt_iop_module_t *module,
                                          const float pzx,
                                          const float pzy,
                                          const int which,
                                          const uint32_t state,
                                          dt_masks_form_t *form,
                                          const dt_imgid_t parentid,
                                          dt_masks_form_gui_t *gui,
                                          const int index)
{
  (void)module;
  (void)pzx;
  (void)pzy;
  (void)parentid;
  (void)index;

  if(which != 1)
    return 0;

  _object_data_t *d = _get_data(gui);
  /* dispatched from the main loop a decode pumps: the prompt would go on the
     list while the decode that should answer it is refused, leaving the mask
     and the prompts describing different selections */
  if(d && d->decoding)
    return 1;

  if(!d || !d->dragging)
    return 0;

  d->dragging = FALSE;

  /* the stored mask and prompts come back before this click is added to
     them: a click is accepted as soon as the encoding is ready, which can
     be before the redraw that queued the restore has run. the decoder runs
     once below, on the restored prompts and this one together */
  if(!d->restored && _is_edit(form))
    _resume_edit(d, form, FALSE);

  // calloc: the union leaves most of a prompt unused, and the blob is
  // hashed and stored whole
  dt_masks_point_object_t *prompt = calloc(1, sizeof(dt_masks_point_object_t));
  if(!prompt)
    return 1;

  // back to input-image coordinates once, like every form's points
  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);
  float pt[2] = { d->drag_start_x, d->drag_start_y };
  dt_dev_distort_backtransform(darktable.develop, pt, 1);
  prompt->prompt.pos[0] = pt[0] / iwidth;
  prompt->prompt.pos[1] = pt[1] / iheight;
  // click: foreground point, shift+click: background point (only
  // after first selection)
  prompt->prompt.label = (d->has_selection && dt_modifier_is(state, GDK_SHIFT_MASK))
    ? 0.0f : 1.0f;
  d->prompts = g_list_append(d->prompts, prompt);
  d->has_selection = TRUE;
  d->changed = TRUE;

  _run_decoder(d);

  // refresh mask properties panel so sliders update for
  // the current creation step (size vs cleanup/smoothing)
  if(darktable.develop->proxy.masks.module)
    darktable.develop->proxy.masks.list_change(darktable.develop->proxy.masks.module);

  dt_control_queue_redraw_center();
  return 1;
}

static int _object_events_mouse_moved(dt_iop_module_t *module,
                                      const float pzx,
                                      const float pzy,
                                      const double pressure,
                                      const int which,
                                      const float zoom_scale,
                                      dt_masks_form_t *form,
                                      const dt_imgid_t parentid,
                                      dt_masks_form_gui_t *gui,
                                      const int index)
{
  (void)module;
  (void)pressure;
  (void)which;
  (void)zoom_scale;
  (void)form;
  (void)parentid;
  (void)index;

  gui->form_selected = FALSE;
  gui->border_selected = FALSE;
  gui->source_selected = FALSE;
  gui->feather_selected = -1;
  gui->point_selected = -1;
  gui->seg_selected = -1;
  gui->point_border_selected = -1;

  dt_control_queue_redraw_center();

  return 1;
}

// timer callback: periodically redraw center so +/- cursor tracks shift key
static gboolean _modifier_poll(gpointer data)
{
  (void)data;
  dt_control_queue_redraw_center();
  return G_SOURCE_CONTINUE;
}

static void _object_events_post_expose(cairo_t *cr,
                                       const float zoom_scale,
                                       dt_masks_form_gui_t *gui,
                                       const int index,
                                       const int num_points)
{
  (void)index;
  (void)num_points;

  // ensure scratchpad exists
  _object_data_t *d = _get_data(gui);
  if(!d)
  {
    d = g_new0(_object_data_t, 1);

    // restore persistent model (stays loaded across mask sessions)
    // if the active model changed in preferences, discard the old one
    {
      dt_ai_seg_t *ps = &darktable.ai_seg;
      char *active = dt_ai_models_get_active_for_task("mask");
      const char *persistent_id = dt_seg_get_model_id(ps->ctx);
      if(ps->ctx && active
         && g_strcmp0(active, persistent_id) != 0)
      {
        dt_print(DT_DEBUG_AI,
                 "[object mask] model changed (%s -> %s), "
                 "discarding persistent model",
                 persistent_id, active);
        dt_seg_free(ps->ctx);
        ps->ctx = NULL;
        dt_ai_env_destroy(ps->env);
        ps->env = NULL;
        ps->model_loaded = FALSE;
      }
      g_free(active);
      d->env = ps->env;
      d->seg = ps->ctx;
      d->model_loaded = ps->model_loaded;
      ps->env = NULL;
      ps->ctx = NULL;
      ps->model_loaded = FALSE;

      // connect view-change signal once to free model on darkroom exit
      if(!ps->signal_connected)
      {
        DT_CONTROL_SIGNAL_CONNECT(DT_SIGNAL_VIEWMANAGER_VIEW_CHANGED,
                                  _on_view_changed, NULL);
        ps->signal_connected = TRUE;
      }
    }

    gui->scratchpad = d;
    gui->scratchpad_cleanup = _free_data;
  }

  // detect distortion changes (crop/rotate on same image):
  // reset encoding so the image is re-analyzed
  const dt_imgid_t cur_imgid = darktable.develop->image_storage.id;
  const int cur_state = g_atomic_int_get(&d->encode_state);
  /* not while a decode is running: this draw can be dispatched from the main
     loop _run_decoder pumps, and the reset frees the mask, resets the
     segmentation context and drops the prompts the decode is working from.
     everything below only reads d, so the rest of the draw is left alone */
  if(!d->decoding
     && (cur_state == ENCODE_READY || cur_state == ENCODE_ERROR)
     && (d->encoded_imgid != cur_imgid
         || d->encoded_distort_hash != _compute_distort_hash(darktable.develop)))
  {
    if(d->encode_thread)
    {
      g_thread_join(d->encode_thread);
      d->encode_thread = NULL;
    }
    if(d->seg)
      dt_seg_reset_encoding(d->seg);
    g_free(d->mask);
    d->mask = NULL;
    d->mask_w = d->mask_h = 0;
    d->encode_w = d->encode_h = 0;
    d->encode_state = ENCODE_IDLE;
    // reset selection, outline, and prompts so the new image starts fresh
    d->has_selection = FALSE;
    d->restored = FALSE;
    d->changed = FALSE;
    _clear_prompts(d);
    _free_outline(d);
  }

  // eager encoding: load model and encode image as soon as tool opens
  if(d->encode_state == ENCODE_IDLE)
  {
    dt_control_log(_("object mask: analyzing image..."));
    d->encode_state = ENCODE_MSG_SHOWN;
    dt_control_queue_redraw_center();
    return;
  }

  if(d->encode_state == ENCODE_MSG_SHOWN)
  {
    // frame 2: launch background thread to render and encode the image.
    // the thread creates a temporary export pipe at high resolution
    // instead of using the low-res preview backbuf.
    // flush history to database so the encode thread's dt_dev_load_image
    // sees the current edits (crop/rotate may not be flushed yet)
    dt_dev_write_history(darktable.develop);

    const dt_hash_t cur_hash = _compute_distort_hash(darktable.develop);

    _encode_thread_data_t *td = g_new(_encode_thread_data_t, 1);
    td->d = d;
    td->imgid = cur_imgid;
    td->history_end = darktable.develop->history_end;
    td->distort_hash = cur_hash;

    d->encoded_imgid = cur_imgid;
    d->encoded_distort_hash = cur_hash;
    d->encode_state = ENCODE_RUNNING;
    // start poll timer BEFORE the thread, it will detect completion
    // and also tracks modifier keys once encoding is ready
    if(!d->modifier_poll_id)
      d->modifier_poll_id = g_timeout_add(100, _modifier_poll, NULL);
    d->encode_thread = g_thread_new("ai-mask-encode", _encode_thread_func, td);
    return;
  }

  if(g_atomic_int_get(&d->encode_state) == ENCODE_RUNNING)
  {
    // keep the message visible while the thread is working
    dt_control_log(_("object mask: analyzing image..."));
    return;
  }

  if(g_atomic_int_get(&d->encode_state) == ENCODE_READY && d->encode_thread)
  {
    // thread finished (detected by poll timer redraw), join it
    g_thread_join(d->encode_thread);
    d->encode_thread = NULL;
    dt_control_log_ack_all();
    if(!_is_edit(darktable.develop->form_visible))
      dt_control_log(_("click on object to create mask"));
  }

  if(g_atomic_int_get(&d->encode_state) == ENCODE_ERROR)
  {
    if(d->encode_thread)
    {
      g_thread_join(d->encode_thread);
      d->encode_thread = NULL;
      // log only once when the thread is first joined
      dt_control_log(_("object mask preparation failed"));
    }
    return;
  }

  if(d->encode_state != ENCODE_READY)
    return;

  /* the restore reads the sidecar and may decode the prompts again, which
     pumps the main loop: it cannot run from here, inside cairo's draw, where
     an event dispatched on the way would commit the edit and free d */
  if(!d->restored && !d->resume_id && _is_edit(darktable.develop->form_visible))
    d->resume_id = g_idle_add(_resume_edit_cb, d);

  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);

  // --- Draw red overlay of current mask ---
  if(d->mask && d->mask_w > 0 && d->mask_h > 0)
  {
    const float mask_thresh = CLAMP(dt_conf_get_float(CONF_OBJECT_THRESHOLD_KEY), 0.3f, 0.9f);
    cairo_surface_t *surface = _tint(d->mask, d->mask_w, d->mask_h, FALSE, mask_thresh);
    if(surface)
    {
      cairo_save(cr);
      cairo_scale(cr, wd / d->mask_w, ht / d->mask_h);
      cairo_set_source_surface(cr, surface, 0, 0);
      cairo_paint(cr);
      cairo_restore(cr);
      cairo_surface_destroy(surface);
    }
  }

  // draw the outline (real path style with anchor dots)
  if(d->outline_forms)
  {
    const float msx = (d->mask_w > 0) ? wd / (float)d->mask_w : 1.0f;
    const float msy = (d->mask_h > 0) ? ht / (float)d->mask_h : 1.0f;

    for(GList *fl = d->outline_forms; fl; fl = g_list_next(fl))
    {
      dt_masks_form_t *f = fl->data;
      GList *pts = f->points;
      if(!pts) continue;

      dt_masks_point_path_t *first_pt = pts->data;
      cairo_move_to(cr,
                    first_pt->corner[0] * msx,
                    first_pt->corner[1] * msy);

      // cairo_curve_to(c1, c2, end) expects:
      //   c1 = outgoing handle of previous point (prev.ctrl2)
      //   c2 = incoming handle of this point (this.ctrl1)
      dt_masks_point_path_t *prev_pt = first_pt;
      for(GList *p = g_list_next(pts); p; p = g_list_next(p))
      {
        dt_masks_point_path_t *pt = p->data;
        cairo_curve_to(cr,
                       prev_pt->ctrl2[0] * msx, prev_pt->ctrl2[1] * msy,
                       pt->ctrl1[0] * msx, pt->ctrl1[1] * msy,
                       pt->corner[0] * msx, pt->corner[1] * msy);
        prev_pt = pt;
      }

      // close path back to first point
      cairo_curve_to(cr,
                     prev_pt->ctrl2[0] * msx, prev_pt->ctrl2[1] * msy,
                     first_pt->ctrl1[0] * msx, first_pt->ctrl1[1] * msy,
                     first_pt->corner[0] * msx, first_pt->corner[1] * msy);

      dt_masks_line_stroke(cr, FALSE, FALSE, FALSE, zoom_scale);

      for(GList *p = pts; p; p = g_list_next(p))
      {
        dt_masks_point_path_t *pt = p->data;
        dt_masks_draw_anchor(cr, FALSE, zoom_scale,
                             pt->corner[0] * msx, pt->corner[1] * msy);
      }
    }
  }

  // query pointer position and modifier state directly from GDK so the
  // cursor is drawn at the correct location even before the first
  // mouse_moved event fires.
  GtkWidget *cw = dt_ui_center(darktable.gui->ui);
  GdkWindow *win = gtk_widget_get_window(cw);
  GdkDevice *pointer = gdk_seat_get_pointer
    (gdk_display_get_default_seat(gdk_display_get_default()));
  GdkModifierType mod = 0;
  int dev_x = 0, dev_y = 0;
  if(win && pointer)
    gdk_window_get_device_position(win, pointer, &dev_x, &dev_y, &mod);

  // skip indicator when pointer is over a window above us (e.g. prefs)
  if(pointer
     && gdk_device_get_window_at_position(pointer, NULL, NULL) != win)
    return;
  const gboolean has_sel = d && d->has_selection;
  const gboolean ctrl_shift_held
    = has_sel
      && (mod & (GDK_CONTROL_MASK | GDK_SHIFT_MASK))
           == (GDK_CONTROL_MASK | GDK_SHIFT_MASK);
  const gboolean shift_held
    = has_sel && !ctrl_shift_held
      && (mod & GDK_SHIFT_MASK) != 0;

  // convert device coordinates to preview pipe pixel space
  {
    float pzx, pzy, zs;
    dt_dev_get_pointer_zoom_pos(&darktable.develop->full,
                                (float)dev_x, (float)dev_y,
                                &pzx, &pzy, &zs);
    gui->posx = pzx * wd;
    gui->posy = pzy * ht;
  }

  // draw cursor indicator for click interaction
  if(gui->posx >= 0.0f && gui->posx <= wd
     && gui->posy >= 0.0f && gui->posy <= ht)
  {
    const float r = DT_PIXEL_APPLY_DPI(8.0f) / zoom_scale;
    const float lw = DT_PIXEL_APPLY_DPI(2.0f) / zoom_scale;
    cairo_set_line_width(cr, lw);

    if(ctrl_shift_held)
    {
      // clear mode: draw undo/revert arrow above cursor
      cairo_set_source_rgba(cr, 0.9, 0.9, 0.9, 0.9);
      const float s = r * 0.7f; // icon size
      const float cx = gui->posx;
      const float cy = gui->posy - s * 1.8f; // above cursor

      cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
      cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);

      // arrowhead pointing left
      const float ax = cx - s * 0.8f;
      const float ay = cy - s;
      cairo_move_to(cr, ax + s * 0.65f, ay - s * 0.6f);
      cairo_line_to(cr, ax, ay);
      cairo_line_to(cr, ax + s * 0.65f, ay + s * 0.6f);
      cairo_stroke(cr);

      // horizontal line from arrow tip to half-circle top
      cairo_move_to(cr, ax, ay);
      cairo_line_to(cr, cx, ay);

      // half-circle curving right and down
      cairo_arc(cr, cx, cy, s, -G_PI * 0.5f, G_PI * 0.5f);

      // small horizontal tail at bottom going left
      cairo_line_to(cr, cx - s * 0.5f, cy + s);
      cairo_stroke(cr);
    }
    else
    {
      cairo_set_source_rgba(cr, 0.9, 0.9, 0.9, 0.9);
      // horizontal line (common to both + and -)
      cairo_move_to(cr, gui->posx - r, gui->posy);
      cairo_line_to(cr, gui->posx + r, gui->posy);
      cairo_stroke(cr);
      if(!shift_held)
      {
        // add mode: vertical line to form "+"
        cairo_move_to(cr, gui->posx, gui->posy - r);
        cairo_line_to(cr, gui->posx, gui->posy + r);
        cairo_stroke(cr);
      }
    }
  }

}

static GSList *_object_setup_mouse_actions
  (const struct dt_masks_form_t *const form)
{
  GSList *lm = NULL;
  lm = dt_mouse_action_create_simple(
    lm,
    DT_MOUSE_ACTION_LEFT,
    0,
    _("[OBJECT] select / add foreground point"));
  lm = dt_mouse_action_create_simple(
    lm,
    DT_MOUSE_ACTION_LEFT,
    GDK_SHIFT_MASK,
    _("[OBJECT] add background point"));
  lm = dt_mouse_action_create_simple(
    lm,
    DT_MOUSE_ACTION_LEFT,
    GDK_CONTROL_MASK | GDK_SHIFT_MASK,
    _("[OBJECT] clear selection"));
  lm = dt_mouse_action_create_simple(
    lm,
    DT_MOUSE_ACTION_RIGHT,
    0,
    _("[OBJECT] apply mask"));
  // the trace settings, as _object_events_mouse_scrolled takes them
  if(_commits_paths(form))
  {
    lm = dt_mouse_action_create_simple(
      lm,
      DT_MOUSE_ACTION_SCROLL,
      0,
      _("[OBJECT] change smoothing"));
    lm = dt_mouse_action_create_simple(
      lm,
      DT_MOUSE_ACTION_SCROLL,
      GDK_SHIFT_MASK,
      _("[OBJECT] change cleanup"));
  }
  lm = dt_mouse_action_create_simple(
    lm,
    DT_MOUSE_ACTION_SCROLL,
    GDK_CONTROL_MASK,
    _("[OBJECT] change opacity"));
  return lm;
}

static void _object_set_hint_message(const dt_masks_form_gui_t *const gui,
                                     const dt_masks_form_t *const form,
                                     const int opacity,
                                     char *const restrict msgbuf,
                                     const size_t msgbuf_len)
{
  if(gui->creation)
  {
    const _object_data_t *d = _get_data((dt_masks_form_gui_t *)gui);
    if(!d || d->encode_state != ENCODE_READY)
      return;  // no hints while encoding
    // the right-click line names what a commit produces, and the trace
    // settings are listed only while they apply
    const gboolean editing = _is_edit(form);
    if(d->has_selection && _commits_paths(form))
      g_snprintf(msgbuf,
                 msgbuf_len,
                 _("<b>add</b>: click, <b>subtract</b>: shift+click, "
                   "<b>clear</b>: ctrl+shift+click, "
                   "<b>apply as paths</b>: right-click\n"
                   "<b>smoothing</b>: scroll (%3.2f), "
                   "<b>cleanup</b>: shift+scroll (%d), "
                   "<b>opacity</b>: ctrl+scroll (%d%%)"),
                 dt_conf_get_float(CONF_OBJECT_SMOOTHING_KEY),
                 dt_conf_get_int(CONF_OBJECT_CLEANUP_KEY), opacity);
    else if(editing)
      g_snprintf(msgbuf,
                 msgbuf_len,
                 _("<b>add</b>: click, <b>subtract</b>: shift+click, "
                   "<b>clear</b>: ctrl+shift+click, "
                   "<b>apply</b>: right-click, <b>cancel</b>: esc\n"
                   "<b>opacity</b>: ctrl+scroll (%d%%)"),
                 opacity);
    else if(d->has_selection)
      g_snprintf(msgbuf,
                 msgbuf_len,
                 _("<b>add</b>: click, <b>subtract</b>: shift+click, "
                   "<b>clear</b>: ctrl+shift+click, "
                   "<b>apply</b>: right-click\n"
                   "<b>opacity</b>: ctrl+scroll (%d%%)"),
                 opacity);
    else
      g_snprintf(msgbuf,
                 msgbuf_len,
                 _("<b>select</b>: click on object, "
                   "<b>opacity</b>: ctrl+scroll (%d%%)"),
                 opacity);
  }
}

static gboolean _refresh_properties(gpointer data)
{
  (void)data;
  dt_dev_masks_list_change(darktable.develop);
  return G_SOURCE_REMOVE;
}

static void _object_modify_property(dt_masks_form_t *const form,
                                    const dt_masks_property_t prop,
                                    const float old_val,
                                    const float new_val,
                                    float *sum,
                                    int *count,
                                    float *min,
                                    float *max)
{
  dt_masks_form_gui_t *gui = darktable.develop->form_gui;
  _object_data_t *d = gui ? _get_data(gui) : NULL;

  if(!gui || !gui->creation) return;

  // an edit gets neither the switch nor the trace settings. *count left at
  // 0 hides a property's widget (libs/masks.c)
  const gboolean editing = _is_edit(form);
  const gboolean traced = _commits_paths(form);

  switch(prop)
  {
    case DT_MASKS_PROPERTY_SIZE:
      break; // no size slider for click-based interaction
    case DT_MASKS_PROPERTY_CLEANUP:
    {
      if(!traced) break;
      const int old = dt_conf_get_int(CONF_OBJECT_CLEANUP_KEY);
      const int cleanup = CLAMP(old + (int)(new_val - old_val), 0, 100);
      dt_conf_set_int(CONF_OBJECT_CLEANUP_KEY, cleanup);
      if(d && cleanup != old) _schedule_outline(d);
      *sum += cleanup;
      ++*count;
      break;
    }
    case DT_MASKS_PROPERTY_SMOOTHING:
    {
      if(!traced) break;
      const float old = dt_conf_get_float(CONF_OBJECT_SMOOTHING_KEY);
      const float smoothing = CLAMP(old + (new_val - old_val), 0.0f, 1.3f);
      dt_conf_set_float(CONF_OBJECT_SMOOTHING_KEY, smoothing);
      if(d && smoothing != old) _schedule_outline(d);
      *sum += smoothing;
      ++*count;
      break;
    }
    case DT_MASKS_PROPERTY_FEATHER:
    {
      if(!traced) break;
      const float ratio = (!old_val || !new_val) ? 1.0f : new_val / old_val;
      float feather = dt_conf_get_float(CONF_OBJECT_FEATHER_KEY);
      if(feather < 0.0005f && ratio > 1.0f)
        feather = 0.001f; // bootstrap from zero on increase
      feather = CLAMP(feather * ratio, 0.0005f, 1.0f);
      dt_conf_set_float(CONF_OBJECT_FEATHER_KEY, feather);
      *sum += feather + feather; // both borders (same as path)
      *max = fminf(*max, 1.0f / feather);
      *min = fmaxf(*min, 0.0005f / feather);
      *count += 2; // both borders (same as path)
      break;
    }
    case DT_MASKS_PROPERTY_REFINE:
    {
      // toggle applies on the next decoder run, not immediately
      if(new_val != old_val)
        dt_conf_set_bool(CONF_OBJECT_REFINE_BOUNDARY_KEY, new_val > 0.5f);
      const gboolean enabled
        = dt_conf_get_bool(CONF_OBJECT_REFINE_BOUNDARY_KEY);
      *sum += enabled ? 1.0f : 0.0f;
      ++*count;
      break;
    }
    case DT_MASKS_PROPERTY_VECTORIZE:
    {
      if(editing) break;
      // locked on without sidecar files: the collapsed range has the mask
      // manager grey the switch out (libs/masks.c)
      const gboolean locked = !dt_dtdata_enabled();
      if(locked) *max = *min;
      if(new_val != old_val && !locked)
      {
        dt_conf_set_bool(CONF_OBJECT_VECTORIZE_KEY, new_val > 0.5f);
        if(d) _schedule_outline(d);
        // the trace settings show or hide with the switch. not from here:
        // this runs inside the mask manager's update of the switch itself
        g_idle_add(_refresh_properties, NULL);
      }
      *sum += _as_paths() ? 1.0f : 0.0f;
      ++*count;
      break;
    }
    default:;
  }
}

/* a click on a committed object's icon reopens it: the form becomes the one
   being created, and the tool resumes from its stored mask once the image
   is encoded (_resume_edit) */
static int _start_edit(dt_iop_module_t *module, dt_masks_form_t *form)
{
  if(!dt_masks_object_available())
  {
    dt_control_log(_("AI model is not available. Check preferences > AI"));
    return 1;
  }
  // an edit is stored as pixels, which need the sidecar
  if(!dt_dtdata_enabled())
  {
    dt_control_log(_("sidecar files are disabled, the object cannot be edited"));
    return 1;
  }
  dt_masks_change_form_gui(form);
  // module is whichever one has focus. the commit files history under it,
  // enables it and returns the canvas to it, so it takes the edit only if
  // the object is one of its masks
  darktable.develop->form_gui->creation_module
    = dt_masks_is_in_module(form->formid, module) ? module : NULL;
  dt_control_queue_redraw_center();
  return 1;
}

#endif // HAVE_AI

/* --- one table for both lives of the form: the handlers below route to
   the AI tool while it is being created and to the stored mask once it
   is committed. without AI only the second exists, so a committed object
   still renders, shows its icon and selects, but none can be made --- */

static void _object_set_form_name(dt_masks_form_t *const form,
                                  const size_t nb)
{
  snprintf(form->name, sizeof(form->name), _("ai object #%d"), (int)nb);
}

static int _object_mouse_moved(dt_iop_module_t *module,
                               const float pzx,
                               const float pzy,
                               const double pressure,
                               const int which,
                               const float zoom_scale,
                               dt_masks_form_t *form,
                               const dt_imgid_t parentid,
                               dt_masks_form_gui_t *gui,
                               const int index)
{
  if(!gui) return 0;
#ifdef HAVE_AI
  if(gui->creation)
    return _object_events_mouse_moved(module, pzx, pzy, pressure, which, zoom_scale,
                                      form, parentid, gui, index);
#endif
  return _raster_mouse_moved(module, pzx, pzy, pressure, which, zoom_scale,
                             form, parentid, gui, index);
}

static int _object_mouse_scrolled(dt_iop_module_t *module,
                                  const float pzx,
                                  const float pzy,
                                  const gboolean up,
                                  const uint32_t state,
                                  dt_masks_form_t *form,
                                  const dt_imgid_t parentid,
                                  dt_masks_form_gui_t *gui,
                                  const int index)
{
#ifdef HAVE_AI
  if(gui && gui->creation)
    return _object_events_mouse_scrolled(module, pzx, pzy, up, state,
                                         form, parentid, gui, index);
#endif
  return 0;
}

static int _object_button_pressed(dt_iop_module_t *module,
                                  const float pzx,
                                  const float pzy,
                                  const double pressure,
                                  const int which,
                                  const int type,
                                  const uint32_t state,
                                  dt_masks_form_t *form,
                                  const dt_imgid_t parentid,
                                  dt_masks_form_gui_t *gui,
                                  const int index)
{
#ifdef HAVE_AI
  if(gui && gui->creation)
    return _object_events_button_pressed(module, pzx, pzy, pressure, which, type, state,
                                         form, parentid, gui, index);
  if(gui && which == 1 && type == GDK_BUTTON_PRESS && gui->form_selected
     && gui->point_selected == 0 && _raster_ref(form))
    return _start_edit(module, form);
#endif
  return 0;
}

static int _object_button_released(dt_iop_module_t *module,
                                   const float pzx,
                                   const float pzy,
                                   const int which,
                                   const uint32_t state,
                                   dt_masks_form_t *form,
                                   const dt_imgid_t parentid,
                                   dt_masks_form_gui_t *gui,
                                   const int index)
{
#ifdef HAVE_AI
  if(gui && gui->creation)
    return _object_events_button_released(module, pzx, pzy, which, state,
                                          form, parentid, gui, index);
#endif
  return 0;
}

static void _object_post_expose(cairo_t *cr,
                                const float zoom_scale,
                                dt_masks_form_gui_t *gui,
                                const int index,
                                const int num_points)
{
  if(!gui) return;
#ifdef HAVE_AI
  if(gui->creation)
  {
    _object_events_post_expose(cr, zoom_scale, gui, index, num_points);
    return;
  }
#endif
  _raster_post_expose(cr, zoom_scale, gui, index, num_points);
}

const dt_masks_functions_t dt_masks_functions_object = {
  .point_struct_size = sizeof(struct dt_masks_point_object_t),
  .sanitize_config = NULL,
#ifdef HAVE_AI
  .setup_mouse_actions = _object_setup_mouse_actions,
  .set_hint_message = _object_set_hint_message,
  .modify_property = _object_modify_property,
#else
  .setup_mouse_actions = NULL,
  .set_hint_message = NULL,
  .modify_property = NULL,
#endif
  .set_form_name = _object_set_form_name,
  .duplicate_points = _raster_duplicate_points,
  .initial_source_pos = NULL,
  .get_distance = _raster_get_distance,
  .get_points = NULL,
  .get_points_border = _raster_get_points_border,
  // the object never joins a clone group, the only user of these
  .get_mask = NULL,
  .get_mask_roi = _raster_get_mask_roi,
  .get_area = NULL,
  .get_source_area = NULL,
  .mouse_moved = _object_mouse_moved,
  .mouse_scrolled = _object_mouse_scrolled,
  .button_pressed = _object_button_pressed,
  .button_released = _object_button_released,
  .post_expose = _object_post_expose
};

#ifdef HAVE_AI
gboolean dt_masks_object_available(void)
{
  if(!dt_ai_registry_is_enabled())
    return FALSE;
  char *model_id = dt_ai_models_get_active_for_task("mask");
  dt_ai_model_t *model = dt_ai_models_get_by_id(model_id);
  g_free(model_id);
  const gboolean available = model && model->status == DT_AI_MODEL_DOWNLOADED;
  dt_ai_model_free(model);
  return available;
}
#endif // HAVE_AI

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
