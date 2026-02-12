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

#include "common/ai_models.h"
#include "common/colorspaces.h"
#include "common/debug.h"
#include "common/mipmap_cache.h"
#include "control/conf.h"
#include "control/control.h"
#include "gui/gtk.h"
#include "ai/segmentation.h"
#include "develop/blend.h"
#include "develop/imageop.h"
#include "develop/masks.h"
#include "develop/openmp_maths.h"
#include "develop/pixelpipe_hb.h"
#include "imageio/imageio_common.h"
#include "common/ras2vect.h"

#include <math.h>
#include <string.h>

#define CONF_OBJECT_MODEL_KEY "plugins/darkroom/masks/object/model"
#define CONF_OBJECT_MODE_KEY "plugins/darkroom/masks/object/mode"
#define CONF_OBJECT_AUTO_GRID_KEY "plugins/darkroom/masks/object/auto_grid"
#define DEFAULT_OBJECT_MODEL_ID "mask-light-hq-sam"

// Target resolution for SAM encoding (longest side in pixels).
// Higher than preview pipe for better segmentation quality.
#define SAM_ENCODE_TARGET 2048

// Maximum number of auto-segmented masks to keep after NMS
#define AUTO_MAX_MASKS 256

// --- Per-session segmentation state (stored in gui->scratchpad) ---

typedef enum _encode_state_t
{
  ENCODE_ERROR = -1,
  ENCODE_IDLE = 0,
  ENCODE_MSG_SHOWN = 1, // busy message queued, waiting for next expose
  ENCODE_READY = 2,     // encoding complete, results available
  ENCODE_RUNNING = 3,   // background thread in progress
} _encode_state_t;

typedef enum _auto_state_t
{
  AUTO_ERROR = -1,
  AUTO_IDLE = 0,
  AUTO_RUNNING = 1,
  AUTO_READY = 2,
} _auto_state_t;

// RLE-encoded binary mask (pairs of start_offset, length)
typedef struct _auto_mask_t
{
  int *runs;       // pairs of (start_offset, length), n_runs * 2 ints
  int n_runs;
  int area;        // total foreground pixels
  float iou_score;
} _auto_mask_t;

// Auto-segmentation state (allocated when mode=auto, after encoding)
typedef struct _auto_data_t
{
  _auto_mask_t masks[AUTO_MAX_MASKS];
  int n_masks;
  int16_t *label_map;  // pixel -> mask_id (-1 = none)
  int lw, lh;          // label map dimensions (= encoding dims)
  gboolean selected[AUTO_MAX_MASKS];
  int hover_mask;      // -1 = none (atomic access)
  int auto_state;      // _auto_state_t (atomic access)
  GThread *auto_thread;
  int grid_total;
  int grid_done;       // progress counter (atomic)
  int cancel;          // cancellation flag (atomic)
} _auto_data_t;

// Minimum drag distance (preview pipe pixels) to distinguish click from box drag
#define BOX_DRAG_THRESHOLD 5.0f

typedef struct _object_data_t
{
  dt_ai_environment_t *env; // AI environment for model registry
  dt_seg_context_t *seg;    // SAM context (encoder+decoder)
  float *mask;              // current mask buffer (preview pipe size)
  int mask_w, mask_h;       // mask dimensions
  gboolean model_loaded;    // whether the model was loaded
  int encode_state;         // uses _encode_state_t values (atomic access)
  dt_imgid_t encoded_imgid; // image ID that was encoded
  int encode_w, encode_h;   // encoding resolution (for coordinate mapping)
  guint modifier_poll_id;   // timer to detect shift key changes
  GThread *encode_thread;   // background encoding thread
  gboolean busy;            // TRUE if dt_control_busy_enter() was called
  _auto_data_t *autodata;   // auto-segmentation data (NULL in prompt mode)
  gboolean dragging;        // TRUE between press and release during potential box drag
  float drag_start_x;       // press position (preview pipe pixel space)
  float drag_start_y;
  float drag_end_x;         // current drag position (updated in mouse_moved)
  float drag_end_y;
} _object_data_t;

static _object_data_t *_get_data(dt_masks_form_gui_t *gui)
{
  return (gui && gui->scratchpad) ? (_object_data_t *)gui->scratchpad : NULL;
}

static gboolean _is_auto_mode(void)
{
  gchar *mode = dt_conf_get_string(CONF_OBJECT_MODE_KEY);
  const gboolean is_auto = mode && g_strcmp0(mode, "auto") == 0;
  g_free(mode);
  return is_auto;
}

// --- RLE utilities for auto-segmentation ---

// Convert a float mask to RLE. Returns number of runs (0 if empty).
static int _mask_to_rle(const float *mask, int w, int h, float threshold,
                         int **out_runs, int *out_area)
{
  const int npix = w * h;
  int n_runs = 0;
  int area = 0;
  gboolean in_run = FALSE;

  for(int i = 0; i < npix; i++)
  {
    if(mask[i] > threshold)
    {
      if(!in_run)
      {
        n_runs++;
        in_run = TRUE;
      }
      area++;
    }
    else
      in_run = FALSE;
  }

  if(n_runs == 0)
  {
    *out_runs = NULL;
    *out_area = 0;
    return 0;
  }

  int *runs = g_new(int, n_runs * 2);
  int ri = 0;
  in_run = FALSE;

  for(int i = 0; i < npix; i++)
  {
    if(mask[i] > threshold)
    {
      if(!in_run)
      {
        runs[ri * 2] = i;
        runs[ri * 2 + 1] = 1;
        in_run = TRUE;
      }
      else
        runs[ri * 2 + 1]++;
    }
    else
    {
      if(in_run)
      {
        ri++;
        in_run = FALSE;
      }
    }
  }

  *out_runs = runs;
  *out_area = area;
  return n_runs;
}

// Compute IoU between two RLE masks (runs must be sorted by start offset)
static float _rle_iou(const int *runs_a, int n_a, int area_a,
                       const int *runs_b, int n_b, int area_b)
{
  if(area_a == 0 || area_b == 0)
    return 0.0f;

  int intersection = 0;
  int ia = 0, ib = 0;

  while(ia < n_a && ib < n_b)
  {
    const int a_start = runs_a[ia * 2];
    const int a_end = a_start + runs_a[ia * 2 + 1];
    const int b_start = runs_b[ib * 2];
    const int b_end = b_start + runs_b[ib * 2 + 1];

    const int ov_start = MAX(a_start, b_start);
    const int ov_end = MIN(a_end, b_end);
    if(ov_start < ov_end)
      intersection += ov_end - ov_start;

    if(a_end <= b_end)
      ia++;
    else
      ib++;
  }

  const int union_area = area_a + area_b - intersection;
  return union_area > 0 ? (float)intersection / (float)union_area : 0.0f;
}

// Free auto-segmentation data
static void _auto_data_free(_auto_data_t *ad)
{
  if(!ad)
    return;
  if(ad->auto_thread)
    g_thread_join(ad->auto_thread);
  for(int i = 0; i < ad->n_masks; i++)
    g_free(ad->masks[i].runs);
  g_free(ad->label_map);
  g_free(ad);
}

// Free all resources in _object_data_t (must be called after thread has joined)
static void _destroy_data(_object_data_t *d)
{
  if(!d)
    return;
  if(d->busy)
    dt_control_busy_leave();
  if(d->modifier_poll_id)
    g_source_remove(d->modifier_poll_id);
  if(d->encode_thread)
    g_thread_join(d->encode_thread);
  _auto_data_free(d->autodata);
  if(d->seg)
    dt_seg_free(d->seg);
  if(d->env)
    dt_ai_env_destroy(d->env);
  g_free(d->mask);
  g_free(d);
}

// Idle callback for deferred cleanup when background thread was still running
static gboolean _deferred_cleanup(gpointer data)
{
  _object_data_t *d = data;
  const int state = g_atomic_int_get(&d->encode_state);
  if(state == ENCODE_RUNNING)
    return G_SOURCE_CONTINUE;
  if(d->autodata && g_atomic_int_get(&d->autodata->auto_state) == AUTO_RUNNING)
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

  // Signal auto thread to cancel
  if(d->autodata)
    g_atomic_int_set(&d->autodata->cancel, 1);

  const int state = g_atomic_int_get(&d->encode_state);
  const gboolean auto_running
    = d->autodata && g_atomic_int_get(&d->autodata->auto_state) == AUTO_RUNNING;
  if(state == ENCODE_RUNNING || auto_running)
  {
    // Thread still running — defer cleanup so we don't block the UI
    g_timeout_add(200, _deferred_cleanup, d);
    return;
  }
  _destroy_data(d);
}

// Data passed to the background encoding thread
typedef struct _encode_thread_data_t
{
  _object_data_t *d;
  dt_imgid_t imgid; // image to encode (thread renders via export pipe)
} _encode_thread_data_t;

// Background thread: loads model, renders image via export pipe, and encodes.
// Does ZERO GLib/GTK calls — only computation + atomic state set.
// The poll timer on the main thread detects completion.
static gpointer _encode_thread_func(gpointer data)
{
  _encode_thread_data_t *td = data;
  _object_data_t *d = td->d;
  const dt_imgid_t imgid = td->imgid;
  g_free(td);

  // Load model if needed
  if(!d->model_loaded)
  {
    if(!d->env)
      d->env = dt_ai_env_init(NULL);

    char *model_id = dt_conf_get_string(CONF_OBJECT_MODEL_KEY);
    d->seg = dt_seg_load(d->env, model_id, DT_AI_PROVIDER_AUTO);
    g_free(model_id);

    if(!d->seg)
    {
      g_atomic_int_set(&d->encode_state, ENCODE_ERROR);
      return NULL;
    }
    d->model_loaded = TRUE;
  }

  // Render image at high resolution via temporary export pipeline
  dt_develop_t dev;
  dt_dev_init(&dev, FALSE);
  dt_dev_load_image(&dev, imgid);

  dt_mipmap_buffer_t buf;
  dt_mipmap_cache_get(&buf, imgid, DT_MIPMAP_FULL, DT_MIPMAP_BLOCKING, 'r');

  if(!buf.buf || !buf.width || !buf.height)
  {
    dt_print(DT_DEBUG_AI, "[object mask] Failed to get image buffer for encoding");
    dt_dev_cleanup(&dev);
    g_atomic_int_set(&d->encode_state, ENCODE_ERROR);
    return NULL;
  }

  const int wd = dev.image_storage.width;
  const int ht = dev.image_storage.height;

  dt_dev_pixelpipe_t pipe;
  if(!dt_dev_pixelpipe_init_export(&pipe, wd, ht, IMAGEIO_RGB | IMAGEIO_INT8, FALSE))
  {
    dt_print(DT_DEBUG_AI, "[object mask] Failed to init export pipe for encoding");
    dt_mipmap_cache_release(&buf);
    dt_dev_cleanup(&dev);
    g_atomic_int_set(&d->encode_state, ENCODE_ERROR);
    return NULL;
  }

  dt_dev_pixelpipe_set_icc(&pipe, DT_COLORSPACE_SRGB, NULL, DT_INTENT_PERCEPTUAL);
  dt_dev_pixelpipe_set_input(&pipe, &dev, (float *)buf.buf,
                             buf.width, buf.height, buf.iscale);
  dt_dev_pixelpipe_create_nodes(&pipe, &dev);
  dt_dev_pixelpipe_synch_all(&pipe, &dev);

  dt_dev_pixelpipe_get_dimensions(&pipe, &dev, pipe.iwidth, pipe.iheight,
                                  &pipe.processed_width, &pipe.processed_height);

  const double scale = fmin((double)SAM_ENCODE_TARGET / (double)pipe.processed_width,
                            (double)SAM_ENCODE_TARGET / (double)pipe.processed_height);
  const double final_scale = fmin(scale, 1.0); // don't upscale
  const int out_w = (int)(final_scale * pipe.processed_width);
  const int out_h = (int)(final_scale * pipe.processed_height);

  dt_print(DT_DEBUG_AI, "[object mask] Rendering %dx%d (scale=%.3f) for encoding...",
           out_w, out_h, final_scale);

  dt_dev_pixelpipe_process_no_gamma(&pipe, &dev, 0, 0, out_w, out_h, final_scale);

  // backbuf is float RGBA after process_no_gamma — convert to uint8 RGB for SAM
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
    dt_print(DT_DEBUG_AI, "[object mask] Failed to render image for encoding");
    g_atomic_int_set(&d->encode_state, ENCODE_ERROR);
    return NULL;
  }

  // Store encoding dimensions for coordinate mapping
  d->encode_w = out_w;
  d->encode_h = out_h;

  // Encode the image
  gboolean ok = dt_seg_encode_image(d->seg, rgb, out_w, out_h);

  // If accelerated encoding failed, fall back to CPU
  if(!ok)
  {
    dt_print(DT_DEBUG_AI, "[object mask] Encoding failed, retrying with CPU provider");
    dt_seg_free(d->seg);
    char *model_id = dt_conf_get_string(CONF_OBJECT_MODEL_KEY);
    d->seg = dt_seg_load(d->env, model_id, DT_AI_PROVIDER_CPU);
    g_free(model_id);

    if(d->seg)
      ok = dt_seg_encode_image(d->seg, rgb, out_w, out_h);
    else
      d->model_loaded = FALSE;
  }

  g_free(rgb);

  g_atomic_int_set(&d->encode_state, ok ? ENCODE_READY : ENCODE_ERROR);
  return NULL;
}

// Background thread: generate auto-segmentation masks from dense grid prompts
static gpointer _auto_thread_func(gpointer data)
{
  _object_data_t *d = data;
  _auto_data_t *ad = d->autodata;

  const int grid_n = CLAMP(dt_conf_get_int(CONF_OBJECT_AUTO_GRID_KEY), 8, 64);

  int enc_w, enc_h;
  dt_seg_get_encoded_dims(d->seg, &enc_w, &enc_h);

  ad->lw = enc_w;
  ad->lh = enc_h;
  ad->grid_total = grid_n * grid_n;
  g_atomic_int_set(&ad->grid_done, 0);

  // Filtering thresholds (matching SAM SamAutomaticMaskGenerator defaults)
  const float pred_iou_thresh = 0.88f;
  const float stability_thresh = 0.95f;
  // Stability score thresholds in sigmoid space: sigmoid(-1)=0.269, sigmoid(1)=0.731
  const float stab_lo = 0.269f;
  const float stab_hi = 0.731f;
  const int total_pixels = enc_w * enc_h;
  const int min_area = MAX(100, total_pixels / 1000); // 0.1% of image
  const int max_area = (int)(total_pixels * 0.95f);   // 95% of image

  // Phase 1: Decode all grid points, collect candidate masks
  const int max_candidates = grid_n * grid_n * 8;
  _auto_mask_t *candidates = g_new0(_auto_mask_t, max_candidates);
  int n_candidates = 0;

  for(int gy = 0; gy < grid_n && !g_atomic_int_get(&ad->cancel); gy++)
  {
    for(int gx = 0; gx < grid_n && !g_atomic_int_get(&ad->cancel); gx++)
    {
      const float px = ((float)gx + 0.5f) / (float)grid_n * (float)enc_w;
      const float py = ((float)gy + 0.5f) / (float)grid_n * (float)enc_h;

      dt_seg_point_t point = {.x = px, .y = py, .label = 1};

      float *masks = NULL;
      float *ious = NULL;
      int n_masks = 0, mw = 0, mh = 0;

      if(!dt_seg_compute_mask_raw(d->seg, &point, &masks, &ious, &n_masks, &mw, &mh))
      {
        g_atomic_int_add(&ad->grid_done, 1);
        continue;
      }

      const size_t per_mask = (size_t)mw * mh;

      for(int m = 0; m < n_masks && n_candidates < max_candidates; m++)
      {
        // Filter 1: predicted IoU quality
        if(ious[m] < pred_iou_thresh)
          continue;

        // Filter 2: stability score — ratio of areas at two sigmoid thresholds.
        // Stable masks have sharp boundaries: area_wide ≈ area_tight, score ≈ 1.0
        const float *mdata = masks + m * per_mask;
        int count_lo = 0, count_hi = 0;
        for(size_t p = 0; p < per_mask; p++)
        {
          if(mdata[p] > stab_lo) count_lo++;
          if(mdata[p] > stab_hi) count_hi++;
        }
        const float stability = count_hi > 0
          ? (float)count_hi / (float)count_lo : 0.0f;
        if(stability < stability_thresh)
          continue;

        int *runs = NULL;
        int area = 0;
        const int n_runs
          = _mask_to_rle(mdata, mw, mh, 0.5f, &runs, &area);

        // Filter 3: area bounds
        if(n_runs > 0 && area >= min_area && area <= max_area)
        {
          candidates[n_candidates].runs = runs;
          candidates[n_candidates].n_runs = n_runs;
          candidates[n_candidates].area = area;
          candidates[n_candidates].iou_score = ious[m];
          n_candidates++;
        }
        else
          g_free(runs);
      }

      g_free(masks);
      g_free(ious);
      g_atomic_int_add(&ad->grid_done, 1);
    }
  }

  if(g_atomic_int_get(&ad->cancel))
  {
    for(int i = 0; i < n_candidates; i++)
      g_free(candidates[i].runs);
    g_free(candidates);
    g_atomic_int_set(&ad->auto_state, AUTO_ERROR);
    return NULL;
  }

  // Phase 2: NMS — sort by area descending, accept if IoU < 0.7 with all accepted
  for(int i = 1; i < n_candidates; i++)
  {
    _auto_mask_t tmp = candidates[i];
    int j = i - 1;
    while(j >= 0 && candidates[j].area < tmp.area)
    {
      candidates[j + 1] = candidates[j];
      j--;
    }
    candidates[j + 1] = tmp;
  }

  ad->n_masks = 0;
  for(int i = 0; i < n_candidates && ad->n_masks < AUTO_MAX_MASKS; i++)
  {
    gboolean keep = TRUE;
    for(int j = 0; j < ad->n_masks; j++)
    {
      const float iou
        = _rle_iou(candidates[i].runs, candidates[i].n_runs, candidates[i].area,
                    ad->masks[j].runs, ad->masks[j].n_runs, ad->masks[j].area);
      if(iou > 0.7f)
      {
        keep = FALSE;
        break;
      }
    }

    if(keep)
    {
      ad->masks[ad->n_masks] = candidates[i];
      candidates[i].runs = NULL; // ownership transferred
      ad->n_masks++;
    }
  }

  for(int i = 0; i < n_candidates; i++)
    g_free(candidates[i].runs);
  g_free(candidates);

  // Phase 3: Build label map — smallest masks paint last (most specific wins hover)
  ad->label_map = g_try_malloc((size_t)enc_w * enc_h * sizeof(int16_t));
  if(!ad->label_map)
  {
    g_atomic_int_set(&ad->auto_state, AUTO_ERROR);
    return NULL;
  }
  memset(ad->label_map, 0xFF, (size_t)enc_w * enc_h * sizeof(int16_t)); // -1

  // Build index sorted by area ascending
  int sort_idx[AUTO_MAX_MASKS];
  for(int i = 0; i < ad->n_masks; i++)
    sort_idx[i] = i;

  for(int i = 1; i < ad->n_masks; i++)
  {
    const int tmp = sort_idx[i];
    int j = i - 1;
    while(j >= 0 && ad->masks[sort_idx[j]].area > ad->masks[tmp].area)
    {
      sort_idx[j + 1] = sort_idx[j];
      j--;
    }
    sort_idx[j + 1] = tmp;
  }

  // Paint largest first, smallest last (smallest overwrites → hover picks most specific)
  for(int si = ad->n_masks - 1; si >= 0; si--)
  {
    const int mi = sort_idx[si];
    const _auto_mask_t *m = &ad->masks[mi];
    for(int r = 0; r < m->n_runs; r++)
    {
      const int start = m->runs[r * 2];
      const int len = m->runs[r * 2 + 1];
      const int end = MIN(start + len, enc_w * enc_h);
      for(int p = start; p < end; p++)
        ad->label_map[p] = (int16_t)mi;
    }
  }

  dt_print(DT_DEBUG_AI,
           "[object mask] Auto-segmentation complete: %d masks from %d grid points",
           ad->n_masks, grid_n * grid_n);

  g_atomic_int_set(&ad->auto_state, AUTO_READY);
  return NULL;
}

// Keep only the connected component containing the seed pixel (seed_x, seed_y).
// If the seed is outside any foreground region, keep the largest component instead.
// Operates in-place: non-selected foreground pixels are zeroed.
static void _keep_seed_component(float *mask, int w, int h, float threshold,
                                  int seed_x, int seed_y)
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

  // Prefer component containing the seed point; fall back to largest
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

// Run the decoder with accumulated points and update the cached mask
static void _run_decoder(dt_masks_form_gui_t *gui)
{
  _object_data_t *d = _get_data(gui);
  if(!d || !d->seg || !dt_seg_is_encoded(d->seg))
    return;
  if(gui->guipoints_count <= 0)
    return;

  const float *gp = dt_masks_dynbuf_buffer(gui->guipoints);
  const float *gpp = dt_masks_dynbuf_buffer(gui->guipoints_payload);

  // Points are stored in preview pipe pixel space — scale to encoding space
  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);
  const float sx = (wd > 0) ? (float)d->encode_w / wd : 1.0f;
  const float sy = (ht > 0) ? (float)d->encode_h / ht : 1.0f;

  dt_seg_point_t *points = g_new(dt_seg_point_t, gui->guipoints_count);
  for(int i = 0; i < gui->guipoints_count; i++)
  {
    points[i].x = gp[i * 2 + 0] * sx;
    points[i].y = gp[i * 2 + 1] * sy;
    points[i].label = (int)gpp[i];
  }

  // Find seed point for connected component filter:
  // use last foreground point, or center of last box
  int seed_x = -1, seed_y = -1;
  for(int i = gui->guipoints_count - 1; i >= 0; i--)
  {
    if(points[i].label == 1)
    {
      seed_x = (int)points[i].x;
      seed_y = (int)points[i].y;
      break;
    }
    else if(points[i].label == 3 && i > 0 && points[i - 1].label == 2)
    {
      // Box: use center of the two corners
      seed_x = (int)((points[i - 1].x + points[i].x) * 0.5f);
      seed_y = (int)((points[i - 1].y + points[i].y) * 0.5f);
      break;
    }
  }

  int mw, mh;
  float *mask = dt_seg_compute_mask(d->seg, points, gui->guipoints_count, &mw, &mh);
  g_free(points);

  if(mask)
  {
    // Remove disconnected blobs: keep only the component at the seed point
    seed_x = CLAMP(seed_x, 0, mw - 1);
    seed_y = CLAMP(seed_y, 0, mh - 1);
    _keep_seed_component(mask, mw, mh, 0.5f, seed_x, seed_y);

    g_free(d->mask);
    d->mask = mask;
    d->mask_w = mw;
    d->mask_h = mh;
  }
}

// Finalize: vectorize the mask and register as a group of path forms
static void
_finalize_mask(dt_iop_module_t *module, dt_masks_form_t *form, dt_masks_form_gui_t *gui)
{
  _object_data_t *d = _get_data(gui);
  if(!d || !d->mask)
    return;

  // Invert mask for potrace (potrace traces dark regions)
  // Our mask: 1.0 = foreground; potrace expects: 0.0 = foreground
  const size_t n = (size_t)d->mask_w * d->mask_h;
  float *inv_mask = g_try_malloc(n * sizeof(float));
  if(!inv_mask)
    return;

  for(size_t i = 0; i < n; i++)
    inv_mask[i] = 1.0f - d->mask[i];

  // Pass NULL for image: the AI mask lives in preview backbuf pixel space.
  // We backtransform through the pipeline to input image coords below.
  const int cleanup = dt_conf_get_int("plugins/darkroom/masks/object/cleanup");
  const float smoothing = dt_conf_get_float("plugins/darkroom/masks/object/smoothing");

  GList *signs = NULL;
  GList *forms
    = ras2forms(inv_mask, d->mask_w, d->mask_h, NULL, cleanup, (double)smoothing, &signs);
  g_free(inv_mask);

  // darktable mask coordinates are stored in input-image-normalized space:
  //   coord = backtransform(backbuf_pixel) / iwidth
  // This undoes all geometric pipeline transforms (crop, rotation, lens, etc.)
  // so that the mask can be applied at any point in the pipeline.
  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);

  // Vectorized coordinates are in mask space (encoding resolution).
  // dt_dev_distort_backtransform expects preview pipe pixel space.
  const float msx = (d->mask_w > 0) ? wd / (float)d->mask_w : 1.0f;
  const float msy = (d->mask_h > 0) ? ht / (float)d->mask_h : 1.0f;

  for(GList *l = forms; l; l = g_list_next(l))
  {
    dt_masks_form_t *f = l->data;
    const int npts = g_list_length(f->points);
    if(npts == 0)
      continue;

    // Collect all coordinates into a flat array for batch backtransform.
    // Each path point has 3 coordinate pairs: corner, ctrl1, ctrl2.
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

    // Scale from mask space (encoding resolution) to preview pipe space
    for(int j = 0; j < npts * 6; j += 2)
    {
      pts[j + 0] *= msx;
      pts[j + 1] *= msy;
    }

    dt_dev_distort_backtransform(darktable.develop, pts, npts * 3);

    // Write back and normalize by input image dimensions
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
    g_list_free(signs);
    dt_control_log(_("no mask extracted from AI segmentation"));
    return;
  }

  // Always wrap paths in a group — holes use difference mode

  // Count existing AI object groups/paths for numbering
  dt_develop_t *dev = darktable.develop;
  guint grp_nb = 0;
  guint path_nb = 0;
  for(GList *l = dev->forms; l; l = g_list_next(l))
  {
    const dt_masks_form_t *f = l->data;
    if(strncmp(f->name, "ai object group", 15) == 0)
      grp_nb++;
    if(strncmp(f->name, "ai object #", 11) == 0)
      path_nb++;
  }
  grp_nb++;
  path_nb++;

  // Name each path form
  for(GList *l = forms; l; l = g_list_next(l))
  {
    dt_masks_form_t *f = l->data;
    snprintf(f->name, sizeof(f->name), "ai object #%d", (int)path_nb++);
  }

  dt_masks_form_t *grp = dt_masks_create(DT_MASKS_GROUP);
  snprintf(grp->name, sizeof(grp->name), "ai object group #%d", (int)grp_nb);

  // Register all path forms so they exist in dev->forms
  for(GList *l = forms; l; l = g_list_next(l))
  {
    dt_masks_form_t *f = l->data;
    dev->forms = g_list_append(dev->forms, f);
  }

  // Add each path to the group; holes get difference mode
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

  // Register the group
  dev->forms = g_list_append(dev->forms, grp);
  dt_dev_add_masks_history_item(dev, NULL, TRUE);

  g_list_free(signs);

  dt_print(DT_DEBUG_AI, "[object mask] created %d paths", nbform);
}

// Finalize auto mode: union selected RLE masks into float mask and vectorize
static void _finalize_auto_mask(dt_iop_module_t *module, dt_masks_form_t *form,
                                 dt_masks_form_gui_t *gui)
{
  _object_data_t *d = _get_data(gui);
  if(!d || !d->autodata)
    return;

  _auto_data_t *ad = d->autodata;
  if(ad->n_masks == 0 || ad->lw <= 0 || ad->lh <= 0)
    return;

  // Check if any mask is selected
  gboolean any_selected = FALSE;
  for(int i = 0; i < ad->n_masks; i++)
  {
    if(ad->selected[i])
    {
      any_selected = TRUE;
      break;
    }
  }
  if(!any_selected)
    return;

  // Build union float mask from selected RLE masks
  const size_t npix = (size_t)ad->lw * ad->lh;
  float *mask = g_try_malloc0(npix * sizeof(float));
  if(!mask)
    return;

  for(int i = 0; i < ad->n_masks; i++)
  {
    if(!ad->selected[i])
      continue;
    const _auto_mask_t *m = &ad->masks[i];
    for(int r = 0; r < m->n_runs; r++)
    {
      const int start = m->runs[r * 2];
      const int len = m->runs[r * 2 + 1];
      const int end = MIN(start + len, (int)npix);
      for(int p = start; p < end; p++)
        mask[p] = 1.0f;
    }
  }

  // Store in d->mask so _finalize_mask can use it
  g_free(d->mask);
  d->mask = mask;
  d->mask_w = ad->lw;
  d->mask_h = ad->lh;

  _finalize_mask(module, form, gui);
}

// --- Mask Event Handlers ---

static void _object_get_distance(
  const float x,
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
  (void)x;
  (void)y;
  (void)as;
  (void)gui;
  (void)index;
  (void)num_points;
  (void)inside;
  (void)inside_border;
  (void)near;
  (void)inside_source;
  (void)dist;
}

static int _object_events_mouse_scrolled(
  dt_iop_module_t *module,
  const float pzx,
  const float pzy,
  const gboolean up,
  const uint32_t state,
  dt_masks_form_t *form,
  const dt_imgid_t parentid,
  dt_masks_form_gui_t *gui,
  const int index)
{
  if(dt_modifier_is(state, GDK_CONTROL_MASK))
  {
    dt_masks_form_change_opacity(form, parentid, up ? 0.05f : -0.05f);
    return 1;
  }
  return 0;
}

// Clear accumulated points, mask preview, and iterative refinement state
static void _clear_selection(dt_masks_form_gui_t *gui)
{
  _object_data_t *d = _get_data(gui);
  if(!d)
    return;

  if(gui->guipoints)
    dt_masks_dynbuf_reset(gui->guipoints);
  if(gui->guipoints_payload)
    dt_masks_dynbuf_reset(gui->guipoints_payload);
  gui->guipoints_count = 0;

  g_free(d->mask);
  d->mask = NULL;
  d->mask_w = d->mask_h = 0;

  if(d->seg)
    dt_seg_reset_prev_mask(d->seg);

  dt_control_queue_redraw_center();
}

static int _object_events_button_pressed(
  dt_iop_module_t *module,
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
  if(!gui)
    return 0;

  _object_data_t *d = _get_data(gui);
  const gboolean auto_mode
    = d && d->autodata && g_atomic_int_get(&d->autodata->auto_state) == AUTO_READY;

  if(gui->creation && which == 1 && dt_modifier_is(state, GDK_MOD1_MASK))
  {
    if(auto_mode)
    {
      // Alt+click: clear all selections
      memset(d->autodata->selected, 0, sizeof(d->autodata->selected));
      dt_control_queue_redraw_center();
    }
    else
    {
      // Alt+click: clear selection (prompt mode)
      if(d && d->encode_state == ENCODE_READY && gui->guipoints_count > 0)
        _clear_selection(gui);
    }
    return 1;
  }
  else if(gui->creation && which == 1)
  {
    if(auto_mode)
    {
      // Auto mode: click = select, shift+click = deselect
      _auto_data_t *ad = d->autodata;
      const int hover = g_atomic_int_get(&ad->hover_mask);
      if(hover >= 0 && hover < ad->n_masks)
      {
        if(dt_modifier_is(state, GDK_SHIFT_MASK))
          ad->selected[hover] = FALSE;
        else
          ad->selected[hover] = TRUE;
      }
      dt_control_queue_redraw_center();
      return 1;
    }

    // Prompt mode: only accept clicks after encoding is complete
    if(!d || d->encode_state != ENCODE_READY)
      return 1;

    // Start drag tracking — actual point/box is added on button release
    float wd, ht, iwidth, iheight;
    dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);

    d->dragging = TRUE;
    d->drag_start_x = pzx * wd;
    d->drag_start_y = pzy * ht;
    d->drag_end_x = d->drag_start_x;
    d->drag_end_y = d->drag_start_y;
    return 1;
  }
  else if(gui->creation && which == 3)
  {
    // Don't exit while background threads are running
    if(d && g_atomic_int_get(&d->encode_state) == ENCODE_RUNNING)
      return 1;
    if(d && d->autodata && g_atomic_int_get(&d->autodata->auto_state) == AUTO_RUNNING)
      return 1;

    // Right-click: finalize mask
    if(auto_mode)
      _finalize_auto_mask(module, form, gui);
    else if(gui->guipoints_count > 0)
      _finalize_mask(module, form, gui);

    // Cleanup and exit creation mode
    _free_data(gui);

    dt_masks_dynbuf_free(gui->guipoints);
    dt_masks_dynbuf_free(gui->guipoints_payload);
    gui->guipoints = NULL;
    gui->guipoints_payload = NULL;
    gui->guipoints_count = 0;

    gui->creation_continuous = FALSE;
    gui->creation_continuous_module = NULL;

    dt_masks_set_edit_mode(module, DT_MASKS_EDIT_FULL);
    dt_masks_iop_update(module);
    dt_control_queue_redraw_center();
    return 1;
  }

  return 0;
}

static int _object_events_button_released(
  dt_iop_module_t *module,
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
  (void)form;
  (void)parentid;
  (void)index;

  if(!gui || which != 1)
    return 0;

  _object_data_t *d = _get_data(gui);
  if(!d || !d->dragging)
    return 0;

  d->dragging = FALSE;

  // Initialize point buffers if needed
  if(!gui->guipoints)
    gui->guipoints = dt_masks_dynbuf_init(200000, "object guipoints");
  if(!gui->guipoints)
    return 1;
  if(!gui->guipoints_payload)
    gui->guipoints_payload = dt_masks_dynbuf_init(100000, "object guipoints_payload");
  if(!gui->guipoints_payload)
    return 1;

  const float dx = d->drag_end_x - d->drag_start_x;
  const float dy = d->drag_end_y - d->drag_start_y;
  const float dist = sqrtf(dx * dx + dy * dy);

  if(dist < BOX_DRAG_THRESHOLD)
  {
    // Short click: single point (foreground or background)
    const float label = dt_modifier_is(state, GDK_SHIFT_MASK) ? 0.0f : 1.0f;
    dt_masks_dynbuf_add_2(gui->guipoints, d->drag_start_x, d->drag_start_y);
    dt_masks_dynbuf_add(gui->guipoints_payload, label);
    gui->guipoints_count++;
  }
  else
  {
    // Box drag: add top-left (label=2) and bottom-right (label=3) corners
    const float x0 = MIN(d->drag_start_x, d->drag_end_x);
    const float y0 = MIN(d->drag_start_y, d->drag_end_y);
    const float x1 = MAX(d->drag_start_x, d->drag_end_x);
    const float y1 = MAX(d->drag_start_y, d->drag_end_y);

    dt_masks_dynbuf_add_2(gui->guipoints, x0, y0);
    dt_masks_dynbuf_add(gui->guipoints_payload, 2.0f);
    gui->guipoints_count++;

    dt_masks_dynbuf_add_2(gui->guipoints, x1, y1);
    dt_masks_dynbuf_add(gui->guipoints_payload, 3.0f);
    gui->guipoints_count++;
  }

  _run_decoder(gui);
  dt_control_queue_redraw_center();
  return 1;
}

static int _object_events_mouse_moved(
  dt_iop_module_t *module,
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

  if(!gui)
    return 0;

  gui->form_selected = FALSE;
  gui->border_selected = FALSE;
  gui->source_selected = FALSE;
  gui->feather_selected = -1;
  gui->point_selected = -1;
  gui->seg_selected = -1;
  gui->point_border_selected = -1;

  if(gui->creation)
  {
    _object_data_t *d = _get_data(gui);

    // Track drag position for box prompts (prompt mode)
    if(d && d->dragging)
    {
      float wd, ht, iwidth, iheight;
      dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);
      d->drag_end_x = pzx * wd;
      d->drag_end_y = pzy * ht;
    }

    // Auto mode: update hover_mask from label_map
    if(d && d->autodata
       && g_atomic_int_get(&d->autodata->auto_state) == AUTO_READY
       && d->autodata->label_map)
    {
      _auto_data_t *ad = d->autodata;

      // pzx/pzy are normalized [0,1] — convert to label_map pixel coords
      const int lx = (int)(pzx * (float)ad->lw);
      const int ly = (int)(pzy * (float)ad->lh);

      int new_hover = -1;
      if(lx >= 0 && lx < ad->lw && ly >= 0 && ly < ad->lh)
        new_hover = (int)ad->label_map[ly * ad->lw + lx];

      g_atomic_int_set(&ad->hover_mask, new_hover);
    }

    dt_control_queue_redraw_center();
  }

  return 1;
}

// Timer callback: periodically redraw center so +/- cursor tracks shift key
static gboolean _modifier_poll(gpointer data)
{
  (void)data;
  dt_control_queue_redraw_center();
  return G_SOURCE_CONTINUE;
}

static void _object_events_post_expose(
  cairo_t *cr,
  const float zoom_scale,
  dt_masks_form_gui_t *gui,
  const int index,
  const int num_points)
{
  (void)index;
  (void)num_points;
  if(!gui)
    return;
  if(!gui->creation)
    return;

  // Ensure scratchpad exists
  _object_data_t *d = _get_data(gui);
  if(!d)
  {
    d = g_new0(_object_data_t, 1);
    gui->scratchpad = d;
  }

  // Detect image change: reset encoding if we switched to a different image
  const dt_imgid_t cur_imgid = darktable.develop->image_storage.id;
  const int cur_state = g_atomic_int_get(&d->encode_state);
  if((cur_state == ENCODE_READY || cur_state == ENCODE_ERROR)
     && d->encoded_imgid != cur_imgid)
  {
    if(d->encode_thread)
    {
      g_thread_join(d->encode_thread);
      d->encode_thread = NULL;
    }
    if(d->busy)
    {
      dt_control_busy_leave();
      d->busy = FALSE;
    }
    if(d->seg)
      dt_seg_reset_encoding(d->seg);
    g_free(d->mask);
    d->mask = NULL;
    d->mask_w = d->mask_h = 0;
    d->encode_w = d->encode_h = 0;
    _auto_data_free(d->autodata);
    d->autodata = NULL;
    d->encode_state = ENCODE_IDLE;
  }

  // Eager encoding: load model and encode image as soon as tool opens
  if(d->encode_state == ENCODE_IDLE)
  {
    // Frame 1: show "working..." and return so it renders before we copy backbuf
    dt_control_busy_enter();
    d->busy = TRUE;
    d->encode_state = ENCODE_MSG_SHOWN;
    dt_control_queue_redraw_center();
    return;
  }

  if(d->encode_state == ENCODE_MSG_SHOWN)
  {
    // Frame 2: launch background thread to render and encode the image.
    // The thread creates a temporary export pipe at high resolution
    // instead of using the low-res preview backbuf.
    _encode_thread_data_t *td = g_new(_encode_thread_data_t, 1);
    td->d = d;
    td->imgid = cur_imgid;

    d->encoded_imgid = cur_imgid;
    d->encode_state = ENCODE_RUNNING;
    // Start poll timer BEFORE the thread — it will detect completion
    // and also tracks modifier keys once encoding is ready
    if(!d->modifier_poll_id)
      d->modifier_poll_id = g_timeout_add(100, _modifier_poll, NULL);
    d->encode_thread = g_thread_new("ai-mask-encode", _encode_thread_func, td);
    return;
  }

  if(g_atomic_int_get(&d->encode_state) == ENCODE_RUNNING)
    return; // background thread in progress, poll timer will trigger redraw

  if(g_atomic_int_get(&d->encode_state) == ENCODE_READY && d->encode_thread)
  {
    // Thread finished (detected by poll timer redraw) — join it
    g_thread_join(d->encode_thread);
    d->encode_thread = NULL;
    if(d->busy)
    {
      dt_control_busy_leave();
      d->busy = FALSE;
    }
  }

  if(g_atomic_int_get(&d->encode_state) == ENCODE_ERROR)
  {
    if(d->encode_thread)
    {
      g_thread_join(d->encode_thread);
      d->encode_thread = NULL;
      // Log only once when the thread is first joined
      dt_control_log(_("AI mask encoding failed"));
    }
    if(d->busy)
    {
      dt_control_busy_leave();
      d->busy = FALSE;
    }
    return;
  }

  if(d->encode_state != ENCODE_READY)
    return;

  const gboolean auto_mode = _is_auto_mode();

  // --- Auto mode: launch auto-segmentation after encoding ---
  if(auto_mode && !d->autodata)
  {
    _auto_data_t *ad = g_new0(_auto_data_t, 1);
    ad->hover_mask = -1;
    ad->auto_state = AUTO_RUNNING;
    d->autodata = ad;

    dt_control_busy_enter();
    d->busy = TRUE;

    ad->auto_thread = g_thread_new("ai-mask-auto", _auto_thread_func, d);
    dt_control_queue_redraw_center();
    return;
  }

  if(auto_mode && d->autodata)
  {
    _auto_data_t *ad = d->autodata;
    const int astate = g_atomic_int_get(&ad->auto_state);

    if(astate == AUTO_RUNNING)
    {
      // Show progress
      const int done = g_atomic_int_get(&ad->grid_done);
      const int total = ad->grid_total;
      if(total > 0)
        dt_control_log(_("auto-segmenting: %d/%d grid points"), done, total);
      return;
    }

    if(astate == AUTO_ERROR)
    {
      if(ad->auto_thread)
      {
        g_thread_join(ad->auto_thread);
        ad->auto_thread = NULL;
        dt_control_log(_("auto-segmentation failed"));
      }
      if(d->busy)
      {
        dt_control_busy_leave();
        d->busy = FALSE;
      }
      return;
    }

    if(astate == AUTO_READY && ad->auto_thread)
    {
      g_thread_join(ad->auto_thread);
      ad->auto_thread = NULL;
      if(d->busy)
      {
        dt_control_busy_leave();
        d->busy = FALSE;
      }
      dt_control_log(_("auto-segmentation: %d objects found"), ad->n_masks);
    }

    if(astate != AUTO_READY)
      return;
  }

  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);

  // --- Auto mode drawing: red overlay for hover/selected masks ---
  if(auto_mode && d->autodata)
  {
    _auto_data_t *ad = d->autodata;
    if(ad->lw > 0 && ad->lh > 0 && ad->label_map)
    {
      const int mw = ad->lw;
      const int mh = ad->lh;
      const int stride = cairo_format_stride_for_width(CAIRO_FORMAT_ARGB32, mw);
      unsigned char *buf = g_try_malloc0((size_t)stride * mh);
      if(buf)
      {
        const int hover = g_atomic_int_get(&ad->hover_mask);

        for(int y = 0; y < mh; y++)
        {
          unsigned char *row = buf + y * stride;
          for(int x = 0; x < mw; x++)
          {
            const int16_t mid = ad->label_map[y * mw + x];
            if(mid < 0 || mid >= ad->n_masks)
              continue;

            const gboolean is_selected = ad->selected[mid];
            const gboolean is_hover = (mid == hover);

            unsigned char alpha = 0;
            if(is_selected && is_hover)
              alpha = 120;
            else if(is_selected)
              alpha = 100;
            else if(is_hover)
              alpha = 60;

            if(alpha > 0)
            {
              row[x * 4 + 0] = 0;     // B
              row[x * 4 + 1] = 0;     // G
              row[x * 4 + 2] = alpha; // R (premultiplied)
              row[x * 4 + 3] = alpha; // A
            }
          }
        }

        cairo_surface_t *surface
          = cairo_image_surface_create_for_data(buf, CAIRO_FORMAT_ARGB32, mw, mh, stride);

        if(surface)
        {
          cairo_save(cr);
          cairo_scale(cr, wd / mw, ht / mh);
          cairo_set_source_surface(cr, surface, 0, 0);
          cairo_paint(cr);
          cairo_restore(cr);
          cairo_surface_destroy(surface);
        }
        g_free(buf);
      }
    }
  }
  else
  {
    // --- Prompt mode drawing: red overlay of current mask ---
    if(d->mask && d->mask_w > 0 && d->mask_h > 0)
    {
      const int mw = d->mask_w;
      const int mh = d->mask_h;
      const int stride = cairo_format_stride_for_width(CAIRO_FORMAT_ARGB32, mw);
      unsigned char *buf = g_try_malloc0((size_t)stride * mh);
      if(buf)
      {
        for(int y = 0; y < mh; y++)
        {
          unsigned char *row = buf + y * stride;
          for(int x = 0; x < mw; x++)
          {
            const float val = d->mask[y * mw + x];
            if(val > 0.5f)
            {
              const unsigned char alpha = 80;
              row[x * 4 + 0] = 0;     // B
              row[x * 4 + 1] = 0;     // G
              row[x * 4 + 2] = alpha; // R (premultiplied)
              row[x * 4 + 3] = alpha; // A
            }
          }
        }

        cairo_surface_t *surface
          = cairo_image_surface_create_for_data(buf, CAIRO_FORMAT_ARGB32, mw, mh, stride);

        if(surface)
        {
          cairo_save(cr);
          cairo_scale(cr, wd / mw, ht / mh);
          cairo_set_source_surface(cr, surface, 0, 0);
          cairo_paint(cr);
          cairo_restore(cr);
          cairo_surface_destroy(surface);
        }
        g_free(buf);
      }
    }
  }

  // Query pointer modifier state reliably (gdk_keymap doesn't work on macOS)
  GtkWidget *cw = dt_ui_center(darktable.gui->ui);
  GdkWindow *win = gtk_widget_get_window(cw);
  GdkDevice *pointer
    = gdk_seat_get_pointer(gdk_display_get_default_seat(gdk_display_get_default()));
  GdkModifierType mod = 0;
  if(win && pointer)
    gdk_window_get_device_position(win, pointer, NULL, NULL, &mod);
  const gboolean shift_held = (mod & GDK_SHIFT_MASK) != 0;

  // Draw +/- cursor indicator (white, like other mask controls)
  const float r = DT_PIXEL_APPLY_DPI(8.0f) / zoom_scale;
  const float lw = DT_PIXEL_APPLY_DPI(2.0f) / zoom_scale;
  cairo_set_line_width(cr, lw);
  cairo_set_source_rgba(cr, 0.9, 0.9, 0.9, 0.9);

  // Horizontal line (common to both + and -)
  cairo_move_to(cr, gui->posx - r, gui->posy);
  cairo_line_to(cr, gui->posx + r, gui->posy);
  cairo_stroke(cr);

  if(!shift_held)
  {
    // Add mode: vertical line to form "+"
    cairo_move_to(cr, gui->posx, gui->posy - r);
    cairo_line_to(cr, gui->posx, gui->posy + r);
    cairo_stroke(cr);
  }

  // Draw dashed rectangle while dragging a box prompt
  if(d->dragging)
  {
    const float box_lw = DT_PIXEL_APPLY_DPI(1.5f) / zoom_scale;
    cairo_set_line_width(cr, box_lw);
    cairo_set_source_rgba(cr, 0.9, 0.9, 0.9, 0.8);

    const double dashes[] = { 4.0 / zoom_scale, 4.0 / zoom_scale };
    cairo_set_dash(cr, dashes, 2, 0);

    cairo_rectangle(cr,
      MIN(d->drag_start_x, d->drag_end_x),
      MIN(d->drag_start_y, d->drag_end_y),
      fabsf(d->drag_end_x - d->drag_start_x),
      fabsf(d->drag_end_y - d->drag_start_y));
    cairo_stroke(cr);

    cairo_set_dash(cr, NULL, 0, 0);
  }

}

// --- Stub functions (object is transient — result is path masks) ---

static int _object_get_points(
  dt_develop_t *dev,
  const float x,
  const float y,
  const float radius,
  const float radius2,
  const float rotation,
  float **points,
  int *points_count)
{
  (void)dev;
  (void)x;
  (void)y;
  (void)radius;
  (void)radius2;
  (void)rotation;
  *points = NULL;
  *points_count = 0;
  return 0;
}

static int _object_get_points_border(
  dt_develop_t *dev,
  struct dt_masks_form_t *form,
  float **points,
  int *points_count,
  float **border,
  int *border_count,
  const int source,
  const dt_iop_module_t *module)
{
  (void)dev;
  (void)form;
  (void)points;
  (void)points_count;
  (void)border;
  (void)border_count;
  (void)source;
  (void)module;
  return 0;
}

static int _object_get_source_area(
  dt_iop_module_t *module,
  dt_dev_pixelpipe_iop_t *piece,
  dt_masks_form_t *form,
  int *width,
  int *height,
  int *posx,
  int *posy)
{
  (void)module;
  (void)piece;
  (void)form;
  (void)width;
  (void)height;
  (void)posx;
  (void)posy;
  return 1;
}

static int _object_get_area(
  const dt_iop_module_t *const restrict module,
  const dt_dev_pixelpipe_iop_t *const restrict piece,
  dt_masks_form_t *const restrict form,
  int *width,
  int *height,
  int *posx,
  int *posy)
{
  (void)module;
  (void)piece;
  (void)form;
  (void)width;
  (void)height;
  (void)posx;
  (void)posy;
  return 1;
}

static int _object_get_mask(
  const dt_iop_module_t *const restrict module,
  const dt_dev_pixelpipe_iop_t *const restrict piece,
  dt_masks_form_t *const restrict form,
  float **buffer,
  int *width,
  int *height,
  int *posx,
  int *posy)
{
  (void)module;
  (void)piece;
  (void)form;
  (void)buffer;
  (void)width;
  (void)height;
  (void)posx;
  (void)posy;
  return 1;
}

static int _object_get_mask_roi(
  const dt_iop_module_t *const restrict module,
  const dt_dev_pixelpipe_iop_t *const restrict piece,
  dt_masks_form_t *const form,
  const dt_iop_roi_t *const roi,
  float *const restrict buffer)
{
  (void)module;
  (void)piece;
  (void)form;
  (void)roi;
  (void)buffer;
  return 1;
}

static GSList *_object_setup_mouse_actions(const struct dt_masks_form_t *const form)
{
  (void)form;
  GSList *lm = NULL;
  lm = dt_mouse_action_create_simple(
    lm,
    DT_MOUSE_ACTION_LEFT,
    0,
    _("[OBJECT] add foreground point"));
  lm = dt_mouse_action_create_simple(
    lm,
    DT_MOUSE_ACTION_LEFT_DRAG,
    0,
    _("[OBJECT] add box prompt"));
  lm = dt_mouse_action_create_simple(
    lm,
    DT_MOUSE_ACTION_LEFT,
    GDK_SHIFT_MASK,
    _("[OBJECT] add background point"));
  lm = dt_mouse_action_create_simple(
    lm,
    DT_MOUSE_ACTION_RIGHT,
    0,
    _("[OBJECT] apply mask"));
  lm = dt_mouse_action_create_simple(
    lm,
    DT_MOUSE_ACTION_SCROLL,
    GDK_CONTROL_MASK,
    _("[OBJECT] change opacity"));
  return lm;
}

static void _object_sanitize_config(dt_masks_type_t type) { (void)type; }

static void _object_set_form_name(dt_masks_form_t *const form, const size_t nb)
{
  snprintf(form->name, sizeof(form->name), _("object #%d"), (int)nb);
}

static void _object_set_hint_message(
  const dt_masks_form_gui_t *const gui,
  const dt_masks_form_t *const form,
  const int opacity,
  char *const restrict msgbuf,
  const size_t msgbuf_len)
{
  (void)form;
  if(gui->creation)
    g_snprintf(
      msgbuf,
      msgbuf_len,
      _("<b>add</b>: click, <b>box</b>: drag, <b>subtract</b>: shift+click, "
        "<b>clear</b>: alt+click, <b>apply</b>: right-click\n"
        "<b>opacity</b>: ctrl+scroll (%d%%)"),
      opacity);
}

static void _object_duplicate_points(
  dt_develop_t *dev,
  dt_masks_form_t *const base,
  dt_masks_form_t *const dest)
{
  (void)dev;
  (void)base;
  (void)dest;
}

static void _object_modify_property(
  dt_masks_form_t *const form,
  const dt_masks_property_t prop,
  const float old_val,
  const float new_val,
  float *sum,
  int *count,
  float *min,
  float *max)
{
  (void)form;
  (void)prop;
  (void)old_val;
  (void)new_val;
  (void)sum;
  (void)count;
  (void)min;
  (void)max;
}

static void
_object_initial_source_pos(const float iwd, const float iht, float *x, float *y)
{
  (void)iwd;
  (void)iht;
  (void)x;
  (void)y;
}

// The function table for object masks
const dt_masks_functions_t dt_masks_functions_object = {
  .point_struct_size = sizeof(struct dt_masks_point_object_t),
  .sanitize_config = _object_sanitize_config,
  .setup_mouse_actions = _object_setup_mouse_actions,
  .set_form_name = _object_set_form_name,
  .set_hint_message = _object_set_hint_message,
  .modify_property = _object_modify_property,
  .duplicate_points = _object_duplicate_points,
  .initial_source_pos = _object_initial_source_pos,
  .get_distance = _object_get_distance,
  .get_points = _object_get_points,
  .get_points_border = _object_get_points_border,
  .get_mask = _object_get_mask,
  .get_mask_roi = _object_get_mask_roi,
  .get_area = _object_get_area,
  .get_source_area = _object_get_source_area,
  .mouse_moved = _object_events_mouse_moved,
  .mouse_scrolled = _object_events_mouse_scrolled,
  .button_pressed = _object_events_button_pressed,
  .button_released = _object_events_button_released,
  .post_expose = _object_events_post_expose};

gboolean dt_masks_object_available(void)
{
  if(!darktable.ai_registry || !darktable.ai_registry->ai_enabled)
    return FALSE;
  char *model_id = dt_conf_get_string(CONF_OBJECT_MODEL_KEY);
  dt_ai_model_t *model = dt_ai_models_get_by_id(darktable.ai_registry, model_id);
  g_free(model_id);
  const gboolean available = model && model->status == DT_AI_MODEL_DOWNLOADED;
  dt_ai_model_free(model);
  return available;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
