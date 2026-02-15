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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "bauhaus/bauhaus.h"
#include "common/darktable.h"
#include "common/imagebuf.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "control/control.h"
#include "common/gaussian.h"
#include "common/guided_filter.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"

#ifdef HAVE_AI
#include "ai/depth.h"
#endif

#include <math.h>
#include <stdlib.h>
#include <string.h>

DT_MODULE_INTROSPECTION(1, dt_iop_dgrade_params_t)

#define CONF_DEPTH_MODEL_KEY "plugins/darkroom/masks/depth/model"

typedef struct dt_iop_dgrade_params_t
{
  float center;    // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.5 $DESCRIPTION: "distance"
  float range;     // $MIN: 0.01 $MAX: 1.0 $DEFAULT: 0.5 $DESCRIPTION: "range"
  float exposure;  // $MIN: -3.0 $MAX: 3.0 $DEFAULT: 0.0 $DESCRIPTION: "exposure"
  float warmth;    // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "warmth"
  float falloff;   // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 1.0 $DESCRIPTION: "falloff"
  float feather;   // $MIN: 0.0 $MAX: 100.0 $DEFAULT: 0.0 $DESCRIPTION: "feather"
  float refine;    // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "edge refine"
  float mask_contrast; // $MIN: 0.1 $MAX: 4.0 $DEFAULT: 1.0 $DESCRIPTION: "mask contrast"
} dt_iop_dgrade_params_t;

typedef struct dt_iop_dgrade_gui_data_t
{
  GtkDrawingArea *area;
  GtkWidget *center, *range, *exposure, *warmth;

  // Advanced mask controls (collapsible section)
  dt_gui_collapsible_section_t cs_mask;
  GtkWidget *falloff, *feather, *refine, *mask_contrast;

  // Depth map thumbnail for GUI visualization (set from preview pipe)
  float *depth_thumb;       // downscaled depth map [0,1], size thumb_w * thumb_h
  int thumb_w, thumb_h;
  gboolean depth_valid;
} dt_iop_dgrade_gui_data_t;

#ifdef HAVE_AI

typedef struct dt_dgrade_cache_t
{
  dt_pthread_mutex_t lock;
  volatile dt_hash_t hash;
  int width, height;
  float *volatile depth_map;  // normalized [0,1] depth map
  dt_depth_context_t *ctx;    // loaded model (persists across images)
} dt_dgrade_cache_t;

#endif // HAVE_AI

const char *name()
{
  return _("depth grading");
}

const char *aliases()
{
  return _("depth|grading|distance|atmosphere");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self,
      _("adjust exposure and color temperature based on estimated scene depth"),
      _("creative"),
      _("linear, RGB, scene-referred"),
      _("linear, RGB"),
      _("linear, RGB, scene-referred"));
}

int default_group()
{
  return IOP_GROUP_EFFECT;
}

int flags()
{
  return IOP_FLAGS_WRITE_RASTER | IOP_FLAGS_ONE_INSTANCE;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

int legacy_params(dt_iop_module_t *self,
                  const void *const old_params,
                  const int old_version,
                  void **new_params,
                  int32_t *new_params_size,
                  int *new_version)
{
  return 1;
}

#ifdef HAVE_AI

static inline dt_hash_t _get_cache_hash(dt_iop_module_t *self)
{
  return dt_hash(DT_INITHASH,
                 &self->dev->image_storage.id,
                 sizeof(self->dev->image_storage.id));
}

// Bilinear resize a single-channel float buffer
static void _resize_bilinear(const float *src, int src_w, int src_h,
                              float *dst, int dst_w, int dst_h)
{
  DT_OMP_FOR()
  for(int y = 0; y < dst_h; y++)
  {
    const float sy = (dst_h > 1) ? (float)y * (float)(src_h - 1) / (float)(dst_h - 1) : 0.0f;
    const int y0 = MIN((int)sy, src_h - 1);
    const int y1 = MIN(y0 + 1, src_h - 1);
    const float fy = sy - (float)y0;

    for(int x = 0; x < dst_w; x++)
    {
      const float sx = (dst_w > 1) ? (float)x * (float)(src_w - 1) / (float)(dst_w - 1) : 0.0f;
      const int x0 = MIN((int)sx, src_w - 1);
      const int x1 = MIN(x0 + 1, src_w - 1);
      const float fx = sx - (float)x0;

      const float v00 = src[y0 * src_w + x0];
      const float v01 = src[y0 * src_w + x1];
      const float v10 = src[y1 * src_w + x0];
      const float v11 = src[y1 * src_w + x1];

      dst[y * dst_w + x] = v00 * (1.0f - fx) * (1.0f - fy) + v01 * fx * (1.0f - fy)
        + v10 * (1.0f - fx) * fy + v11 * fx * fy;
    }
  }
}

static void _clear_cache(dt_dgrade_cache_t *cache)
{
  dt_free_align(cache->depth_map);
  cache->depth_map = NULL;
  cache->width = cache->height = 0;
  cache->hash = DT_INVALID_HASH;
}

// Convert linear RGB float4 to sRGB uint8 RGB (3ch)
static uint8_t *_float4_to_srgb_u8(const float *in, int width, int height)
{
  const size_t npixels = (size_t)width * height;
  uint8_t *out = g_try_malloc(npixels * 3);
  if(!out)
    return NULL;

  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++)
  {
    for(int c = 0; c < 3; c++)
    {
      // Clamp to [0,1]
      float v = in[i * 4 + c];
      v = (v < 0.0f) ? 0.0f : (v > 1.0f) ? 1.0f : v;
      // Linear to sRGB gamma
      v = (v <= 0.0031308f) ? 12.92f * v : 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
      out[i * 3 + c] = (uint8_t)(v * 255.0f + 0.5f);
    }
  }
  return out;
}

// Compute or retrieve cached depth map.
// Returns a depth map at roi_in dimensions. The cache stores the depth map
// at whatever resolution it was first computed; subsequent requests at
// different resolutions (e.g. preview vs full pipe) get a bilinear resize
// of the cached data instead of rerunning the model.
static float *_get_depth_map(dt_iop_module_t *self,
                              const float *input,
                              const dt_iop_roi_t *const roi_in)
{
  dt_dgrade_cache_t *cd = self->data;
  if(!cd) return NULL;

  float *result = NULL;
  const int req_w = roi_in->width;
  const int req_h = roi_in->height;

  dt_pthread_mutex_lock(&cd->lock);

  const dt_hash_t hash = _get_cache_hash(self);
  if(hash == cd->hash && cd->depth_map)
  {
    // Cache hit (same image) — return at requested resolution
    const size_t npixels = (size_t)req_w * req_h;
    result = dt_alloc_align_float(npixels);
    if(result)
    {
      if(cd->width == req_w && cd->height == req_h)
        memcpy(result, cd->depth_map, npixels * sizeof(float));
      else
        _resize_bilinear(cd->depth_map, cd->width, cd->height,
                         result, req_w, req_h);
    }
    dt_pthread_mutex_unlock(&cd->lock);
    return result;
  }

  // Cache miss (different image or empty) — need to compute
  _clear_cache(cd);

  // Load model if not loaded yet
  if(!cd->ctx)
  {
    gchar *model_id = dt_conf_get_string(CONF_DEPTH_MODEL_KEY);
    if(!model_id || !model_id[0])
    {
      g_free(model_id);
      dt_print(DT_DEBUG_AI, "[dgrade] no depth model configured in '%s'", CONF_DEPTH_MODEL_KEY);
      dt_pthread_mutex_unlock(&cd->lock);
      return NULL;
    }

    dt_ai_environment_t *env = dt_ai_env_init(NULL);
    if(env)
    {
      cd->ctx = dt_depth_load(env, model_id);
      if(!cd->ctx)
        dt_print(DT_DEBUG_AI, "[dgrade] failed to load depth model '%s'", model_id);
    }
    g_free(model_id);
  }

  if(!cd->ctx)
  {
    dt_pthread_mutex_unlock(&cd->lock);
    return NULL;
  }

  // Convert float4 linear RGB to sRGB uint8
  uint8_t *rgb_u8 = _float4_to_srgb_u8(input, req_w, req_h);
  if(!rgb_u8)
  {
    dt_pthread_mutex_unlock(&cd->lock);
    return NULL;
  }

  // Run depth model
  int out_w = 0, out_h = 0;
  float *depth = dt_depth_compute(cd->ctx, rgb_u8, req_w, req_h,
                                   &out_w, &out_h);
  g_free(rgb_u8);

  if(!depth)
  {
    dt_print(DT_DEBUG_AI, "[dgrade] depth computation failed");
    dt_pthread_mutex_unlock(&cd->lock);
    return NULL;
  }

  // Store in cache at computed resolution
  cd->depth_map = depth;
  cd->width = out_w;
  cd->height = out_h;
  cd->hash = hash;

  dt_print(DT_DEBUG_AI, "[dgrade] depth map computed %dx%d", out_w, out_h);

  // Return at requested resolution
  const size_t npixels = (size_t)req_w * req_h;
  result = dt_alloc_align_float(npixels);
  if(result)
  {
    if(out_w == req_w && out_h == req_h)
      memcpy(result, depth, npixels * sizeof(float));
    else
      _resize_bilinear(depth, out_w, out_h, result, req_w, req_h);
  }

  dt_pthread_mutex_unlock(&cd->lock);
  return result;
}

#endif // HAVE_AI

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  memcpy(piece->data, p1, self->params_size);

  g_hash_table_remove_all(self->raster_mask.source.masks);
  g_hash_table_insert(self->raster_mask.source.masks,
                      GINT_TO_POINTER(BLEND_RASTER_ID), g_strdup("depth mask"));
}

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid,
             void *const ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  const dt_iop_dgrade_params_t *d = piece->data;
  const size_t ch = piece->colors;

  if(!dt_iop_have_required_input_format(4, self, piece->colors,
                                         ivoid, ovoid, roi_in, roi_out))
    return;

  // Start by copying input to output
  dt_iop_copy_image_roi(ovoid, ivoid, ch, roi_in, roi_out);

#ifdef HAVE_AI
  // Check if we need to do anything (effect parameters at zero = no-op)
  const gboolean has_effect = (d->exposure != 0.0f || d->warmth != 0.0f);
  const gboolean request = dt_iop_is_raster_mask_used(piece->module, BLEND_RASTER_ID);
  const gboolean need_gui_thumb = (piece->pipe->type & DT_DEV_PIXELPIPE_PREVIEW) && self->gui_data;

  if(!has_effect && !request && !need_gui_thumb)
    return;

  // Get cached depth map
  float *depth_map = _get_depth_map(self, (const float *)ivoid, roi_in);
  if(!depth_map)
    return;

  // Pass depth map to GUI for visualization (from preview pipe)
  if(need_gui_thumb)
  {
    dt_iop_dgrade_gui_data_t *g = self->gui_data;
    const int w = roi_in->width;
    const int h = roi_in->height;

    dt_iop_gui_enter_critical_section(self);
    if(g->thumb_w != w || g->thumb_h != h)
    {
      g_free(g->depth_thumb);
      g->depth_thumb = g_malloc(sizeof(float) * w * h);
      g->thumb_w = w;
      g->thumb_h = h;
    }
    if(g->depth_thumb)
    {
      memcpy(g->depth_thumb, depth_map, sizeof(float) * w * h);
      g->depth_valid = TRUE;
    }
    dt_iop_gui_leave_critical_section(self);

    // Queue redraw of the visualization widget
    dt_control_queue_redraw_widget(GTK_WIDGET(g->area));
  }

  // Build gradient mask from depth map + center/range/falloff parameters.
  const float center = d->center;
  const float half_range = MAX(d->range * 0.5f, 0.001f);
  const float inv_half_range = 1.0f / half_range;
  const float falloff = MAX(d->falloff, 0.001f);
  const float fo_edge = 1.0f - falloff;

  const size_t npixels = (size_t)roi_out->width * roi_out->height;
  float *band_mask = dt_alloc_align_float(npixels);
  if(!band_mask)
  {
    dt_free_align(depth_map);
    return;
  }

  // Raised cosine with parameterized falloff:
  // falloff=1.0: full smooth cosine (default). falloff→0: hard binary cutoff.
  DT_OMP_FOR()
  for(size_t i = 0; i < npixels; i++)
  {
    const float dist = fabsf(depth_map[i] - center);
    const float t = CLAMPF(dist * inv_half_range, 0.0f, 1.0f);
    const float tt = CLAMPF((t - fo_edge) / falloff, 0.0f, 1.0f);
    band_mask[i] = 0.5f * (1.0f + cosf(M_PI * tt));
  }

  dt_free_align(depth_map);

  // Feather: gaussian blur on mask for spatial light bleed
  if(d->feather > 0.0f)
  {
    const float sigma = d->feather * roi_out->scale / piece->iscale;
    if(sigma > 0.1f)
      dt_gaussian_fast_blur(band_mask, band_mask, roi_out->width, roi_out->height,
                            sigma, 0.0f, 1.0f, 1);
  }

  // Edge refine: guided filter snaps mask to image edges
  if(d->refine > 0.0f)
  {
    const int w = CLAMP((int)(MIN(roi_out->width, roi_out->height) * 0.02f), 1, 20);
    const float sqrt_eps = 1.0f - 0.999f * d->refine;
    guided_filter((const float *)ivoid, band_mask, band_mask,
                  roi_out->width, roi_out->height, 4, w, sqrt_eps, 1.0f, 0.0f, 1.0f);
  }

  // Mask contrast
  if(d->mask_contrast != 1.0f)
  {
    const float mc = d->mask_contrast;
    DT_OMP_FOR()
    for(size_t i = 0; i < npixels; i++)
      band_mask[i] = powf(band_mask[i], mc);
  }

  // Apply exposure + warmth effect
  if(has_effect)
  {
    float *const restrict out = (float *const restrict)ovoid;
    const float *const restrict mask = band_mask;
    const float exposure = d->exposure;
    const float warmth = d->warmth;

    DT_OMP_FOR()
    for(size_t i = 0; i < npixels; i++)
    {
      const float m = mask[i];
      if(m < 1e-6f) continue;

      const size_t px = i * 4;
      const float ev_factor = exp2f(exposure * m);
      const float w = warmth * m;

      out[px + 0] = out[px + 0] * ev_factor * (1.0f + 0.5f * w);  // R
      out[px + 1] = out[px + 1] * ev_factor;                        // G
      out[px + 2] = out[px + 2] * ev_factor * (1.0f - 0.5f * w);  // B
    }
  }

  // Publish raster mask if requested by another module
  if(request)
  {
    // dt_iop_piece_set_raster takes ownership of the mask buffer
    dt_iop_piece_set_raster(piece, band_mask, roi_in, roi_out);
  }
  else
  {
    dt_iop_piece_clear_raster(piece, NULL);
    dt_free_align(band_mask);
  }

#else // !HAVE_AI
  // AI not available — module is a no-op
  (void)d;
#endif
}

void init(dt_iop_module_t *self)
{
  dt_iop_default_init(self);

#ifdef HAVE_AI
  dt_dgrade_cache_t *cd = calloc(1, sizeof(dt_dgrade_cache_t));
  cd->hash = DT_INVALID_HASH;
  dt_pthread_mutex_init(&cd->lock, NULL);
  cd->depth_map = NULL;
  cd->width = cd->height = 0;
  cd->ctx = NULL;
  self->data = cd;
#endif
}

void cleanup(dt_iop_module_t *self)
{
#ifdef HAVE_AI
  dt_dgrade_cache_t *cd = self->data;
  if(cd)
  {
    _clear_cache(cd);
    if(cd->ctx) dt_depth_free(cd->ctx);
    dt_pthread_mutex_destroy(&cd->lock);
    free(cd);
    self->data = NULL;
  }
#endif

  dt_iop_default_cleanup(self);
}

// Draw callback for depth map visualization widget
static gboolean _area_draw(GtkWidget *widget, cairo_t *crf, dt_iop_module_t *self)
{
  dt_iop_dgrade_gui_data_t *g = self->gui_data;
  const dt_iop_dgrade_params_t *p = self->params;

  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
  const int width = allocation.width;
  const int height = allocation.height - DT_RESIZE_HANDLE_SIZE;

  if(width < 2 || height < 2) return FALSE;

  cairo_surface_t *cst = dt_cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
  cairo_t *cr = cairo_create(cst);

  // Dark background
  cairo_set_source_rgb(cr, 0.1, 0.1, 0.1);
  cairo_paint(cr);

  dt_iop_gui_enter_critical_section(self);
  const gboolean valid = g->depth_valid;
  const int tw = g->thumb_w;
  const int th = g->thumb_h;
  float *thumb = NULL;
  if(valid && g->depth_thumb && tw > 0 && th > 0)
  {
    thumb = g_malloc(sizeof(float) * tw * th);
    if(thumb) memcpy(thumb, g->depth_thumb, sizeof(float) * tw * th);
  }
  dt_iop_gui_leave_critical_section(self);

  if(thumb)
  {
    // Render depth map with gradient mask overlay
    const float center = p->center;
    const float half_range = MAX(p->range * 0.5f, 0.001f);
    const float inv_half_range = 1.0f / half_range;
    const float falloff = MAX(p->falloff, 0.001f);
    const float fo_edge = 1.0f - falloff;
    const float mc = p->mask_contrast;

    // Create an image surface from depth data
    cairo_surface_t *depth_surf = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, tw, th);
    unsigned char *data = cairo_image_surface_get_data(depth_surf);
    const int stride = cairo_image_surface_get_stride(depth_surf);

    for(int y = 0; y < th; y++)
    {
      unsigned char *row = data + y * stride;
      for(int x = 0; x < tw; x++)
      {
        const float depth = thumb[y * tw + x];

        // Band mask with parameterized falloff
        const float dist = fabsf(depth - center);
        const float t = CLAMPF(dist * inv_half_range, 0.0f, 1.0f);
        const float tt = CLAMPF((t - fo_edge) / falloff, 0.0f, 1.0f);
        float mask = 0.5f * (1.0f + cosf(M_PI * tt));

        // Mask contrast (skip feather/refine — spatial ops)
        if(mc != 1.0f) mask = powf(mask, mc);

        // Grayscale depth value
        const float gray = depth * 255.0f;

        // Blend: unmasked areas are dim grayscale, masked areas are tinted
        const float dim = 0.3f;
        const float blend = dim + (1.0f - dim) * mask;
        const unsigned char r = (unsigned char)CLAMPF(gray * blend + mask * 40.0f, 0.0f, 255.0f);
        const unsigned char g_val = (unsigned char)CLAMPF(gray * blend + mask * 60.0f, 0.0f, 255.0f);
        const unsigned char b = (unsigned char)CLAMPF(gray * blend + mask * 80.0f, 0.0f, 255.0f);

        // ARGB32 in native byte order (pre-multiplied)
        row[x * 4 + 0] = b;  // B
        row[x * 4 + 1] = g_val; // G
        row[x * 4 + 2] = r;  // R
        row[x * 4 + 3] = 255; // A
      }
    }

    cairo_surface_mark_dirty(depth_surf);

    // Scale depth image to fit the widget while preserving aspect ratio
    const double scale_x = (double)width / tw;
    const double scale_y = (double)height / th;
    const double scale = MIN(scale_x, scale_y);
    const double ox = (width - tw * scale) * 0.5;
    const double oy = (height - th * scale) * 0.5;

    cairo_save(cr);
    cairo_translate(cr, ox, oy);
    cairo_scale(cr, scale, scale);
    cairo_set_source_surface(cr, depth_surf, 0, 0);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
    cairo_paint(cr);
    cairo_restore(cr);

    cairo_surface_destroy(depth_surf);
    g_free(thumb);

    // Draw band indicator lines at the bottom
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(2.0));

    // Band range bar
    const float bar_y = height - DT_PIXEL_APPLY_DPI(4.0f);
    const float bar_h = DT_PIXEL_APPLY_DPI(3.0f);

    // Full range background
    cairo_set_source_rgba(cr, 0.3, 0.3, 0.3, 0.7);
    cairo_rectangle(cr, 0, bar_y, width, bar_h);
    cairo_fill(cr);

    // Selected band
    const float band_x0 = CLAMPF(center - half_range, 0.0f, 1.0f) * width;
    const float band_x1 = CLAMPF(center + half_range, 0.0f, 1.0f) * width;
    cairo_set_source_rgba(cr, 0.7, 0.7, 0.7, 0.9);
    cairo_rectangle(cr, band_x0, bar_y, band_x1 - band_x0, bar_h);
    cairo_fill(cr);

    // Center marker
    const float cx = CLAMPF(center, 0.0f, 1.0f) * width;
    cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.9);
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(2.0));
    cairo_move_to(cr, cx, bar_y);
    cairo_line_to(cr, cx, bar_y + bar_h);
    cairo_stroke(cr);
  }
  else
  {
    // No depth data yet — show placeholder text
    cairo_set_source_rgba(cr, 0.5, 0.5, 0.5, 0.8);
    cairo_select_font_face(cr, "sans-serif", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, DT_PIXEL_APPLY_DPI(11.0));
    cairo_text_extents_t ext;
    const char *msg = _("enable module to compute depth map");
    cairo_text_extents(cr, msg, &ext);
    cairo_move_to(cr, (width - ext.width) / 2.0, (height + ext.height) / 2.0);
    cairo_show_text(cr, msg);
  }

  cairo_destroy(cr);
  cairo_set_source_surface(crf, cst, 0, 0);
  cairo_paint(crf);
  cairo_surface_destroy(cst);
  return FALSE;
}

int button_pressed(dt_iop_module_t *self,
                   const float pzx,
                   const float pzy,
                   const double pressure,
                   const int which,
                   const int type,
                   const uint32_t state,
                   const float zoom_scale)
{
  if(which != 1 || !self->enabled) return 0;

  dt_iop_dgrade_gui_data_t *g = self->gui_data;
  if(!g) return 0;

  // Transform from normalized preview space to module input pixel space
  float wd, ht;
  dt_dev_get_preview_size(self->dev, &wd, &ht);
  float pts[2] = { pzx * wd, pzy * ht };

  dt_dev_distort_backtransform_plus(self->dev, self->dev->preview_pipe,
                                     self->iop_order,
                                     DT_DEV_TRANSFORM_DIR_FORW_EXCL,
                                     pts, 1);

  dt_iop_gui_enter_critical_section(self);
  float depth_value = -1.0f;

  if(g->depth_valid && g->depth_thumb && g->thumb_w > 0 && g->thumb_h > 0)
  {
    const int dx = (int)pts[0];
    const int dy = (int)pts[1];

    if(dx >= 0 && dx < g->thumb_w && dy >= 0 && dy < g->thumb_h)
      depth_value = g->depth_thumb[dy * g->thumb_w + dx];
  }
  dt_iop_gui_leave_critical_section(self);

  if(depth_value >= 0.0f)
  {
    dt_bauhaus_slider_set(g->center, depth_value);
    return 1;
  }

  return 0;
}

static gboolean _area_button_press(GtkWidget *widget, GdkEventButton *event, dt_iop_module_t *self)
{
  if(event->button != 1) return FALSE;

  dt_iop_dgrade_gui_data_t *g = self->gui_data;

  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
  const int width = allocation.width;
  const int height = allocation.height - DT_RESIZE_HANDLE_SIZE;

  dt_iop_gui_enter_critical_section(self);
  const gboolean valid = g->depth_valid;
  const int tw = g->thumb_w;
  const int th = g->thumb_h;
  float depth_value = -1.0f;

  if(valid && g->depth_thumb && tw > 0 && th > 0)
  {
    const double scale_x = (double)width / tw;
    const double scale_y = (double)height / th;
    const double scale = MIN(scale_x, scale_y);
    const double ox = (width - tw * scale) * 0.5;
    const double oy = (height - th * scale) * 0.5;

    const int dx = (int)((event->x - ox) / scale);
    const int dy = (int)((event->y - oy) / scale);

    if(dx >= 0 && dx < tw && dy >= 0 && dy < th)
      depth_value = g->depth_thumb[dy * tw + dx];
  }
  dt_iop_gui_leave_critical_section(self);

  if(depth_value >= 0.0f)
  {
    dt_bauhaus_slider_set(g->center, depth_value);
    return TRUE;
  }

  return FALSE;
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_dgrade_gui_data_t *g = IOP_GUI_ALLOC(dgrade);

  g->depth_thumb = NULL;
  g->thumb_w = g->thumb_h = 0;
  g->depth_valid = FALSE;

  // Create the module widget container so the drawing area is packed before sliders
  self->widget = dt_gui_vbox();

  // Depth map visualization area — clamp saved height to sane minimum
  const int saved_h = dt_conf_get_int("plugins/darkroom/dgrade/graphheight");
  if(saved_h < 100) dt_conf_set_int("plugins/darkroom/dgrade/graphheight", 0);
  g->area = GTK_DRAWING_AREA(dt_ui_resize_wrap(NULL,
                                               0,
                                               "plugins/darkroom/dgrade/graphheight"));
  gtk_widget_set_tooltip_text(GTK_WIDGET(g->area),
                              _("depth map preview — bright areas are far, dark areas are near.\n"
                                "highlighted region shows the selected depth band.\n"
                                "click to pick distance at that point."));
  g_signal_connect(G_OBJECT(g->area), "draw", G_CALLBACK(_area_draw), self);
  gtk_widget_add_events(GTK_WIDGET(g->area), GDK_BUTTON_PRESS_MASK);
  g_signal_connect(G_OBJECT(g->area), "button-press-event", G_CALLBACK(_area_button_press), self);
  dt_gui_box_add(self->widget, GTK_WIDGET(g->area));

  g->center = dt_bauhaus_slider_from_params(self, "center");
  dt_bauhaus_slider_set_format(g->center, "%");
  dt_bauhaus_slider_set_factor(g->center, 100.0f);
  gtk_widget_set_tooltip_text(g->center, _("center of the depth band (0%% = near, 100%% = far)"));

  g->range = dt_bauhaus_slider_from_params(self, "range");
  dt_bauhaus_slider_set_format(g->range, "%");
  dt_bauhaus_slider_set_factor(g->range, 100.0f);
  gtk_widget_set_tooltip_text(g->range, _("width of the depth band around center"));

  dt_gui_box_add(self->widget, dt_ui_section_label_new(C_("section", "effect")));

  g->exposure = dt_bauhaus_slider_from_params(self, "exposure");
  dt_bauhaus_slider_set_format(g->exposure, _(" EV"));
  dt_bauhaus_slider_set_soft_range(g->exposure, -3.0f, 3.0f);
  gtk_widget_set_tooltip_text(g->exposure, _("exposure adjustment for the selected depth band"));

  g->warmth = dt_bauhaus_slider_from_params(self, "warmth");
  dt_bauhaus_slider_set_soft_range(g->warmth, -1.0f, 1.0f);
  gtk_widget_set_tooltip_text(g->warmth, _("color temperature shift (negative = cool, positive = warm)"));

  // Advanced mask controls — collapsed by default
  dt_gui_new_collapsible_section(&g->cs_mask,
                                 "plugins/darkroom/dgrade/expand_mask",
                                 _("mask refinement"),
                                 GTK_BOX(self->widget),
                                 DT_ACTION(self));
  GtkWidget *saved = self->widget;
  self->widget = GTK_WIDGET(g->cs_mask.container);

  g->falloff = dt_bauhaus_slider_from_params(self, "falloff");
  dt_bauhaus_slider_set_format(g->falloff, "%");
  dt_bauhaus_slider_set_factor(g->falloff, 100.0f);
  gtk_widget_set_tooltip_text(g->falloff, _("transition steepness at band edges\n"
                                             "100%% = smooth cosine, 0%% = hard cutoff"));

  g->feather = dt_bauhaus_slider_from_params(self, "feather");
  dt_bauhaus_slider_set_format(g->feather, _(" px"));
  dt_bauhaus_slider_set_soft_range(g->feather, 0.0f, 50.0f);
  gtk_widget_set_tooltip_text(g->feather, _("spatial blur on the mask — creates light bleed across depth edges"));

  g->refine = dt_bauhaus_slider_from_params(self, "refine");
  dt_bauhaus_slider_set_format(g->refine, "%");
  dt_bauhaus_slider_set_factor(g->refine, 100.0f);
  gtk_widget_set_tooltip_text(g->refine, _("snap mask boundaries to actual object edges in the image"));

  g->mask_contrast = dt_bauhaus_slider_from_params(self, "mask_contrast");
  dt_bauhaus_slider_set_soft_range(g->mask_contrast, 0.1f, 4.0f);
  gtk_widget_set_tooltip_text(g->mask_contrast, _("reshape mask: < 1.0 broadens, > 1.0 narrows"));

  self->widget = saved;
}

void gui_changed(dt_iop_module_t *self, GtkWidget *w, void *previous)
{
  dt_iop_dgrade_gui_data_t *g = self->gui_data;
  // Redraw the depth visualization when mask shape sliders change
  if(w == g->center || w == g->range || w == g->falloff
     || w == g->mask_contrast)
    gtk_widget_queue_draw(GTK_WIDGET(g->area));
}

void gui_cleanup(dt_iop_module_t *self)
{
  dt_iop_dgrade_gui_data_t *g = self->gui_data;
  g_free(g->depth_thumb);
  g->depth_thumb = NULL;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
