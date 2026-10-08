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

#include "common/ai/mask_ai.h"
#include "ai/backend.h"
#include "common/darktable.h"
#include "common/guided_filter.h"
#include "common/math.h"

#include <float.h>
#include <math.h>
#include <string.h>

// a bound on manifest values that size the render, the input tensor and
// the stored mask, far above any model's
#define MASK_AI_MAX_SIZE 4096

// the edge refinement: the guided filter's window radius, in pixels of the
// model's output as upsampled onto the render, so that it spans the blur the
// upsampling leaves, and its regularization, eps = 1e-3 for a guide in [0,1].
// found on skyseg's 320 output upsampled 6.4 times onto 2048 pixels, a radius
// of 8: a wider window leaks the guide's texture into the mask and a smaller
// one leaves the model's blur. the guide is weighted by 10 and eps by 100, the
// same filter: guided_filter.c:186 falls back to a box blur where its system's
// determinant is under 4 * FLT_EPSILON, which colors in [0,1] reach in flat
// areas and along thin lines
#define MASK_AI_REFINE_SPAN 1.25f
#define MASK_AI_REFINE_GUIDE_WEIGHT 10.0f
#define MASK_AI_REFINE_SQRT_EPS (0.032f * MASK_AI_REFINE_GUIDE_WEIGHT)

// the attributes side of the contract (mask_ai.h)
typedef struct _contract_t
{
  int size;       // the square input's side
  float mean[3];  // per channel, on RGB in [0,1]
  float std[3];
  gboolean logits;
  gboolean refine;  // the edges, by a guided filter on the render
} _contract_t;

struct dt_mask_ai_t
{
  dt_ai_environment_t *env;
  dt_ai_context_t *ctx;  // only while dt_mask_ai_run runs
  char *model_id;
  _contract_t c;
  int out_w, out_h;      // the output's fixed size, from the graph
};

// 3 values, or def for all of them when the attribute is absent
static gboolean _read_triple(const dt_ai_model_info_t *info,
                             const char *model_id,
                             const char *key,
                             const float def,
                             float out[3])
{
  int n = 0;
  double *v = dt_ai_model_attribute_double_array(info, key, &n);
  if(!v)
  {
    out[0] = out[1] = out[2] = def;
    return TRUE;
  }
  const gboolean ok = n == 3;
  if(ok)
    for(int c = 0; c < 3; c++) out[c] = (float)v[c];
  else
    dt_print(DT_DEBUG_AI, "[mask ai] %s: %s has %d values, 3 expected",
             model_id, key, n);
  g_free(v);
  return ok;
}

static gboolean _read_contract(const dt_ai_model_info_t *info,
                               const char *model_id,
                               _contract_t *c)
{
  if(!info)
  {
    dt_print(DT_DEBUG_AI, "[mask ai] model %s is not installed", model_id);
    return FALSE;
  }

  if(dt_ai_model_attribute_bool(info, "prompts"))
  {
    dt_print(DT_DEBUG_AI, "[mask ai] %s takes prompts, refused", model_id);
    return FALSE;
  }

  char *strategy = dt_ai_model_attribute_string(info, "strategy");
  const gboolean whole = !g_strcmp0(strategy, "resize_whole_image");
  if(!whole)
    dt_print(DT_DEBUG_AI, "[mask ai] %s: strategy '%s' is not resize_whole_image,"
             " refused", model_id, strategy ? strategy : "");
  g_free(strategy);
  if(!whole) return FALSE;

  int n = 0;
  int *sizes = dt_ai_model_attribute_int_array(info, "input_sizes", &n);
  c->size = 0;
  for(int i = 0; i < n; i++) c->size = MAX(c->size, sizes[i]);
  g_free(sizes);
  if(c->size <= 0 || c->size > MASK_AI_MAX_SIZE)
  {
    dt_print(DT_DEBUG_AI, "[mask ai] %s: no usable input_sizes", model_id);
    return FALSE;
  }

  if(!_read_triple(info, model_id, "input_mean", 0.0f, c->mean)
     || !_read_triple(info, model_id, "input_std", 1.0f, c->std))
    return FALSE;
  for(int k = 0; k < 3; k++)
    if(!(c->std[k] > 0.0f))
    {
      dt_print(DT_DEBUG_AI, "[mask ai] %s: input_std must be positive", model_id);
      return FALSE;
    }

  c->logits = dt_ai_model_attribute_bool(info, "output_logits");

  // a string, not a bool, so that absent can mean on
  char *refine = dt_ai_model_attribute_string(info, "edge_refine");
  c->refine = g_strcmp0(refine, "none") != 0;
  if(refine && c->refine && strcmp(refine, "guided"))
    dt_print(DT_DEBUG_AI, "[mask ai] %s: edge_refine '%s' is unknown, guided is used",
             model_id, refine);
  g_free(refine);
  return TRUE;
}

// a symbolic dimension reads as -1 and takes any value
static inline gboolean _dim_fits(const int64_t dim, const int64_t want)
{
  return dim < 0 || dim == want;
}

// the graph side: one float input [1,3,S,S], one float output [1,1,H,W]
// at a fixed size. a symbolic batch runs as 1
static gboolean _check_graph(dt_ai_context_t *ctx,
                             const char *model_id,
                             const int size,
                             int *out_w,
                             int *out_h)
{
  const int n_in = dt_ai_get_input_count(ctx);
  const int n_out = dt_ai_get_output_count(ctx);
  if(n_in != 1 || n_out != 1)
  {
    dt_print(DT_DEBUG_AI, "[mask ai] %s has %d inputs and %d outputs, 1 and 1"
             " expected", model_id, n_in, n_out);
    return FALSE;
  }

  const dt_ai_dtype_t in_type = dt_ai_get_input_type(ctx, 0);
  const dt_ai_dtype_t out_type = dt_ai_get_output_type(ctx, 0);
  if((in_type != DT_AI_FLOAT && in_type != DT_AI_FLOAT16)
     || (out_type != DT_AI_FLOAT && out_type != DT_AI_FLOAT16))
  {
    dt_print(DT_DEBUG_AI, "[mask ai] %s: input type %d, output type %d, float"
             " expected", model_id, in_type, out_type);
    return FALSE;
  }

  int64_t shape[8] = { 0 };
  int nd = dt_ai_get_input_shape(ctx, 0, shape, 8);
  if(nd != 4 || !_dim_fits(shape[0], 1) || !_dim_fits(shape[1], 3)
     || !_dim_fits(shape[2], size) || !_dim_fits(shape[3], size))
  {
    dt_print(DT_DEBUG_AI, "[mask ai] %s: input is not [1,3,%d,%d] as"
             " input_sizes says", model_id, size, size);
    return FALSE;
  }

  nd = dt_ai_get_output_shape(ctx, 0, shape, 8);
  if(nd != 4 || !_dim_fits(shape[0], 1) || shape[1] != 1
     || shape[2] <= 0 || shape[3] <= 0
     || shape[2] > MASK_AI_MAX_SIZE || shape[3] > MASK_AI_MAX_SIZE)
  {
    dt_print(DT_DEBUG_AI, "[mask ai] %s: output is not [1,1,H,W] with a fixed"
             " H and W", model_id);
    return FALSE;
  }
  *out_h = (int)shape[2];
  *out_w = (int)shape[3];
  return TRUE;
}

// a line of n_in samples, in_stride apart, into n_out samples out_stride
// apart: the mean over each output sample's footprint when shrinking, so a
// detail thinner than the step still counts, bilinear when enlarging
static inline void _resample_line(const float *const restrict in,
                                  const int n_in,
                                  const size_t in_stride,
                                  float *const restrict out,
                                  const int n_out,
                                  const size_t out_stride)
{
  const float ratio = (float)n_in / (float)n_out;
  if(ratio > 1.0f)
  {
    for(int o = 0; o < n_out; o++)
    {
      const float a = o * ratio;
      const float b = MIN((o + 1) * ratio, (float)n_in);
      const int k1 = MIN((int)ceilf(b), n_in);
      float sum = 0.0f;
      for(int k = (int)a; k < k1; k++)
        sum += in[k * in_stride] * (MIN(b, k + 1.0f) - MAX(a, (float)k));
      out[o * out_stride] = sum / (b - a);
    }
  }
  else
  {
    for(int o = 0; o < n_out; o++)
    {
      const float x = CLAMPF((o + 0.5f) * ratio - 0.5f, 0.0f, n_in - 1.0f);
      const int k0 = (int)x;
      const int k1 = MIN(k0 + 1, n_in - 1);
      const float f = x - k0;
      out[o * out_stride] = in[k0 * in_stride] * (1.0f - f) + in[k1 * in_stride] * f;
    }
  }
}

// the view stretched to the square input, whatever its aspect ratio
// (resize_whole_image), normalized, as NCHW. NULL when out of memory
static float *_preprocess(const uint8_t *const restrict rgb,
                          const int w,
                          const int h,
                          const _contract_t *c)
{
  const int s = c->size;
  const size_t npix = (size_t)w * h;
  float *planar = dt_alloc_align_float(3 * npix);
  float *rows = dt_alloc_align_float((size_t)3 * s * h);
  float *out = dt_alloc_align_float((size_t)3 * s * s);
  if(!planar || !rows || !out)
  {
    dt_free_align(planar);
    dt_free_align(rows);
    dt_free_align(out);
    return NULL;
  }

  DT_OMP_FOR()
  for(size_t k = 0; k < npix; k++)
    for(int ch = 0; ch < 3; ch++)
      planar[ch * npix + k] = rgb[k * 3 + ch] * (1.0f / 255.0f);

  // across, then down
  DT_OMP_FOR(collapse(2))
  for(int ch = 0; ch < 3; ch++)
    for(int y = 0; y < h; y++)
      _resample_line(planar + ch * npix + (size_t)y * w, w, 1,
                     rows + ((size_t)ch * h + y) * s, s, 1);

  DT_OMP_FOR(collapse(2))
  for(int ch = 0; ch < 3; ch++)
    for(int x = 0; x < s; x++)
      _resample_line(rows + (size_t)ch * h * s + x, h, s,
                     out + (size_t)ch * s * s + x, s, s);

  dt_free_align(planar);
  dt_free_align(rows);

  for(int ch = 0; ch < 3; ch++)
  {
    float *const restrict plane = out + (size_t)ch * s * s;
    const float mean = c->mean[ch];
    const float inv_std = 1.0f / c->std[ch];
    DT_OMP_FOR_SIMD()
    for(size_t k = 0; k < (size_t)s * s; k++)
      plane[k] = (plane[k] - mean) * inv_std;
  }
  return out;
}

dt_mask_ai_t *dt_mask_ai_open(const char *model_id)
{
  if(!model_id) return NULL;
  dt_ai_environment_t *env = dt_ai_env_init(NULL);
  if(!env) return NULL;

  _contract_t c;
  if(!_read_contract(dt_ai_get_model_info_by_id(env, model_id), model_id, &c))
  {
    dt_ai_env_destroy(env);
    return NULL;
  }
  dt_mask_ai_t *m = g_new0(dt_mask_ai_t, 1);
  m->env = env;
  m->model_id = g_strdup(model_id);
  m->c = c;
  return m;
}

int dt_mask_ai_input_size(const dt_mask_ai_t *m)
{
  return m ? m->c.size : 0;
}

gboolean dt_mask_ai_refines(const dt_mask_ai_t *m)
{
  return m && m->c.refine;
}

// the model's ow x oh soft mask onto the width x height render it was made
// from: upsampled edge to edge, as dt_masks_pixel_store places a mask over
// its render, then a guided filter with the render as guide. it fits the mask
// in each window as a linear function of the colors, so a soft trace the
// model left of a wire or a branch comes back thin and sharp, and nothing is
// thresholded. NULL when out of memory
static float *_refine(const float *const restrict mask,
                      const int ow,
                      const int oh,
                      const uint8_t *const restrict rgb,
                      const int width,
                      const int height)
{
  const size_t npix = (size_t)width * height;
  float *rows = dt_alloc_align_float((size_t)width * oh);
  float *up = dt_alloc_align_float(npix);
  float *guide = dt_alloc_align_float(4 * npix);
  float *out = dt_alloc_align_float(npix);
  if(!rows || !up || !guide || !out)
  {
    dt_free_align(rows);
    dt_free_align(up);
    dt_free_align(guide);
    dt_free_align(out);
    return NULL;
  }

  // across, then down
  DT_OMP_FOR()
  for(int y = 0; y < oh; y++)
    _resample_line(mask + (size_t)y * ow, ow, 1, rows + (size_t)y * width, width, 1);
  DT_OMP_FOR()
  for(int x = 0; x < width; x++)
    _resample_line(rows + x, oh, width, up + x, height, width);
  dt_free_align(rows);

  // four channels: guided_filter reads a fourth from every pixel
  DT_OMP_FOR()
  for(size_t k = 0; k < npix; k++)
  {
    for(int c = 0; c < 3; c++)
      guide[4 * k + c] = rgb[3 * k + c] * (1.0f / 255.0f);
    guide[4 * k + 3] = 0.0f;
  }

  // a model that outputs near the render's size has little blur to undo, and
  // a wider window would only leak colors across the edge
  const float factor = fmaxf((float)width / ow, (float)height / oh);
  const int radius = MAX(1, (int)lrintf(MASK_AI_REFINE_SPAN * factor));
  guided_filter(guide, up, out, width, height, 4, radius,
                MASK_AI_REFINE_SQRT_EPS, MASK_AI_REFINE_GUIDE_WEIGHT, 0.0f, 1.0f);
  dt_free_align(up);
  dt_free_align(guide);
  return out;
}

void dt_mask_ai_close(dt_mask_ai_t *m)
{
  if(!m) return;
  dt_ai_unload_model(m->ctx);
  dt_ai_env_destroy(m->env);
  g_free(m->model_id);
  g_free(m);
}

// the session, with the graph checked against the contract
static gboolean _load(dt_mask_ai_t *m, const dt_ai_provider_t provider)
{
  m->ctx = dt_ai_load_model(m->env, m->model_id, NULL, provider);
  if(!m->ctx)
  {
    dt_print(DT_DEBUG_AI, "[mask ai] failed to load %s", m->model_id);
    return FALSE;
  }
  if(!_check_graph(m->ctx, m->model_id, m->c.size, &m->out_w, &m->out_h))
  {
    dt_ai_unload_model(m->ctx);
    m->ctx = NULL;
    return FALSE;
  }
  return TRUE;
}

// one run into mask. *mismatch is TRUE when the output came out at another
// size than the graph declared, which another provider would not change
static int _run_once(dt_mask_ai_t *m,
                     float *input,
                     float *mask,
                     gboolean *mismatch)
{
  int64_t in_shape[4] = { 1, 3, m->c.size, m->c.size };
  int64_t out_shape[4] = { 1, 1, m->out_h, m->out_w };
  // float32 for a float16 graph too: dt_ai_run converts the input to the
  // type the graph declares and the output back (backend_onnx.c:2764, 3054)
  dt_ai_tensor_t in = { .data = input, .type = DT_AI_FLOAT, .shape = in_shape, .ndim = 4 };
  dt_ai_tensor_t out = { .data = mask, .type = DT_AI_FLOAT, .shape = out_shape, .ndim = 4 };
  int ret = dt_ai_run(m->ctx, &in, 1, &out, 1);
  // an output the backend allocates has its shape written back, and one
  // smaller than declared is copied without an error, leaving the rest of
  // the mask unset
  *mismatch = out.ndim != 4 || out_shape[0] > 1 || out_shape[1] != 1
    || out_shape[2] != m->out_h || out_shape[3] != m->out_w;
  if(ret == 0 && *mismatch)
  {
    dt_print(DT_DEBUG_AI, "[mask ai] %s: the output came out at another size"
             " than the graph declares", m->model_id);
    ret = -1;
  }
  return ret;
}

float *dt_mask_ai_run(dt_mask_ai_t *m,
                      const uint8_t *rgb,
                      const int width,
                      const int height,
                      int *out_w,
                      int *out_h)
{
  *out_w = *out_h = 0;
  if(!m || !rgb || width <= 0 || height <= 0) return NULL;

  // before the session, which takes far more memory than the input. a
  // session the configured provider cannot make is retried on the CPU by
  // the backend (dt_ai_onnx_load_ext), so a failed load is final
  float *input = _preprocess(rgb, width, height, &m->c);
  if(!input)
  {
    dt_print(DT_DEBUG_AI, "[mask ai] no memory for the %dx%d input", m->c.size, m->c.size);
    return NULL;
  }
  if(!_load(m, DT_AI_PROVIDER_CONFIGURED))
  {
    dt_free_align(input);
    return NULL;
  }
  const int ow = m->out_w, oh = m->out_h;
  float *mask = dt_alloc_align_float((size_t)ow * oh);
  int ret = -1;
  gboolean mismatch = FALSE;
  if(mask)
  {
    ret = _run_once(m, input, mask, &mismatch);
    // an accelerated provider can fail on a graph the CPU runs, as the
    // object tool finds. not when the CPU is what is configured; a load
    // that fell back to it gets one needless retry, as in restore.c
    if(ret != 0 && !mismatch && dt_ai_env_get_provider(m->env) != DT_AI_PROVIDER_CPU)
    {
      dt_print(DT_DEBUG_AI, "[mask ai] %s failed (%d), retrying on the CPU",
               m->model_id, ret);
      dt_ai_unload_model(m->ctx);
      ret = _load(m, DT_AI_PROVIDER_CPU) ? _run_once(m, input, mask, &mismatch) : -1;
    }
  }
  dt_ai_unload_model(m->ctx);
  m->ctx = NULL;
  dt_free_align(input);
  if(ret != 0)
  {
    dt_print(DT_DEBUG_AI, "[mask ai] %s failed (%d)", m->model_id, ret);
    dt_free_align(mask);
    return NULL;
  }

  const size_t n = (size_t)ow * oh;
  if(m->c.logits)
  {
    // the guard demo.py of the model uses: a graph with the sigmoid baked in
    // is already in [0,1], and squashing it twice flattens the boundary
    float lo = FLT_MAX, hi = -FLT_MAX;
    DT_OMP_FOR(reduction(min : lo) reduction(max : hi))
    for(size_t k = 0; k < n; k++)
    {
      lo = fminf(lo, mask[k]);
      hi = fmaxf(hi, mask[k]);
    }
    if(lo < 0.0f || hi > 1.0f)
    {
      DT_OMP_FOR_SIMD()
      for(size_t k = 0; k < n; k++)
        mask[k] = 1.0f / (1.0f + expf(-mask[k]));
    }
  }
  // soft as it comes: no threshold
  DT_OMP_FOR_SIMD()
  for(size_t k = 0; k < n; k++)
    mask[k] = CLAMPF(mask[k], 0.0f, 1.0f);

  dt_print(DT_DEBUG_AI, "[mask ai] %s: %dx%d view, %dx%d mask",
           m->model_id, width, height, ow, oh);
  *out_w = ow;
  *out_h = oh;
  if(!m->c.refine) return mask;

  const double t0 = dt_get_wtime();
  float *refined = _refine(mask, ow, oh, rgb, width, height);
  if(!refined)
  {
    // the model's own mask still covers the view, only coarser
    dt_print(DT_DEBUG_AI, "[mask ai] %s: no memory to refine the edges", m->model_id);
    return mask;
  }
  dt_free_align(mask);
  dt_print(DT_DEBUG_AI, "[mask ai] %s: edges refined at %dx%d in %.2fs",
           m->model_id, width, height, dt_get_wtime() - t0);
  *out_w = width;
  *out_h = height;
  return refined;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
