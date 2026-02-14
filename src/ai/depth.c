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

#include "depth.h"
#include "backend.h"
#include "common/darktable.h"
#include <math.h>
#include <string.h>

// Depth Anything V2 expects input size to be a multiple of 14.
// 518 = 37 * 14, the default used by the official repo.
#define DA2_INPUT_SIZE 518

// ImageNet normalization constants (applied after /255 scaling)
static const float IMG_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float IMG_STD[3] = {0.229f, 0.224f, 0.225f};

struct dt_depth_context_t
{
  dt_ai_context_t *model;
};

// --- Preprocessing ---

// Resize RGB uint8 image to DA2_INPUT_SIZE x DA2_INPUT_SIZE,
// normalize with ImageNet mean/std (after /255), convert HWC -> CHW.
// Output: float buffer [1, 3, DA2_INPUT_SIZE, DA2_INPUT_SIZE]
static float *
_preprocess_image(const uint8_t *rgb_data, int width, int height)
{
  const int target = DA2_INPUT_SIZE;
  const size_t buf_size = (size_t)3 * target * target;
  float *output = g_try_malloc0(buf_size * sizeof(float));
  if(!output)
    return NULL;

  // Bilinear resize + normalize + HWC->CHW in one pass
  for(int y = 0; y < target; y++)
  {
    const float src_y = (float)y * (float)(height - 1) / (float)(target - 1);
    const int y0 = (int)src_y;
    const int y1 = (y0 + 1 < height) ? y0 + 1 : y0;
    const float fy = src_y - y0;

    for(int x = 0; x < target; x++)
    {
      const float src_x = (float)x * (float)(width - 1) / (float)(target - 1);
      const int x0 = (int)src_x;
      const int x1 = (x0 + 1 < width) ? x0 + 1 : x0;
      const float fx = src_x - x0;

      for(int c = 0; c < 3; c++)
      {
        // Bilinear interpolation
        const float v00 = rgb_data[(y0 * width + x0) * 3 + c];
        const float v01 = rgb_data[(y0 * width + x1) * 3 + c];
        const float v10 = rgb_data[(y1 * width + x0) * 3 + c];
        const float v11 = rgb_data[(y1 * width + x1) * 3 + c];

        const float val = v00 * (1.0f - fx) * (1.0f - fy) + v01 * fx * (1.0f - fy)
          + v10 * (1.0f - fx) * fy + v11 * fx * fy;

        // Scale to [0,1] then normalize with ImageNet stats, write in CHW layout
        output[c * target * target + y * target + x]
          = (val / 255.0f - IMG_MEAN[c]) / IMG_STD[c];
      }
    }
  }

  return output;
}

// --- Bilinear resize helper ---

// Resize a single-channel float buffer from (src_w, src_h) to (dst_w, dst_h).
static void _resize_bilinear(const float *src, int src_w, int src_h,
                              float *dst, int dst_w, int dst_h)
{
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

// --- Public API ---

dt_depth_context_t *dt_depth_load(dt_ai_environment_t *env, const char *model_id)
{
  if(!env || !model_id)
    return NULL;

  dt_ai_context_t *model
    = dt_ai_load_model(env, model_id, "model.onnx", DT_AI_PROVIDER_AUTO);
  if(!model)
  {
    dt_print(DT_DEBUG_AI, "[depth] failed to load model %s", model_id);
    return NULL;
  }

  dt_depth_context_t *ctx = g_new0(dt_depth_context_t, 1);
  ctx->model = model;

  dt_print(DT_DEBUG_AI, "[depth] model loaded: %s", model_id);
  return ctx;
}

float *dt_depth_compute(dt_depth_context_t *ctx,
                         const uint8_t *rgb_data,
                         int width, int height,
                         int *out_width, int *out_height)
{
  if(!ctx || !rgb_data || width <= 0 || height <= 0)
    return NULL;

  // Preprocess: resize to DA2_INPUT_SIZE, normalize, HWC->CHW
  float *preprocessed = _preprocess_image(rgb_data, width, height);
  if(!preprocessed)
    return NULL;

  // Set up input tensor [1, 3, DA2_INPUT_SIZE, DA2_INPUT_SIZE]
  int64_t input_shape[4] = {1, 3, DA2_INPUT_SIZE, DA2_INPUT_SIZE};
  dt_ai_tensor_t input = {
    .data = preprocessed,
    .type = DT_AI_FLOAT,
    .shape = input_shape,
    .ndim = 4
  };

  // Query output shape from model
  int64_t model_out_shape[4];
  const int out_ndim = dt_ai_get_output_shape(ctx->model, 0, model_out_shape, 4);
  if(out_ndim < 2)
  {
    dt_print(DT_DEBUG_AI, "[depth] unexpected output ndim %d", out_ndim);
    g_free(preprocessed);
    return NULL;
  }

  // DA2 output: [1, 14*floor(H/14), 14*floor(W/14)] or [1, H, W]
  // For fixed 518 input -> output is [1, 518, 518]
  // But shapes may be symbolic (-1), in which case use DA2_INPUT_SIZE
  const int depth_h = (out_ndim >= 3 && model_out_shape[1] > 0)
    ? (int)model_out_shape[1] : DA2_INPUT_SIZE;
  const int depth_w = (out_ndim >= 3 && model_out_shape[2] > 0)
    ? (int)model_out_shape[2] : DA2_INPUT_SIZE;

  const size_t depth_size = (size_t)depth_h * depth_w;
  float *raw_depth = g_try_malloc(depth_size * sizeof(float));
  if(!raw_depth)
  {
    g_free(preprocessed);
    return NULL;
  }

  int64_t output_shape[4] = {1, depth_h, depth_w, 0};
  dt_ai_tensor_t output = {
    .data = raw_depth,
    .type = DT_AI_FLOAT,
    .shape = output_shape,
    .ndim = (out_ndim >= 4) ? 4 : 3
  };

  const int ret = dt_ai_run(ctx->model, &input, 1, &output, 1);
  g_free(preprocessed);

  if(ret != 0)
  {
    dt_print(DT_DEBUG_AI, "[depth] inference failed: %d", ret);
    g_free(raw_depth);
    return NULL;
  }

  // Min-max normalize to [0,1] and invert (DA2 outputs inverse depth:
  // higher = closer; we want 0 = near, 1 = far)
  float dmin = raw_depth[0], dmax = raw_depth[0];
  for(size_t i = 1; i < depth_size; i++)
  {
    if(raw_depth[i] < dmin) dmin = raw_depth[i];
    if(raw_depth[i] > dmax) dmax = raw_depth[i];
  }

  const float range = (dmax - dmin) > 1e-6f ? (dmax - dmin) : 1.0f;
  for(size_t i = 0; i < depth_size; i++)
    raw_depth[i] = 1.0f - (raw_depth[i] - dmin) / range;

  dt_print(DT_DEBUG_AI, "[depth] raw output %dx%d, range [%.3f, %.3f]",
           depth_w, depth_h, dmin, dmax);

  // Resize to original image dimensions
  if(depth_w == width && depth_h == height)
  {
    *out_width = width;
    *out_height = height;
    return raw_depth;
  }

  float *resized = g_try_malloc((size_t)width * height * sizeof(float));
  if(!resized)
  {
    g_free(raw_depth);
    return NULL;
  }

  _resize_bilinear(raw_depth, depth_w, depth_h, resized, width, height);
  g_free(raw_depth);

  *out_width = width;
  *out_height = height;
  return resized;
}

void dt_depth_free(dt_depth_context_t *ctx)
{
  if(!ctx)
    return;
  if(ctx->model)
    dt_ai_unload_model(ctx->model);
  g_free(ctx);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
