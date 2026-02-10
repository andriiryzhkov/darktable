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

#include "segmentation.h"
#include "backend.h"
#include "common/darktable.h"
#include <math.h>
#include <string.h>

// SAM encoder expects 1024x1024 input
#define SAM_INPUT_SIZE 1024

// ImageNet normalization constants
static const float IMG_MEAN[3] = {123.675f, 116.28f, 103.53f};
static const float IMG_STD[3] = {58.395f, 57.12f, 57.375f};

struct dt_seg_context_t
{
  dt_ai_context_t *encoder;
  dt_ai_context_t *decoder;

  // Cached encoder outputs
  float *image_embeddings;  // [1, 256, 64, 64]
  float *interm_embeddings; // [1, 1, 64, 64, 160]
  size_t embed_size;        // total floats in image_embeddings
  size_t interm_size;       // total floats in interm_embeddings

  // Low-res mask from previous decode (for iterative refinement)
  float low_res_masks[1 * 1 * 256 * 256];
  gboolean has_prev_mask;

  // Image dimensions that were encoded
  int encoded_width;
  int encoded_height;
  float scale; // SAM_INPUT_SIZE / max(w, h)
  gboolean image_encoded;
};

// --- Preprocessing ---

// Resize RGB image so longest side = SAM_INPUT_SIZE, pad with zeros,
// normalize with ImageNet mean/std, convert HWC -> CHW.
// Output: float buffer [1, 3, SAM_INPUT_SIZE, SAM_INPUT_SIZE]
static float *
_preprocess_image(const uint8_t *rgb_data, int width, int height, float *out_scale)
{
  const int target = SAM_INPUT_SIZE;
  const float scale = (float)target / (float)(width > height ? width : height);
  const int new_w = MIN((int)(width * scale + 0.5f), target);
  const int new_h = MIN((int)(height * scale + 0.5f), target);

  *out_scale = scale;

  const size_t buf_size = (size_t)3 * target * target;
  float *output = g_try_malloc0(buf_size * sizeof(float));
  if(!output)
    return NULL;

  // Bilinear resize + normalize + HWC->CHW in one pass
  for(int y = 0; y < new_h; y++)
  {
    const float src_y = (float)y / scale;
    const int y0 = (int)src_y;
    const int y1 = (y0 + 1 < height) ? y0 + 1 : y0;
    const float fy = src_y - y0;

    for(int x = 0; x < new_w; x++)
    {
      const float src_x = (float)x / scale;
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

        // Normalize and write in CHW layout
        // CHW: channel c, row y, col x -> offset = c * H * W + y * W + x
        output[c * target * target + y * target + x] = (val - IMG_MEAN[c]) / IMG_STD[c];
      }
    }
  }
  // Padded area is already zero from g_try_malloc0

  return output;
}

// --- Public API ---

dt_seg_context_t *dt_seg_load(dt_ai_environment_t *env, const char *model_id)
{
  if(!env || !model_id)
    return NULL;

  dt_ai_context_t *encoder
    = dt_ai_load_model(env, model_id, "encoder.onnx", DT_AI_PROVIDER_AUTO);
  if(!encoder)
  {
    dt_print(DT_DEBUG_AI, "[segmentation] Failed to load encoder for %s", model_id);
    return NULL;
  }

  dt_ai_context_t *decoder
    = dt_ai_load_model(env, model_id, "decoder.onnx", DT_AI_PROVIDER_AUTO);
  if(!decoder)
  {
    dt_print(DT_DEBUG_AI, "[segmentation] Failed to load decoder for %s", model_id);
    dt_ai_unload_model(encoder);
    return NULL;
  }

  dt_seg_context_t *ctx = g_new0(dt_seg_context_t, 1);
  ctx->encoder = encoder;
  ctx->decoder = decoder;

  dt_print(DT_DEBUG_AI, "[segmentation] Model loaded: %s", model_id);
  return ctx;
}

gboolean
dt_seg_encode_image(dt_seg_context_t *ctx, const uint8_t *rgb_data, int width, int height)
{
  if(!ctx || !rgb_data || width <= 0 || height <= 0)
    return FALSE;

  // Skip if already encoded for this image
  if(ctx->image_encoded)
    return TRUE;

  float scale;
  float *preprocessed = _preprocess_image(rgb_data, width, height, &scale);
  if(!preprocessed)
    return FALSE;

  // Run encoder
  int64_t input_shape[4] = {1, 3, SAM_INPUT_SIZE, SAM_INPUT_SIZE};
  dt_ai_tensor_t input
    = {.data = preprocessed, .type = DT_AI_FLOAT, .shape = input_shape, .ndim = 4};

  // Allocate output buffers
  // image_embeddings: [1, 256, 64, 64]
  const size_t embed_size = 1 * 256 * 64 * 64;
  float *embeddings = g_try_malloc(embed_size * sizeof(float));
  if(!embeddings)
  {
    g_free(preprocessed);
    return FALSE;
  }

  // interm_embeddings: [1, 1, 64, 64, 160]
  const size_t interm_size = 1 * 1 * 64 * 64 * 160;
  float *interm = g_try_malloc(interm_size * sizeof(float));
  if(!interm)
  {
    g_free(preprocessed);
    g_free(embeddings);
    return FALSE;
  }

  int64_t embed_shape[4] = {1, 256, 64, 64};
  int64_t interm_shape[5] = {1, 1, 64, 64, 160};

  dt_ai_tensor_t outputs[2] = {
    {.data = embeddings, .type = DT_AI_FLOAT, .shape = embed_shape, .ndim = 4},
    {.data = interm, .type = DT_AI_FLOAT, .shape = interm_shape, .ndim = 5}};

  dt_print(
    DT_DEBUG_AI,
    "[segmentation] Encoding image %dx%d (scale=%.3f)...",
    width,
    height,
    scale);

  const int ret = dt_ai_run(ctx->encoder, &input, 1, outputs, 2);
  g_free(preprocessed);

  if(ret != 0)
  {
    dt_print(DT_DEBUG_AI, "[segmentation] Encoder failed: %d", ret);
    g_free(embeddings);
    g_free(interm);
    return FALSE;
  }

  // Cache results
  g_free(ctx->image_embeddings);
  g_free(ctx->interm_embeddings);
  ctx->image_embeddings = embeddings;
  ctx->interm_embeddings = interm;
  ctx->embed_size = embed_size;
  ctx->interm_size = interm_size;
  ctx->encoded_width = width;
  ctx->encoded_height = height;
  ctx->scale = scale;
  ctx->image_encoded = TRUE;
  ctx->has_prev_mask = FALSE;

  dt_print(DT_DEBUG_AI, "[segmentation] Image encoded successfully");
  return TRUE;
}

float *dt_seg_compute_mask(
  dt_seg_context_t *ctx,
  const dt_seg_point_t *points,
  int n_points,
  int *out_width,
  int *out_height)
{
  if(!ctx || !ctx->image_encoded || !points || n_points <= 0)
    return NULL;

  // Build decoder inputs
  // point_coords: [1, N, 2] — coordinates scaled to 1024-space
  float *point_coords = g_new(float, n_points * 2);
  float *point_labels = g_new(float, n_points);

  for(int i = 0; i < n_points; i++)
  {
    point_coords[i * 2 + 0] = points[i].x * ctx->scale;
    point_coords[i * 2 + 1] = points[i].y * ctx->scale;
    point_labels[i] = (float)points[i].label;
  }

  const float orig_im_size[2] = {(float)ctx->encoded_height, (float)ctx->encoded_width};
  const float has_mask = ctx->has_prev_mask ? 1.0f : 0.0f;

  // Decoder inputs (7 total, matching the ONNX model)
  int64_t embed_shape[4] = {1, 256, 64, 64};
  int64_t interm_shape[5] = {1, 1, 64, 64, 160};
  int64_t coords_shape[3] = {1, n_points, 2};
  int64_t labels_shape[2] = {1, n_points};
  int64_t mask_in_shape[4] = {1, 1, 256, 256};
  int64_t has_mask_shape[1] = {1};
  int64_t orig_size_shape[1] = {2};

  dt_ai_tensor_t inputs[7] = {
    {.data = ctx->image_embeddings, .type = DT_AI_FLOAT, .shape = embed_shape, .ndim = 4},
    {.data = ctx->interm_embeddings,
     .type = DT_AI_FLOAT,
     .shape = interm_shape,
     .ndim = 5},
    {.data = point_coords, .type = DT_AI_FLOAT, .shape = coords_shape, .ndim = 3},
    {.data = point_labels, .type = DT_AI_FLOAT, .shape = labels_shape, .ndim = 2},
    {.data = ctx->low_res_masks, .type = DT_AI_FLOAT, .shape = mask_in_shape, .ndim = 4},
    {.data = (void *)&has_mask, .type = DT_AI_FLOAT, .shape = has_mask_shape, .ndim = 1},
    {.data = (void *)orig_im_size,
     .type = DT_AI_FLOAT,
     .shape = orig_size_shape,
     .ndim = 1}};

  // Decoder outputs (3 total)
  // masks: [1, 1, H, W] where H,W = orig_im_size
  const int mask_h = ctx->encoded_height;
  const int mask_w = ctx->encoded_width;
  const size_t mask_size = (size_t)mask_h * mask_w;
  float *masks = g_try_malloc(mask_size * sizeof(float));
  if(!masks)
  {
    g_free(point_coords);
    g_free(point_labels);
    return NULL;
  }

  float iou_pred[1];
  const size_t low_res_size = 1 * 1 * 256 * 256;
  float *low_res = g_try_malloc(low_res_size * sizeof(float));
  if(!low_res)
  {
    g_free(point_coords);
    g_free(point_labels);
    g_free(masks);
    return NULL;
  }

  int64_t masks_shape[4] = {1, 1, mask_h, mask_w};
  int64_t iou_shape[2] = {1, 1};
  int64_t low_res_shape[4] = {1, 1, 256, 256};

  dt_ai_tensor_t outputs[3] = {
    {.data = masks, .type = DT_AI_FLOAT, .shape = masks_shape, .ndim = 4},
    {.data = iou_pred, .type = DT_AI_FLOAT, .shape = iou_shape, .ndim = 2},
    {.data = low_res, .type = DT_AI_FLOAT, .shape = low_res_shape, .ndim = 4}};

  const int ret = dt_ai_run(ctx->decoder, inputs, 7, outputs, 3);

  g_free(point_coords);
  g_free(point_labels);

  if(ret != 0)
  {
    dt_print(DT_DEBUG_AI, "[segmentation] Decoder failed: %d", ret);
    g_free(low_res);
    g_free(masks);
    return NULL;
  }

  dt_print(DT_DEBUG_AI, "[segmentation] Mask computed, IoU=%.3f", iou_pred[0]);

  // Cache low-res mask for iterative refinement
  memcpy(ctx->low_res_masks, low_res, sizeof(ctx->low_res_masks));
  g_free(low_res);
  ctx->has_prev_mask = TRUE;

  // Apply sigmoid to convert logits to [0,1] probabilities
  for(size_t i = 0; i < mask_size; i++)
    masks[i] = 1.0f / (1.0f + expf(-masks[i]));

  if(out_width)
    *out_width = mask_w;
  if(out_height)
    *out_height = mask_h;

  return masks;
}

gboolean dt_seg_is_encoded(dt_seg_context_t *ctx)
{
  return ctx ? ctx->image_encoded : FALSE;
}

void dt_seg_reset_prev_mask(dt_seg_context_t *ctx)
{
  if(!ctx)
    return;
  ctx->has_prev_mask = FALSE;
  memset(ctx->low_res_masks, 0, sizeof(ctx->low_res_masks));
}

void dt_seg_reset_encoding(dt_seg_context_t *ctx)
{
  if(!ctx)
    return;

  g_free(ctx->image_embeddings);
  g_free(ctx->interm_embeddings);
  ctx->image_embeddings = NULL;
  ctx->interm_embeddings = NULL;
  ctx->embed_size = 0;
  ctx->interm_size = 0;
  ctx->encoded_width = 0;
  ctx->encoded_height = 0;
  ctx->scale = 0.0f;
  ctx->image_encoded = FALSE;
  ctx->has_prev_mask = FALSE;
  memset(ctx->low_res_masks, 0, sizeof(ctx->low_res_masks));
}

void dt_seg_free(dt_seg_context_t *ctx)
{
  if(!ctx)
    return;

  if(ctx->encoder)
    dt_ai_unload_model(ctx->encoder);
  if(ctx->decoder)
    dt_ai_unload_model(ctx->decoder);
  g_free(ctx->image_embeddings);
  g_free(ctx->interm_embeddings);
  g_free(ctx);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
