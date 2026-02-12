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

// Maximum number of dimensions for encoder output tensors
#define MAX_TENSOR_DIMS 8

struct dt_seg_context_t
{
  dt_ai_context_t *encoder;
  dt_ai_context_t *decoder;

  // Encoder output shapes (queried from model at load time)
  int64_t embed_shape[MAX_TENSOR_DIMS];   // image_embeddings shape from model
  int embed_ndim;
  int64_t interm_shape[MAX_TENSOR_DIMS];  // interm_embeddings shape from model
  int interm_ndim;

  // Decoder output: number of masks per decode (1 = single-mask, 3-4 = multi-mask)
  int num_masks;
  gboolean has_hq; // TRUE if decoder has high_res_masks output (SAM-HQ)

  // Cached encoder outputs
  float *image_embeddings;
  float *interm_embeddings;
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

dt_seg_context_t *dt_seg_load(dt_ai_environment_t *env, const char *model_id,
                              dt_ai_provider_t provider)
{
  if(!env || !model_id)
    return NULL;

  dt_ai_context_t *encoder
    = dt_ai_load_model(env, model_id, "encoder.onnx", provider);
  if(!encoder)
  {
    dt_print(DT_DEBUG_AI, "[segmentation] Failed to load encoder for %s", model_id);
    return NULL;
  }

  dt_ai_context_t *decoder
    = dt_ai_load_model(env, model_id, "decoder.onnx", provider);
  if(!decoder)
  {
    dt_print(DT_DEBUG_AI, "[segmentation] Failed to load decoder for %s", model_id);
    dt_ai_unload_model(encoder);
    return NULL;
  }

  dt_seg_context_t *ctx = g_new0(dt_seg_context_t, 1);
  ctx->encoder = encoder;
  ctx->decoder = decoder;

  // Query encoder output shapes from model metadata
  ctx->embed_ndim = dt_ai_get_output_shape(encoder, 0, ctx->embed_shape, MAX_TENSOR_DIMS);
  ctx->interm_ndim = dt_ai_get_output_shape(encoder, 1, ctx->interm_shape, MAX_TENSOR_DIMS);

  if(ctx->embed_ndim <= 0 || ctx->interm_ndim <= 0)
  {
    dt_print(DT_DEBUG_AI,
             "[segmentation] Failed to query encoder output shapes for %s", model_id);
    dt_seg_free(ctx);
    return NULL;
  }

  // Query decoder masks output shape to detect multi-mask support.
  // Output 0 (masks) shape: [1, N, H, W] where N = num_masks (1 or 3-4).
  int64_t dec_out_shape[MAX_TENSOR_DIMS];
  const int dec_out_ndim = dt_ai_get_output_shape(decoder, 0, dec_out_shape, MAX_TENSOR_DIMS);
  ctx->num_masks = (dec_out_ndim >= 4 && dec_out_shape[1] > 1) ? (int)dec_out_shape[1] : 1;

  // SAM-HQ models have a 4th decoder output (high_res_masks) at 1024x1024
  // with sharper edges from encoder skip connections.
  ctx->has_hq = (dt_ai_get_output_count(decoder) >= 4);

  dt_print(DT_DEBUG_AI,
           "[segmentation] Model loaded: %s (embed_ndim=%d, interm_ndim=%d, num_masks=%d, hq=%d)",
           model_id, ctx->embed_ndim, ctx->interm_ndim, ctx->num_masks, ctx->has_hq);
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

  // Allocate output buffers using shapes queried from model
  size_t embed_size = 1;
  for(int i = 0; i < ctx->embed_ndim; i++)
    embed_size *= (size_t)ctx->embed_shape[i];

  float *embeddings = g_try_malloc(embed_size * sizeof(float));
  if(!embeddings)
  {
    g_free(preprocessed);
    return FALSE;
  }

  size_t interm_size = 1;
  for(int i = 0; i < ctx->interm_ndim; i++)
    interm_size *= (size_t)ctx->interm_shape[i];

  float *interm = g_try_malloc(interm_size * sizeof(float));
  if(!interm)
  {
    g_free(preprocessed);
    g_free(embeddings);
    return FALSE;
  }

  dt_ai_tensor_t outputs[2] = {
    {.data = embeddings, .type = DT_AI_FLOAT,
     .shape = ctx->embed_shape, .ndim = ctx->embed_ndim},
    {.data = interm, .type = DT_AI_FLOAT,
     .shape = ctx->interm_shape, .ndim = ctx->interm_ndim}};

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

  // Build decoder inputs.
  // SAM ONNX requires a padding point (0,0) with label -1 appended
  // to every prompt (see SAM official onnx_model_example.ipynb).
  const int total_points = n_points + 1;
  float *point_coords = g_new(float, total_points * 2);
  float *point_labels = g_new(float, total_points);

  for(int i = 0; i < n_points; i++)
  {
    point_coords[i * 2 + 0] = points[i].x * ctx->scale;
    point_coords[i * 2 + 1] = points[i].y * ctx->scale;
    point_labels[i] = (float)points[i].label;
  }
  // ONNX padding point
  point_coords[n_points * 2 + 0] = 0.0f;
  point_coords[n_points * 2 + 1] = 0.0f;
  point_labels[n_points] = -1.0f;

  const float orig_im_size[2] = {(float)ctx->encoded_height, (float)ctx->encoded_width};
  const float has_mask = ctx->has_prev_mask ? 1.0f : 0.0f;

  // Decoder inputs (7 total, matching the ONNX model)
  int64_t coords_shape[3] = {1, total_points, 2};
  int64_t labels_shape[2] = {1, total_points};
  int64_t mask_in_shape[4] = {1, 1, 256, 256};
  int64_t has_mask_shape[1] = {1};
  int64_t orig_size_shape[1] = {2};

  dt_ai_tensor_t inputs[7] = {
    {.data = ctx->image_embeddings, .type = DT_AI_FLOAT,
     .shape = ctx->embed_shape, .ndim = ctx->embed_ndim},
    {.data = ctx->interm_embeddings, .type = DT_AI_FLOAT,
     .shape = ctx->interm_shape, .ndim = ctx->interm_ndim},
    {.data = point_coords, .type = DT_AI_FLOAT, .shape = coords_shape, .ndim = 3},
    {.data = point_labels, .type = DT_AI_FLOAT, .shape = labels_shape, .ndim = 2},
    {.data = ctx->low_res_masks, .type = DT_AI_FLOAT, .shape = mask_in_shape, .ndim = 4},
    {.data = (void *)&has_mask, .type = DT_AI_FLOAT, .shape = has_mask_shape, .ndim = 1},
    {.data = (void *)orig_im_size,
     .type = DT_AI_FLOAT,
     .shape = orig_size_shape,
     .ndim = 1}};

  // Decoder outputs: masks [1, N, H, W], iou [1, N], low_res [1, N, 256, 256]
  // N = ctx->num_masks (1 for single-mask models, 3-4 for multi-mask)
  const int nm = ctx->num_masks;
  const int mask_h = ctx->encoded_height;
  const int mask_w = ctx->encoded_width;
  const size_t per_mask = (size_t)mask_h * mask_w;
  const size_t total_mask_size = (size_t)nm * per_mask;

  float *masks = g_try_malloc(total_mask_size * sizeof(float));
  if(!masks)
  {
    g_free(point_coords);
    g_free(point_labels);
    return NULL;
  }

  float iou_pred[8]; // max 8 masks
  const size_t low_res_per = 256 * 256;
  float *low_res = g_try_malloc((size_t)nm * low_res_per * sizeof(float));
  if(!low_res)
  {
    g_free(point_coords);
    g_free(point_labels);
    g_free(masks);
    return NULL;
  }

  int64_t masks_shape[4] = {1, nm, mask_h, mask_w};
  int64_t iou_shape[2] = {1, nm};
  int64_t low_res_shape[4] = {1, nm, 256, 256};

  // Optional 4th output: SAM-HQ high_res_masks [1, N, 1024, 1024]
  const size_t hq_per_mask = (size_t)SAM_INPUT_SIZE * SAM_INPUT_SIZE;
  float *hq_masks = NULL;
  int num_outputs = 3;

  if(ctx->has_hq)
  {
    hq_masks = g_try_malloc((size_t)nm * hq_per_mask * sizeof(float));
    if(hq_masks)
      num_outputs = 4;
  }

  int64_t hq_shape[4] = {1, nm, SAM_INPUT_SIZE, SAM_INPUT_SIZE};

  dt_ai_tensor_t outputs[4] = {
    {.data = masks, .type = DT_AI_FLOAT, .shape = masks_shape, .ndim = 4},
    {.data = iou_pred, .type = DT_AI_FLOAT, .shape = iou_shape, .ndim = 2},
    {.data = low_res, .type = DT_AI_FLOAT, .shape = low_res_shape, .ndim = 4},
    {.data = hq_masks, .type = DT_AI_FLOAT, .shape = hq_shape, .ndim = 4}};

  const int ret = dt_ai_run(ctx->decoder, inputs, 7, outputs, num_outputs);

  g_free(point_coords);
  g_free(point_labels);

  if(ret != 0)
  {
    dt_print(DT_DEBUG_AI, "[segmentation] Decoder failed: %d", ret);
    g_free(hq_masks);
    g_free(low_res);
    g_free(masks);
    return NULL;
  }

  // Select the mask with the highest predicted IoU
  int best = 0;
  for(int m = 1; m < nm; m++)
  {
    if(iou_pred[m] > iou_pred[best])
      best = m;
  }

  dt_print(DT_DEBUG_AI, "[segmentation] Mask computed, best=%d/%d IoU=%.3f",
           best, nm, iou_pred[best]);

  // Cache the best low-res mask for iterative refinement
  memcpy(ctx->low_res_masks, low_res + (size_t)best * low_res_per,
         sizeof(ctx->low_res_masks));
  g_free(low_res);
  ctx->has_prev_mask = TRUE;

  // Extract the best mask and apply sigmoid
  float *result = g_try_malloc(per_mask * sizeof(float));
  if(!result)
  {
    g_free(hq_masks);
    g_free(masks);
    return NULL;
  }

  if(hq_masks)
  {
    // SAM-HQ path: bilinear crop+resize from 1024x1024 to encoded dims.
    // The valid (non-padded) region in 1024-space corresponds to the
    // scaled image area placed in the top-left corner during preprocessing.
    const int valid_w = MIN((int)(ctx->encoded_width * ctx->scale + 0.5f), SAM_INPUT_SIZE);
    const int valid_h = MIN((int)(ctx->encoded_height * ctx->scale + 0.5f), SAM_INPUT_SIZE);
    const float *best_hq = hq_masks + (size_t)best * hq_per_mask;

    for(int y = 0; y < mask_h; y++)
    {
      const float src_y = (float)y * (float)(valid_h - 1) / (float)(mask_h - 1);
      const int y0 = MIN((int)src_y, valid_h - 1);
      const int y1 = MIN(y0 + 1, valid_h - 1);
      const float fy = src_y - (float)y0;

      for(int x = 0; x < mask_w; x++)
      {
        const float src_x = (float)x * (float)(valid_w - 1) / (float)(mask_w - 1);
        const int x0 = MIN((int)src_x, valid_w - 1);
        const int x1 = MIN(x0 + 1, valid_w - 1);
        const float fx = src_x - (float)x0;

        const float v00 = best_hq[y0 * SAM_INPUT_SIZE + x0];
        const float v01 = best_hq[y0 * SAM_INPUT_SIZE + x1];
        const float v10 = best_hq[y1 * SAM_INPUT_SIZE + x0];
        const float v11 = best_hq[y1 * SAM_INPUT_SIZE + x1];

        const float val = v00 * (1.0f - fx) * (1.0f - fy) + v01 * fx * (1.0f - fy)
          + v10 * (1.0f - fx) * fy + v11 * fx * fy;

        result[y * mask_w + x] = 1.0f / (1.0f + expf(-val));
      }
    }

    dt_print(DT_DEBUG_AI, "[segmentation] Using HQ mask (valid=%dx%d -> %dx%d)",
             valid_w, valid_h, mask_w, mask_h);
    g_free(hq_masks);
  }
  else
  {
    // Standard path: extract best mask from orig_im_size output
    const float *best_mask = masks + (size_t)best * per_mask;
    for(size_t i = 0; i < per_mask; i++)
      result[i] = 1.0f / (1.0f + expf(-best_mask[i]));
  }

  g_free(masks);

  if(out_width)
    *out_width = mask_w;
  if(out_height)
    *out_height = mask_h;

  return result;
}

gboolean dt_seg_compute_mask_raw(dt_seg_context_t *ctx,
                                  const dt_seg_point_t *point,
                                  float **out_masks,
                                  float **out_ious,
                                  int *out_n_masks,
                                  int *out_width, int *out_height)
{
  if(!ctx || !ctx->image_encoded || !point || !out_masks || !out_ious
     || !out_n_masks || !out_width || !out_height)
    return FALSE;

  // Single foreground point + ONNX padding point
  float point_coords[4] = {
    point->x * ctx->scale, point->y * ctx->scale,
    0.0f, 0.0f  // padding point
  };
  float point_labels[2] = {(float)point->label, -1.0f};

  const float orig_im_size[2] = {(float)ctx->encoded_height, (float)ctx->encoded_width};
  // has_mask=0 means decoder ignores mask_input content, safe to pass ctx buffer
  const float has_mask = 0.0f;

  int64_t coords_shape[3] = {1, 2, 2};
  int64_t labels_shape[2] = {1, 2};
  int64_t mask_in_shape[4] = {1, 1, 256, 256};
  int64_t has_mask_shape[1] = {1};
  int64_t orig_size_shape[1] = {2};

  dt_ai_tensor_t inputs[7] = {
    {.data = ctx->image_embeddings, .type = DT_AI_FLOAT,
     .shape = ctx->embed_shape, .ndim = ctx->embed_ndim},
    {.data = ctx->interm_embeddings, .type = DT_AI_FLOAT,
     .shape = ctx->interm_shape, .ndim = ctx->interm_ndim},
    {.data = point_coords, .type = DT_AI_FLOAT, .shape = coords_shape, .ndim = 3},
    {.data = point_labels, .type = DT_AI_FLOAT, .shape = labels_shape, .ndim = 2},
    {.data = ctx->low_res_masks, .type = DT_AI_FLOAT, .shape = mask_in_shape, .ndim = 4},
    {.data = (void *)&has_mask, .type = DT_AI_FLOAT, .shape = has_mask_shape, .ndim = 1},
    {.data = (void *)orig_im_size, .type = DT_AI_FLOAT, .shape = orig_size_shape, .ndim = 1}};

  const int nm = ctx->num_masks;
  const int mask_h = ctx->encoded_height;
  const int mask_w = ctx->encoded_width;
  const size_t per_mask = (size_t)mask_h * mask_w;

  float *masks = g_try_malloc((size_t)nm * per_mask * sizeof(float));
  if(!masks)
    return FALSE;

  float iou_pred[8];
  float *low_res = g_try_malloc((size_t)nm * 256 * 256 * sizeof(float));
  if(!low_res)
  {
    g_free(masks);
    return FALSE;
  }

  int64_t masks_shape[4] = {1, nm, mask_h, mask_w};
  int64_t iou_shape[2] = {1, nm};
  int64_t low_res_shape[4] = {1, nm, 256, 256};

  dt_ai_tensor_t outputs[3] = {
    {.data = masks, .type = DT_AI_FLOAT, .shape = masks_shape, .ndim = 4},
    {.data = iou_pred, .type = DT_AI_FLOAT, .shape = iou_shape, .ndim = 2},
    {.data = low_res, .type = DT_AI_FLOAT, .shape = low_res_shape, .ndim = 4}};

  const int ret = dt_ai_run(ctx->decoder, inputs, 7, outputs, 3);
  g_free(low_res);

  if(ret != 0)
  {
    g_free(masks);
    return FALSE;
  }

  // Apply sigmoid to all masks in-place
  for(size_t i = 0; i < (size_t)nm * per_mask; i++)
    masks[i] = 1.0f / (1.0f + expf(-masks[i]));

  // Copy IoU predictions
  float *ious = g_new(float, nm);
  memcpy(ious, iou_pred, nm * sizeof(float));

  *out_masks = masks;
  *out_ious = ious;
  *out_n_masks = nm;
  *out_width = mask_w;
  *out_height = mask_h;
  return TRUE;
}

int dt_seg_get_num_masks(dt_seg_context_t *ctx)
{
  return ctx ? ctx->num_masks : 0;
}

void dt_seg_get_encoded_dims(dt_seg_context_t *ctx, int *width, int *height)
{
  if(!ctx)
  {
    if(width) *width = 0;
    if(height) *height = 0;
    return;
  }
  if(width) *width = ctx->encoded_width;
  if(height) *height = ctx->encoded_height;
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
