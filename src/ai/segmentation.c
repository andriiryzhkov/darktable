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

// Maximum number of encoder output tensors (2 for SAM-HQ, 3 for SAM2.1)
#define MAX_ENCODER_OUTPUTS 4

struct dt_seg_context_t
{
  dt_ai_context_t *encoder;
  dt_ai_context_t *decoder;

  // Encoder output shapes (queried from model at load time)
  int n_enc_outputs;                                    // 2 for SAM-HQ, 3 for SAM2.1
  int64_t enc_shapes[MAX_ENCODER_OUTPUTS][MAX_TENSOR_DIMS];
  int enc_ndims[MAX_ENCODER_OUTPUTS];

  // Decoder properties
  int num_masks;            // masks per decode (1 = single-mask, 3-4 = multi-mask)
  gboolean has_hq;          // TRUE if decoder has high_res_masks output (SAM-HQ)
  gboolean has_low_res;     // TRUE if decoder has low_res_masks output (enables iterative refinement)
  gboolean has_orig_im_size; // TRUE if decoder expects orig_im_size input (SAM/SAM-HQ)
  int dec_mask_h, dec_mask_w; // decoder mask output dims (-1 if dynamic)
  int low_res_dim;          // spatial dim of low_res masks (256 for SAM, may differ for SAM2)

  // Encoder-to-decoder reorder map: decoder input i uses encoder output enc_order[i].
  // Needed because SAM2 encoder outputs may be in different order than decoder expects.
  int enc_order[MAX_ENCODER_OUTPUTS];

  // Cached encoder outputs
  float *enc_data[MAX_ENCODER_OUTPUTS];
  size_t enc_sizes[MAX_ENCODER_OUTPUTS];

  // Low-res mask from previous decode (for iterative refinement)
  float *low_res_masks;     // [1][1][low_res_dim][low_res_dim]
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

// --- Bilinear crop+resize helper ---

// Crop the valid (non-padded) region from a SAM-space mask and bilinear
// resize to the encoded image dimensions.  Used for HQ masks and SAM2
// decoder outputs that are at the padded SAM_INPUT_SIZE resolution.
static void _crop_resize_mask(const float *src, int src_w, int src_h,
                               float *dst, int dst_w, int dst_h,
                               float scale, gboolean apply_sigmoid)
{
  const int valid_w = MIN((int)(dst_w * scale + 0.5f), src_w);
  const int valid_h = MIN((int)(dst_h * scale + 0.5f), src_h);

  for(int y = 0; y < dst_h; y++)
  {
    const float sy = (float)y * (float)(valid_h - 1) / (float)(dst_h - 1);
    const int y0 = MIN((int)sy, valid_h - 1);
    const int y1 = MIN(y0 + 1, valid_h - 1);
    const float fy = sy - (float)y0;

    for(int x = 0; x < dst_w; x++)
    {
      const float sx = (float)x * (float)(valid_w - 1) / (float)(dst_w - 1);
      const int x0 = MIN((int)sx, valid_w - 1);
      const int x1 = MIN(x0 + 1, valid_w - 1);
      const float fx = sx - (float)x0;

      const float v00 = src[y0 * src_w + x0];
      const float v01 = src[y0 * src_w + x1];
      const float v10 = src[y1 * src_w + x0];
      const float v11 = src[y1 * src_w + x1];

      float val = v00 * (1.0f - fx) * (1.0f - fy) + v01 * fx * (1.0f - fy)
        + v10 * (1.0f - fx) * fy + v11 * fx * fy;

      if(apply_sigmoid)
        val = 1.0f / (1.0f + expf(-val));

      dst[y * dst_w + x] = val;
    }
  }
}

// --- Public API ---

dt_seg_context_t *dt_seg_load(dt_ai_environment_t *env, const char *model_id)
{
  if(!env || !model_id)
    return NULL;

  // Provider is resolved from the environment (read from config at init time).
  // Passing AUTO lets dt_ai_load_model resolve it.
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

  // Query encoder output count and shapes from model metadata
  ctx->n_enc_outputs = dt_ai_get_output_count(encoder);
  if(ctx->n_enc_outputs <= 0 || ctx->n_enc_outputs > MAX_ENCODER_OUTPUTS)
  {
    dt_print(DT_DEBUG_AI,
             "[segmentation] Unsupported encoder output count %d for %s",
             ctx->n_enc_outputs, model_id);
    dt_seg_free(ctx);
    return NULL;
  }

  for(int i = 0; i < ctx->n_enc_outputs; i++)
  {
    ctx->enc_ndims[i]
      = dt_ai_get_output_shape(encoder, i, ctx->enc_shapes[i], MAX_TENSOR_DIMS);
    if(ctx->enc_ndims[i] <= 0)
    {
      dt_print(DT_DEBUG_AI,
               "[segmentation] Failed to query encoder output %d shape for %s",
               i, model_id);
      dt_seg_free(ctx);
      return NULL;
    }
  }

  // Build encoder-to-decoder reorder map by matching output/input names.
  // SAM2 encoder outputs may be in different order than the decoder expects
  // (e.g. encoder: high_res_feats_0, high_res_feats_1, image_embeddings
  //  vs decoder: image_embed, high_res_feats_0, high_res_feats_1).
  for(int i = 0; i < ctx->n_enc_outputs; i++)
    ctx->enc_order[i] = i; // default: same order

  gboolean used[MAX_ENCODER_OUTPUTS] = {FALSE};
  for(int di = 0; di < ctx->n_enc_outputs; di++)
  {
    const char *dec_name = dt_ai_get_input_name(decoder, di);
    if(!dec_name) continue;

    int best = -1;
    for(int ei = 0; ei < ctx->n_enc_outputs; ei++)
    {
      if(used[ei]) continue;
      const char *enc_name = dt_ai_get_output_name(encoder, ei);
      if(!enc_name) continue;

      if(g_strcmp0(dec_name, enc_name) == 0)
      {
        best = ei;
        break; // exact match
      }
      // Substring fallback: e.g. decoder "image_embed" matches encoder
      // "image_embeddings".  Safe because exact matches are tried first
      // and used[] prevents double-assignment of the same encoder output.
      if(best < 0 && (strstr(enc_name, dec_name) || strstr(dec_name, enc_name)))
        best = ei;
    }

    if(best >= 0)
    {
      ctx->enc_order[di] = best;
      used[best] = TRUE;
    }
  }

  dt_print(DT_DEBUG_AI,
           "[segmentation] Encoder-decoder reorder: [%d, %d, %d, %d] (n=%d)",
           ctx->enc_order[0], ctx->enc_order[1], ctx->enc_order[2], ctx->enc_order[3],
           ctx->n_enc_outputs);

  // Query decoder masks output shape to detect multi-mask support.
  // Output 0 (masks) shape: [1, N, H, W] where N = num_masks (1 or 3-4).
  // SAM2 has fully dynamic masks shape [-1,-1,-1,-1], so fall back to
  // iou_predictions output 1 shape [1, N] which has a static N.
  int64_t dec_out_shape[MAX_TENSOR_DIMS];
  const int dec_out_ndim = dt_ai_get_output_shape(decoder, 0, dec_out_shape, MAX_TENSOR_DIMS);
  ctx->num_masks = (dec_out_ndim >= 4 && dec_out_shape[1] > 1) ? (int)dec_out_shape[1] : 0;

  if(ctx->num_masks == 0)
  {
    // Fallback: check iou_predictions shape [1, N]
    int64_t iou_shape[MAX_TENSOR_DIMS];
    const int iou_ndim = dt_ai_get_output_shape(decoder, 1, iou_shape, MAX_TENSOR_DIMS);
    ctx->num_masks = (iou_ndim >= 2 && iou_shape[1] > 0) ? (int)iou_shape[1] : 1;
  }

  // Detect decoder mask output dimensions (may be dynamic = -1)
  ctx->dec_mask_h = (dec_out_ndim >= 4 && dec_out_shape[2] > 0) ? (int)dec_out_shape[2] : -1;
  ctx->dec_mask_w = (dec_out_ndim >= 4 && dec_out_shape[3] > 0) ? (int)dec_out_shape[3] : -1;

  // SAM2 decoders have fully dynamic output shapes because the
  // symbolic dim "num_labels" prevents ONNX Runtime from resolving
  // intermediate Resize→Clip tensor shapes.  Override num_labels=1
  // (darktable always decodes one prompt at a time) so shape inference
  // can fully resolve all spatial dims.  Use BASIC optimization to
  // enable constant folding which propagates the concrete shapes.
  if(ctx->dec_mask_h < 0 && ctx->dec_mask_w < 0)
  {
    dt_print(DT_DEBUG_AI,
             "[segmentation] Decoder has dynamic output dims — reloading with dim overrides");
    dt_ai_unload_model(ctx->decoder);
    const dt_ai_dim_override_t overrides[] = {{"num_labels", 1}};
    ctx->decoder = dt_ai_load_model_ext(env, model_id, "decoder.onnx",
                                         DT_AI_PROVIDER_AUTO, DT_AI_OPT_BASIC,
                                         overrides, 1);
    if(!ctx->decoder)
    {
      dt_print(DT_DEBUG_AI, "[segmentation] Failed to reload decoder for %s", model_id);
      dt_seg_free(ctx);
      return NULL;
    }
    decoder = ctx->decoder;

    // Re-query output shapes from the reloaded decoder — with num_labels=1
    // overriding the symbolic dim, ORT can resolve concrete output shapes.
    const int new_ndim
      = dt_ai_get_output_shape(decoder, 0, dec_out_shape, MAX_TENSOR_DIMS);
    if(new_ndim >= 4)
    {
      ctx->dec_mask_h = dec_out_shape[2] > 0 ? (int)dec_out_shape[2] : -1;
      ctx->dec_mask_w = dec_out_shape[3] > 0 ? (int)dec_out_shape[3] : -1;
      if(dec_out_shape[1] > 1)
        ctx->num_masks = (int)dec_out_shape[1];
    }
    // Re-query num_masks from iou output if still unresolved
    if(ctx->num_masks <= 1)
    {
      int64_t iou_shape[MAX_TENSOR_DIMS];
      const int iou_ndim
        = dt_ai_get_output_shape(decoder, 1, iou_shape, MAX_TENSOR_DIMS);
      if(iou_ndim >= 2 && iou_shape[1] > 0)
        ctx->num_masks = (int)iou_shape[1];
    }

    dt_print(DT_DEBUG_AI,
             "[segmentation] After reload: dec_dims=%dx%d, num_masks=%d",
             ctx->dec_mask_h, ctx->dec_mask_w, ctx->num_masks);
  }

  // Detect decoder output count for feature detection
  const int n_dec_outputs = dt_ai_get_output_count(decoder);

  // SAM-HQ models have a 4th decoder output (high_res_masks) at 1024x1024
  // with sharper edges from encoder skip connections.
  ctx->has_hq = (n_dec_outputs >= 4);

  // SAM/SAM-HQ have low_res_masks as 3rd output (enables iterative refinement).
  // SAM2 decoders only have 2 outputs (masks + iou), no low_res.
  ctx->has_low_res = (n_dec_outputs >= 3);

  // SAM/SAM-HQ decoders have orig_im_size as an additional input.
  // SAM2 decoders do not. Detect by comparing decoder input count
  // with encoder output count: prompts are always 4 (coords, labels,
  // mask_input, has_mask_input), so >4 remaining means orig_im_size.
  const int n_dec_inputs = dt_ai_get_input_count(decoder);
  ctx->has_orig_im_size = (n_dec_inputs - ctx->n_enc_outputs > 4);

  // Query low_res mask output spatial dimensions (output 2 if it exists)
  ctx->low_res_dim = 256; // default
  if(ctx->has_low_res)
  {
    int64_t lr_shape[MAX_TENSOR_DIMS];
    const int lr_ndim = dt_ai_get_output_shape(decoder, 2, lr_shape, MAX_TENSOR_DIMS);
    if(lr_ndim >= 4 && lr_shape[2] > 0 && lr_shape[3] > 0)
      ctx->low_res_dim = (int)lr_shape[2];
  }

  // Allocate low_res mask buffer (needed as decoder input even if no low_res output)
  const size_t lr_size = (size_t)ctx->low_res_dim * ctx->low_res_dim;
  ctx->low_res_masks = g_try_malloc0(lr_size * sizeof(float));
  if(!ctx->low_res_masks)
  {
    dt_seg_free(ctx);
    return NULL;
  }

  dt_print(DT_DEBUG_AI,
           "[segmentation] Model loaded: %s (enc_outputs=%d, dec_outputs=%d, num_masks=%d, "
           "hq=%d, low_res=%d, orig_size=%d, dec_dims=%dx%d)",
           model_id, ctx->n_enc_outputs, n_dec_outputs, ctx->num_masks, ctx->has_hq,
           ctx->has_low_res, ctx->has_orig_im_size, ctx->dec_mask_h, ctx->dec_mask_w);
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

  // Allocate output buffers for all encoder outputs
  float *enc_bufs[MAX_ENCODER_OUTPUTS] = {NULL};
  size_t enc_buf_sizes[MAX_ENCODER_OUTPUTS] = {0};

  for(int i = 0; i < ctx->n_enc_outputs; i++)
  {
    size_t sz = 1;
    for(int d = 0; d < ctx->enc_ndims[i]; d++)
      sz *= (size_t)ctx->enc_shapes[i][d];
    enc_buf_sizes[i] = sz;
    enc_bufs[i] = g_try_malloc(sz * sizeof(float));
    if(!enc_bufs[i])
    {
      for(int j = 0; j < i; j++) g_free(enc_bufs[j]);
      g_free(preprocessed);
      return FALSE;
    }
  }

  dt_ai_tensor_t outputs[MAX_ENCODER_OUTPUTS];
  for(int i = 0; i < ctx->n_enc_outputs; i++)
  {
    outputs[i] = (dt_ai_tensor_t){
      .data = enc_bufs[i], .type = DT_AI_FLOAT,
      .shape = ctx->enc_shapes[i], .ndim = ctx->enc_ndims[i]};
  }

  dt_print(
    DT_DEBUG_AI,
    "[segmentation] Encoding image %dx%d (scale=%.3f)...",
    width,
    height,
    scale);

  const int ret = dt_ai_run(ctx->encoder, &input, 1, outputs, ctx->n_enc_outputs);
  g_free(preprocessed);

  if(ret != 0)
  {
    dt_print(DT_DEBUG_AI, "[segmentation] Encoder failed: %d", ret);
    for(int i = 0; i < ctx->n_enc_outputs; i++) g_free(enc_bufs[i]);
    return FALSE;
  }

  // Cache results
  for(int i = 0; i < MAX_ENCODER_OUTPUTS; i++)
  {
    g_free(ctx->enc_data[i]);
    ctx->enc_data[i] = NULL;
    ctx->enc_sizes[i] = 0;
  }
  for(int i = 0; i < ctx->n_enc_outputs; i++)
  {
    ctx->enc_data[i] = enc_bufs[i];
    ctx->enc_sizes[i] = enc_buf_sizes[i];
  }

  ctx->encoded_width = width;
  ctx->encoded_height = height;
  ctx->scale = scale;
  ctx->image_encoded = TRUE;
  ctx->has_prev_mask = FALSE;

  dt_print(DT_DEBUG_AI, "[segmentation] Image encoded successfully");
  return TRUE;
}

// Determine the decoder mask output dimensions for buffer allocation.
// For SAM/SAM-HQ with orig_im_size, the decoder outputs at encoded dims.
// For SAM2 without orig_im_size, the decoder outputs at a fixed resolution
// (usually SAM_INPUT_SIZE or as specified in the model).
static void _get_decoder_mask_dims(const dt_seg_context_t *ctx, int *out_h, int *out_w)
{
  if(ctx->dec_mask_h > 0 && ctx->dec_mask_w > 0)
  {
    // Static dims from model metadata
    *out_h = ctx->dec_mask_h;
    *out_w = ctx->dec_mask_w;
  }
  else if(ctx->has_orig_im_size)
  {
    // SAM/SAM-HQ: decoder produces output at orig_im_size = encoded dims
    *out_h = ctx->encoded_height;
    *out_w = ctx->encoded_width;
  }
  else
  {
    // SAM2 with dynamic output dims: mask output matches mask_input
    // resolution (low_res_dim, typically 256x256)
    *out_h = ctx->low_res_dim;
    *out_w = ctx->low_res_dim;
  }
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
  const int lr_dim = ctx->low_res_dim;

  // Build decoder inputs: encoder outputs first, then prompt tensors
  int64_t coords_shape[3] = {1, total_points, 2};
  int64_t labels_shape[2] = {1, total_points};
  int64_t mask_in_shape[4] = {1, 1, lr_dim, lr_dim};
  int64_t has_mask_shape[1] = {1};
  int64_t orig_size_shape[1] = {2};

  dt_ai_tensor_t inputs[MAX_ENCODER_OUTPUTS + 5];
  int ni = 0;

  // Encoder outputs (reordered to match decoder input order)
  for(int i = 0; i < ctx->n_enc_outputs; i++)
  {
    const int ei = ctx->enc_order[i];
    inputs[ni++] = (dt_ai_tensor_t){
      .data = ctx->enc_data[ei], .type = DT_AI_FLOAT,
      .shape = ctx->enc_shapes[ei], .ndim = ctx->enc_ndims[ei]};
  }

  // Prompt inputs
  inputs[ni++] = (dt_ai_tensor_t){
    .data = point_coords, .type = DT_AI_FLOAT, .shape = coords_shape, .ndim = 3};
  inputs[ni++] = (dt_ai_tensor_t){
    .data = point_labels, .type = DT_AI_FLOAT, .shape = labels_shape, .ndim = 2};
  inputs[ni++] = (dt_ai_tensor_t){
    .data = ctx->low_res_masks, .type = DT_AI_FLOAT, .shape = mask_in_shape, .ndim = 4};
  inputs[ni++] = (dt_ai_tensor_t){
    .data = (void *)&has_mask, .type = DT_AI_FLOAT, .shape = has_mask_shape, .ndim = 1};

  if(ctx->has_orig_im_size)
  {
    inputs[ni++] = (dt_ai_tensor_t){
      .data = (void *)orig_im_size, .type = DT_AI_FLOAT,
      .shape = orig_size_shape, .ndim = 1};
  }

  // Decoder outputs: masks [1, N, H, W], iou [1, N], low_res [1, N, lr, lr]
  // N = ctx->num_masks (1 for single-mask models, 3-4 for multi-mask)
  const int nm = ctx->num_masks;

  // Determine decoder mask output resolution
  int dec_h, dec_w;
  _get_decoder_mask_dims(ctx, &dec_h, &dec_w);

  const size_t per_mask = (size_t)dec_h * dec_w;
  const size_t total_mask_size = (size_t)nm * per_mask;

  float *masks = g_try_malloc(total_mask_size * sizeof(float));
  if(!masks)
  {
    g_free(point_coords);
    g_free(point_labels);
    return NULL;
  }

  float iou_pred[8]; // max 8 masks
  const size_t low_res_per = (size_t)lr_dim * lr_dim;

  // low_res output only exists for SAM/SAM-HQ (not SAM2)
  float *low_res = NULL;
  if(ctx->has_low_res)
  {
    low_res = g_try_malloc((size_t)nm * low_res_per * sizeof(float));
    if(!low_res)
    {
      g_free(point_coords);
      g_free(point_labels);
      g_free(masks);
      return NULL;
    }
  }

  int64_t masks_shape[4] = {1, nm, dec_h, dec_w};
  int64_t iou_shape[2] = {1, nm};
  int64_t low_res_shape[4] = {1, nm, lr_dim, lr_dim};

  // Optional HQ output: SAM-HQ high_res_masks [1, N, 1024, 1024]
  const size_t hq_per_mask = (size_t)SAM_INPUT_SIZE * SAM_INPUT_SIZE;
  float *hq_masks = NULL;

  // Build output tensor array: masks + iou always, low_res and hq conditionally
  int num_outputs = 2; // masks + iou (minimum, SAM2)
  dt_ai_tensor_t dec_outputs[4];
  dec_outputs[0] = (dt_ai_tensor_t){
    .data = masks, .type = DT_AI_FLOAT, .shape = masks_shape, .ndim = 4};
  dec_outputs[1] = (dt_ai_tensor_t){
    .data = iou_pred, .type = DT_AI_FLOAT, .shape = iou_shape, .ndim = 2};

  if(ctx->has_low_res)
  {
    dec_outputs[num_outputs++] = (dt_ai_tensor_t){
      .data = low_res, .type = DT_AI_FLOAT, .shape = low_res_shape, .ndim = 4};
  }

  int64_t hq_shape[4] = {1, nm, SAM_INPUT_SIZE, SAM_INPUT_SIZE};
  if(ctx->has_hq)
  {
    hq_masks = g_try_malloc((size_t)nm * hq_per_mask * sizeof(float));
    if(hq_masks)
    {
      dec_outputs[num_outputs++] = (dt_ai_tensor_t){
        .data = hq_masks, .type = DT_AI_FLOAT, .shape = hq_shape, .ndim = 4};
    }
  }

  const int ret = dt_ai_run(ctx->decoder, inputs, ni, dec_outputs, num_outputs);

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

  // Cache the best low-res mask for iterative refinement (only if low_res output exists)
  if(low_res)
  {
    memcpy(ctx->low_res_masks, low_res + (size_t)best * low_res_per,
           low_res_per * sizeof(float));
    ctx->has_prev_mask = TRUE;
  }
  g_free(low_res);

  // Extract the best mask — always output at encoded image resolution
  const int final_w = ctx->encoded_width;
  const int final_h = ctx->encoded_height;
  const size_t result_size = (size_t)final_w * final_h;
  float *result = g_try_malloc(result_size * sizeof(float));
  if(!result)
  {
    g_free(hq_masks);
    g_free(masks);
    return NULL;
  }

  if(hq_masks)
  {
    // SAM-HQ path: bilinear crop+resize from 1024x1024 to encoded dims
    _crop_resize_mask(hq_masks + (size_t)best * hq_per_mask,
                      SAM_INPUT_SIZE, SAM_INPUT_SIZE,
                      result, final_w, final_h,
                      ctx->scale, TRUE);
    dt_print(DT_DEBUG_AI, "[segmentation] Using HQ mask (%dx%d -> %dx%d)",
             SAM_INPUT_SIZE, SAM_INPUT_SIZE, final_w, final_h);
    g_free(hq_masks);
  }
  else if(dec_h != final_h || dec_w != final_w)
  {
    // SAM2 path: decoder output at different resolution, crop+resize.
    // ctx->scale maps original pixels to SAM_INPUT_SIZE (1024) coords.
    // Adjust for the decoder's actual output resolution (e.g. 256).
    const float mask_scale = ctx->scale * (float)dec_h / (float)SAM_INPUT_SIZE;
    _crop_resize_mask(masks + (size_t)best * per_mask,
                      dec_w, dec_h,
                      result, final_w, final_h,
                      mask_scale, TRUE);
    dt_print(DT_DEBUG_AI, "[segmentation] Resized mask (%dx%d -> %dx%d, scale=%.4f)",
             dec_w, dec_h, final_w, final_h, mask_scale);
  }
  else
  {
    // Standard path: extract best mask at encoded dims (SAM-HQ non-HQ)
    const float *best_mask = masks + (size_t)best * per_mask;
    for(size_t i = 0; i < result_size; i++)
      result[i] = 1.0f / (1.0f + expf(-best_mask[i]));
  }

  g_free(masks);

  if(out_width)
    *out_width = final_w;
  if(out_height)
    *out_height = final_h;

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
  const int lr_dim = ctx->low_res_dim;

  int64_t coords_shape[3] = {1, 2, 2};
  int64_t labels_shape[2] = {1, 2};
  int64_t mask_in_shape[4] = {1, 1, lr_dim, lr_dim};
  int64_t has_mask_shape[1] = {1};
  int64_t orig_size_shape[1] = {2};

  // Build decoder inputs: encoder outputs + prompt tensors
  dt_ai_tensor_t inputs[MAX_ENCODER_OUTPUTS + 5];
  int ni = 0;

  for(int i = 0; i < ctx->n_enc_outputs; i++)
  {
    const int ei = ctx->enc_order[i];
    inputs[ni++] = (dt_ai_tensor_t){
      .data = ctx->enc_data[ei], .type = DT_AI_FLOAT,
      .shape = ctx->enc_shapes[ei], .ndim = ctx->enc_ndims[ei]};
  }

  inputs[ni++] = (dt_ai_tensor_t){
    .data = point_coords, .type = DT_AI_FLOAT, .shape = coords_shape, .ndim = 3};
  inputs[ni++] = (dt_ai_tensor_t){
    .data = point_labels, .type = DT_AI_FLOAT, .shape = labels_shape, .ndim = 2};
  inputs[ni++] = (dt_ai_tensor_t){
    .data = ctx->low_res_masks, .type = DT_AI_FLOAT, .shape = mask_in_shape, .ndim = 4};
  inputs[ni++] = (dt_ai_tensor_t){
    .data = (void *)&has_mask, .type = DT_AI_FLOAT, .shape = has_mask_shape, .ndim = 1};

  if(ctx->has_orig_im_size)
  {
    inputs[ni++] = (dt_ai_tensor_t){
      .data = (void *)orig_im_size, .type = DT_AI_FLOAT,
      .shape = orig_size_shape, .ndim = 1};
  }

  const int nm = ctx->num_masks;

  // Use decoder mask dims for buffer allocation
  int dec_h, dec_w;
  _get_decoder_mask_dims(ctx, &dec_h, &dec_w);
  const size_t per_mask = (size_t)dec_h * dec_w;

  float *masks = g_try_malloc((size_t)nm * per_mask * sizeof(float));
  if(!masks)
    return FALSE;

  float iou_pred[8];
  const size_t low_res_per = (size_t)lr_dim * lr_dim;

  // low_res output only exists for SAM/SAM-HQ (not SAM2)
  float *low_res = NULL;
  if(ctx->has_low_res)
  {
    low_res = g_try_malloc((size_t)nm * low_res_per * sizeof(float));
    if(!low_res)
    {
      g_free(masks);
      return FALSE;
    }
  }

  int64_t masks_shape[4] = {1, nm, dec_h, dec_w};
  int64_t iou_shape[2] = {1, nm};
  int64_t low_res_shape[4] = {1, nm, lr_dim, lr_dim};

  int num_outputs = 2; // masks + iou (minimum, SAM2)
  dt_ai_tensor_t dec_outputs[3];
  dec_outputs[0] = (dt_ai_tensor_t){
    .data = masks, .type = DT_AI_FLOAT, .shape = masks_shape, .ndim = 4};
  dec_outputs[1] = (dt_ai_tensor_t){
    .data = iou_pred, .type = DT_AI_FLOAT, .shape = iou_shape, .ndim = 2};

  if(ctx->has_low_res)
  {
    dec_outputs[num_outputs++] = (dt_ai_tensor_t){
      .data = low_res, .type = DT_AI_FLOAT, .shape = low_res_shape, .ndim = 4};
  }

  const int ret = dt_ai_run(ctx->decoder, inputs, ni, dec_outputs, num_outputs);
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
  *out_width = dec_w;
  *out_height = dec_h;
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
  if(ctx->low_res_masks)
    memset(ctx->low_res_masks, 0,
           (size_t)ctx->low_res_dim * ctx->low_res_dim * sizeof(float));
}

void dt_seg_reset_encoding(dt_seg_context_t *ctx)
{
  if(!ctx)
    return;

  for(int i = 0; i < MAX_ENCODER_OUTPUTS; i++)
  {
    g_free(ctx->enc_data[i]);
    ctx->enc_data[i] = NULL;
    ctx->enc_sizes[i] = 0;
  }

  ctx->encoded_width = 0;
  ctx->encoded_height = 0;
  ctx->scale = 0.0f;
  ctx->image_encoded = FALSE;
  ctx->has_prev_mask = FALSE;
  if(ctx->low_res_masks)
    memset(ctx->low_res_masks, 0,
           (size_t)ctx->low_res_dim * ctx->low_res_dim * sizeof(float));
}

void dt_seg_free(dt_seg_context_t *ctx)
{
  if(!ctx)
    return;

  if(ctx->encoder)
    dt_ai_unload_model(ctx->encoder);
  if(ctx->decoder)
    dt_ai_unload_model(ctx->decoder);
  for(int i = 0; i < MAX_ENCODER_OUTPUTS; i++)
    g_free(ctx->enc_data[i]);
  g_free(ctx->low_res_masks);
  g_free(ctx);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
