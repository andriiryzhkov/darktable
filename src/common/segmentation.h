/*
 *    This file is part of darktable,
 *    Copyright (C) 2025 darktable developers.
 *
 *    darktable is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    darktable is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Opaque pointer to internal state
  typedef struct seg_context_t seg_context_t;

  typedef struct seg_point_t
  {
    float x;
    float y;
    int label;
  } seg_point_t;

  typedef struct seg_image_t
  {
    int nx;
    int ny;
    uint8_t *data; // RGB format
  } seg_image_t;

  typedef struct seg_params_t
  {
    int32_t seed;
    int32_t n_threads;
    const char *model;
    const char *fname_inp;
    const char *fname_out;
    float mask_threshold;
    float iou_threshold;
    float stability_score_threshold;
    float stability_score_offset;
    float eps;
    float eps_decoder_transformer;
    seg_point_t pt;
  } seg_params_t;

  // Initialize default parameters
  void seg_params_init(seg_params_t *params);

  // Load the model and return a context
  seg_context_t *seg_load_model(const seg_params_t *params);

  // Compute image embeddings
  bool seg_compute_image_embeddings(seg_context_t *ctx, seg_image_t *img, int n_threads);

  // Compute masks for given point
  // Returns array of masks and writes number of masks to n_masks
  seg_image_t *seg_compute_masks(seg_context_t *ctx, const seg_image_t *img, int n_threads,
                                 const seg_point_t *points, int n_points,
                                 int *n_masks, int mask_on_val, int mask_off_val);

  // Free a mask array returned by sam_compute_masks
  void seg_free_masks(seg_image_t *masks, int n_masks);

  // Free the context and associated resources
  void seg_free(seg_context_t *ctx);

  // Get timing information
  void seg_get_timings(seg_context_t *ctx, int *t_load_ms, int *t_compute_img_ms, int *t_compute_masks_ms);

#ifdef __cplusplus
}
#endif

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
