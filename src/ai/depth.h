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

#pragma once

#include "backend.h"
#include <glib.h>

/**
 * @brief Opaque depth estimation context (holds model session).
 */
typedef struct dt_depth_context_t dt_depth_context_t;

/**
 * @brief Load a depth estimation model from the model registry.
 *        Expects model.onnx in the model directory.
 * @param env AI environment (model registry).
 * @param model_id Model ID to look up in the registry.
 * @return Context, or NULL on error.
 */
dt_depth_context_t *dt_depth_load(dt_ai_environment_t *env,
                                   const char *model_id);

/**
 * @brief Compute a depth map from an RGB image.
 *        Output is a normalized [0,1] float buffer where 0=near, 1=far.
 * @param ctx Depth estimation context.
 * @param rgb_data RGB uint8 image data (HWC layout, 3 channels).
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @param out_width Set to output depth map width (same as input width).
 * @param out_height Set to output depth map height (same as input height).
 * @return Float depth buffer (width*height), caller frees with g_free().
 *         NULL on error.
 */
float *dt_depth_compute(dt_depth_context_t *ctx,
                         const uint8_t *rgb_data,
                         int width, int height,
                         int *out_width, int *out_height);

/**
 * @brief Free the depth estimation context.
 * @param ctx Context to free (NULL-safe).
 */
void dt_depth_free(dt_depth_context_t *ctx);

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
