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

/*
   task_contracts — structural I/O contracts for AI tasks supported by
   darktable.

   The contract describes what an ONNX model MUST look like to be
   usable for a given task ("denoise", "upscale", "mask"). Validation
   covers structural shape only (input/output counts, ranks, channel
   counts, dtype). Semantic correctness (value range, color space,
   training distribution) cannot be checked statically and is the
   responsibility of the model author.

   When adding a new AI task, add a row to the static contract table
   in task_contracts.c.
*/

#pragma once

#include <glib.h>
#include "ai/backend.h"

/**
 * @brief Structural contract for a task's expected ONNX model shape.
 *
 * - input_channels / output_channels: NCHW channel count. Use 0 to
 *   skip the check (e.g. for tasks where channels are variable).
 * - spatial_rank: total tensor rank, typically 4 (NCHW).
 * - min_inputs / max_inputs: range of accepted input tensor counts.
 *   denoise has min=1, max=2 (image, optional noise map).
 * - min_outputs / max_outputs: range of accepted output tensor counts.
 *   SAM2-style encoders produce embeddings plus multi-resolution
 *   feature maps, so mask accepts 1..4 outputs.
 */
typedef struct dt_ai_task_contract_t
{
  const char *task;
  int min_inputs;
  int max_inputs;
  int min_outputs;
  int max_outputs;
  int input_channels;
  int output_channels;
  int spatial_rank;
  dt_ai_dtype_t dtype;
} dt_ai_task_contract_t;

/**
 * @brief Look up the contract for a task by name.
 * @param task Task identifier (e.g. "denoise", "upscale", "mask").
 * @return Contract pointer (owned by the table), or NULL if the task
 *         is not supported.
 */
const dt_ai_task_contract_t *dt_ai_get_task_contract(const char *task);

/**
 * @brief Validate a loaded ONNX context against a task contract.
 *
 * Performs structural checks only:
 *   - input count within [min_inputs, max_inputs]
 *   - output count within [min_outputs, max_outputs]
 *   - output dtype == contract->dtype
 *   - output rank matches contract->spatial_rank
 *   - output channel dim matches contract->output_channels (if > 0)
 *
 * Symbolic / dynamic spatial dimensions are accepted (they're
 * resolved by dim overrides at session creation).
 *
 * @param c     Contract from dt_ai_get_task_contract(). NULL-safe (returns NULL).
 * @param ctx   Loaded model context. NULL-safe (returns error).
 * @return NULL on pass, or a newly allocated error string the caller
 *         must free with g_free().
 */
char *dt_ai_validate_against_contract(const dt_ai_task_contract_t *c,
                                      dt_ai_context_t *ctx);

/**
 * @brief Load a model and validate it against the task contract.
 *
 * Convenience wrapper around dt_ai_load_model_ext() that looks up the
 * contract for `task`, validates the loaded context, and on failure
 * unloads the model and logs a diagnostic via dt_print(DT_DEBUG_AI).
 *
 * Use this for any model load that is bound to a known task. Use the
 * raw dt_ai_load_model_ext() for special cases (e.g. the segmentation
 * decoder, which has a different shape than the encoder).
 *
 * @param env          Library environment.
 * @param task         Task identifier ("denoise", "upscale", "mask").
 *                     If no contract exists for the task, the model is
 *                     loaded without validation (caller decision).
 * @param model_id     Registry model id.
 * @param model_file   Filename within the model directory (NULL = "model.onnx").
 * @param provider     Execution provider (DT_AI_PROVIDER_CONFIGURED for user).
 * @param opt_level    Graph optimization level.
 * @param dim_overrides Symbolic dimension overrides (NULL = none).
 * @param n_overrides  Number of overrides.
 * @return Loaded context on success, NULL on load failure or
 *         contract violation.
 */
dt_ai_context_t *dt_ai_load_for_task(dt_ai_environment_t *env,
                                     const char *task,
                                     const char *model_id,
                                     const char *model_file,
                                     dt_ai_provider_t provider,
                                     dt_ai_opt_level_t opt_level,
                                     const dt_ai_dim_override_t *dim_overrides,
                                     int n_overrides);

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on