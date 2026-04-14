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
   task_contracts — implementation of the structural I/O contract
   table for AI tasks supported by darktable.

   Each contract describes the ONNX I/O shape a model must expose to
   be usable for a given task. Validation is structural only: input
   and output counts, dtype, tensor rank, and (when fixed) channel
   count. Symbolic/dynamic spatial dimensions are accepted because
   they are resolved by dim overrides at session creation.

   What this catches:
     - Models with the wrong number of inputs/outputs (would crash
       or read garbage)
     - Models with unexpected dtype (e.g. quantized UINT8) that
       the consumer code is not prepared to handle
     - Models with the wrong tensor rank (would write past output
       buffers)
     - Models with unexpected channel count (would silently produce
       wrong colors)

   What this does NOT catch (cannot be done statically):
     - Wrong value range (model expects [-1,1] vs [0,1])
     - Wrong color space / primaries / gamma
     - Wrong training distribution (cartoons vs photos)
     - Whether the model actually denoises / upscales correctly

   Validation is invoked by:
     - common/ai/restore.c      — denoise and upscale tasks
     - common/ai/segmentation.c — mask task (encoder)

   To add a new task: append a row to _contracts[] below and document
   the I/O contract.
*/

#include "common/ai/task_contracts.h"
#include "common/darktable.h"
#include <string.h>

// Static contract table — single source of truth for which tasks
// darktable supports and what structural shape their models must have.
// Consumers look up entries via dt_ai_get_task_contract(task_name).
//
// Column meaning (matches dt_ai_task_contract_t in the header):
//
//   task        — task identifier string used by the model registry
//                 and consumer modules; must be unique
//   in_min      — minimum number of input tensors the model must accept
//   in_max      — maximum number of input tensors the model may accept
//   out_min     — minimum number of output tensors
//   out_max     — maximum number of output tensors
//                 (SAM2-style encoders produce several outputs)
//   in_ch       — required input channel count (NCHW dim 1);
//                 use 0 to skip the check (variable / not enforced)
//   out_ch      — required output channel count (NCHW dim 1);
//                 use 0 to skip the check
//   rank        — required tensor rank for inputs and outputs;
//                 typically 4 for NCHW image tensors
//   dtype       — required input/output element type (DT_AI_FLOAT etc.)
//
// To add a new task:
//   1. Pick a stable lowercase task identifier (e.g. "denoise")
//   2. Add a row below with the structural I/O the consumer code
//      expects. Use 0 for any channel count that is not fixed
//   3. Add the task identifier to the consumer module that drives it
//      (e.g. a new file under common/ai/) and call
//      dt_ai_validate_against_contract() right after dt_ai_load_model
//      to enforce it
//   4. Document the I/O contract in this comment block so future model
//      authors know what shape to ship
static const dt_ai_task_contract_t _contracts[] = {
  // task        in_min in_max out_min out_max in_ch out_ch rank dtype
  { "denoise",   1,     2,     1,      1,      3,    3,     4,   DT_AI_FLOAT },
  { "upscale",   1,     1,     1,      1,      3,    3,     4,   DT_AI_FLOAT },
  // mask: SAM2-style encoders produce image embeddings plus 1-3
  // multi-resolution feature maps, so the output count is a range.
  // out_ch = 0 skips the channel check (varies by backbone).
  // decoder is validated separately by the segmentation module.
  { "mask",      1,     1,     1,      4,      3,    0,     4,   DT_AI_FLOAT },
};

const dt_ai_task_contract_t *dt_ai_get_task_contract(const char *task)
{
  if(!task) return NULL;
  for(size_t i = 0; i < sizeof(_contracts) / sizeof(_contracts[0]); i++)
    if(strcmp(_contracts[i].task, task) == 0)
      return &_contracts[i];
  return NULL;
}

char *dt_ai_validate_against_contract(const dt_ai_task_contract_t *c,
                                      dt_ai_context_t *ctx)
{
  if(!c) return NULL;  // unknown task — caller decides whether to allow
  if(!ctx) return g_strdup("model context is NULL");

  // input count
  const int n_in = dt_ai_get_input_count(ctx);
  if(n_in < c->min_inputs || n_in > c->max_inputs)
    return g_strdup_printf("task '%s' expects %d-%d inputs, model has %d",
                           c->task, c->min_inputs, c->max_inputs, n_in);

  // output count
  const int n_out = dt_ai_get_output_count(ctx);
  if(n_out < c->min_outputs || n_out > c->max_outputs)
    return g_strdup_printf("task '%s' expects %d-%d outputs, model has %d",
                           c->task, c->min_outputs, c->max_outputs, n_out);

  // output dtype
  const dt_ai_dtype_t out_type = dt_ai_get_output_type(ctx, 0);
  if(out_type != c->dtype)
    return g_strdup_printf("task '%s' expects output dtype %d, model has %d",
                           c->task, (int)c->dtype, (int)out_type);

  // output rank and channel count
  int64_t shape[8] = { 0 };
  const int ndim = dt_ai_get_output_shape(ctx, 0, shape, 8);
  if(ndim < 0)
    return g_strdup_printf("task '%s': could not read output shape", c->task);
  if(ndim != c->spatial_rank)
    return g_strdup_printf("task '%s' expects rank-%d output, model has rank %d",
                           c->task, c->spatial_rank, ndim);
  // channel dim is at index 1 for NCHW. only enforce if contract sets it
  // (>0); negative shape values mean dynamic/symbolic and are accepted.
  if(c->output_channels > 0 && shape[1] > 0
     && shape[1] != (int64_t)c->output_channels)
    return g_strdup_printf("task '%s' expects %d output channels, model has %" G_GINT64_FORMAT,
                           c->task, c->output_channels, shape[1]);

  return NULL;
}

dt_ai_context_t *dt_ai_load_for_task(dt_ai_environment_t *env,
                                     const char *task,
                                     const char *model_id,
                                     const char *model_file,
                                     dt_ai_provider_t provider,
                                     dt_ai_opt_level_t opt_level,
                                     const dt_ai_dim_override_t *dim_overrides,
                                     int n_overrides)
{
  dt_ai_context_t *ctx = dt_ai_load_model_ext(env, model_id, model_file,
                                              provider, opt_level,
                                              dim_overrides, n_overrides);
  if(!ctx) return NULL;

  const dt_ai_task_contract_t *contract = dt_ai_get_task_contract(task);
  char *err = dt_ai_validate_against_contract(contract, ctx);
  if(err)
  {
    dt_print(DT_DEBUG_AI,
             "[ai] model '%s' is not compatible with task '%s': %s",
             model_id ? model_id : "(null)", task ? task : "(null)", err);
    g_free(err);
    dt_ai_unload_model(ctx);
    return NULL;
  }
  return ctx;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on