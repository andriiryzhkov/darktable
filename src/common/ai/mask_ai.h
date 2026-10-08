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

#include <glib.h>
#include <stdint.h>

/* the model of an AI mask type (a "mask-ai-*" task): one forward pass over
   the whole view, no prompts, a soft mask out. the contract is read from
   the model's attributes when it is opened:

     prompts           false, or absent
     strategy          "resize_whole_image": the view is stretched to the
                       square input, whatever its aspect ratio
     input_sizes       the square input's side S, the largest if several
     input_mean/_std   3 values each, applied to RGB in [0,1]; 0 and 1
                       when absent
     output_logits     TRUE when the output needs a sigmoid
     edge_refine       "guided", or absent: the mask is refined onto the
                       view by a guided filter. "none" keeps the model's
                       output as it is

   the graph has one input, [1,3,S,S] float32 or float16, and one output,
   [1,1,H,W] with a fixed H and W, checked when the session is loaded. all
   of it runs on the calling thread, which should not be the gui's */

typedef struct dt_mask_ai_t dt_mask_ai_t;

/** the model's attributes, checked against the contract, without a session
    yet. NULL when it is missing or does not fit, logged with -d ai */
dt_mask_ai_t *dt_mask_ai_open(const char *model_id);

/** the side S of the model's square input */
int dt_mask_ai_input_size(const dt_mask_ai_t *m);

/** the long side the view is rendered at, at least, for a model that
    refines its edges: the guide the mask is refined on, larger when the
    model's input needs it, never above the image's size */
#define DT_MASK_AI_GUIDE_SIZE 2048

/** TRUE unless the model's edge_refine is "none" */
gboolean dt_mask_ai_refines(const dt_mask_ai_t *m);

/** run the model once on width x height 8-bit RGB (3 bytes a pixel, the
    view to mask): the session is loaded on the configured provider, its
    graph checked, run and unloaded again. the mask, soft in [0,1], is
    *out_w x *out_h and covers the whole view: refined at the view's size
    when the model refines its edges, else at the model's output size. free
    with dt_free_align(). NULL on error, logged with -d ai */
float *dt_mask_ai_run(dt_mask_ai_t *m,
                      const uint8_t *rgb,
                      const int width,
                      const int height,
                      int *out_w,
                      int *out_h);

void dt_mask_ai_close(dt_mask_ai_t *m);

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
