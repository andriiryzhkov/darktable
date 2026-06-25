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

#include "common/image.h"
#include <glib.h>
#include <sqlite3.h>

// embedding dimension for OpenCLIP ViT-B-32
#define DT_AI_EMBED_DIM 512

// initialize embeddings database (sqlite-vec).
// called at startup when AI is enabled
void dt_ai_embeddings_init(void);

// close embeddings database
void dt_ai_embeddings_cleanup(void);

// check if image has a computed embedding
gboolean dt_ai_embed_has(dt_imgid_t imgid);

// compute and store embedding for one image.
// uses mipmap thumbnail, resizes to 224x224, runs OpenCLIP encoder.
// returns TRUE on success
gboolean dt_ai_embed_compute(dt_imgid_t imgid);

// queue a background job to compute embeddings for a list of images
void dt_ai_embed_batch(GList *images);

// remove stored embeddings for a list of images. synchronous, no UI
void dt_ai_embed_remove(GList *images);

// retrieve stored embedding (caller must g_free).
// returns NULL if not computed. sets *dim to DT_AI_EMBED_DIM
float *dt_ai_embed_get(dt_imgid_t imgid, int *dim);

// apply auto-tags to an image based on its embedding.
// compares against model and user tag embeddings from
// embed.tag_embeddings, attaches tags above the threshold
void dt_ai_embed_auto_tag(dt_imgid_t imgid);

// recompute centroid embeddings for user-defined tags
// from their tagged indexed images
void dt_ai_embed_update_user_tags(void);

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
