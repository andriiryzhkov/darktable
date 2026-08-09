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

#include "common/lut_export.h"

#include <glib.h>

G_BEGIN_DECLS

// what the LUT is being made for: which color space it will be fed, which
// one it must produce, and how finely to sample. one target is one entry in
// the export dialog's dropdown.
//
// targets live in the `targets` array of lut_export.json beside the color
// spaces they reference by id, and merge from the user's copy the same way.
// adding a camera that uses an existing log format is therefore a data
// change with no code at all
typedef struct dt_lut_target_t
{
  gchar *id;
  gchar *name;         // shown in the UI
  gchar *description;  // optional; a tooltip's worth of context

  gchar *input_cs;   // color space id the LUT's input axis is in
  gchar *output_cs;  // color space id its output must be in

  int grid_size;                  // N in NxNxN, typically 17 / 33 / 65
  dt_lut_export_format_t format;  // container to write
} dt_lut_target_t;

// look up a target by id, loading the registry on first use. returns NULL if
// no definition carries that id, or if it names a color space that does not
// exist — a target that cannot be rendered is worse than no target. the
// returned pointer is owned by the registry
const dt_lut_target_t *dt_lut_target_find(const char *id);

// every usable target, for populating a UI, sorted by name. the list is
// owned by the caller (g_list_free — the elements belong to the registry)
GList *dt_lut_target_list(void);

// drop the registry. called at shutdown; safe to call when never loaded
void dt_lut_target_cleanup(void);

G_END_DECLS

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
