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

#include "common/darktable.h"

#include <glib.h>
#include <stddef.h>
#include <stdint.h>

G_BEGIN_DECLS

// output file format for the generated LUT
typedef enum dt_lut_export_format_t
{
  DT_LUT_EXPORT_FORMAT_CUBE = 0,     // Adobe/Autodesk .cube text
  DT_LUT_EXPORT_FORMAT_HALD_PNG = 1  // hald-CLUT PNG (16-bit)
} dt_lut_export_format_t;

// what the exporter had to do to the style to make it fit a LUT, and how
// well the result fills the output domain. every field is either NULL or a
// newly allocated human-readable string; free with dt_lut_export_report_free
typedef struct dt_lut_export_report_t
{
  gchar *skipped;   // modules dropped as not LUT-representable
  gchar *adjusted;  // modules baked with their spatial options switched off
  gchar *range;     // how badly the style fits the target's input range:
                    // clipping, shoulder compression, and the cause where known
} dt_lut_export_report_t;

// free the strings held by a report and zero it. the report itself is owned
// by the caller (typically a stack variable) and is not freed
void dt_lut_export_report_free(dt_lut_export_report_t *report);

// write a rendered grid out and tell the user what happened, including what
// the exporter had to leave out. rgb may be NULL, meaning the render itself
// failed — the report is still worth showing, since it usually says why.
// every UI entry point goes through this so the wording cannot drift.
// title labels the messages and becomes the .cube TITLE
gboolean dt_lut_export_write_and_report(const float *rgb,
                                        int grid_size,
                                        const char *title,
                                        const char *filepath,
                                        const dt_lut_export_report_t *report);

// write an in-memory float RGB grid_size^3 buffer to a .cube file. pixels
// are expected in the output color space already (linear color values in
// [0,1]). returns TRUE on success. overwrites destination if it exists
gboolean dt_lut_export_write_cube(const float *rgb,
                                  int grid_size,
                                  const char *title,
                                  const char *filepath);

// run an identity hald through DT's pixelpipe with the given style applied.
// returns a g_malloc'd buffer of grid_size^3 pixels, 3 floats each, tightly
// packed, b varying fastest; caller frees with g_free. NULL on any failure,
// including an unknown color space id.
//
// input_cs and output_cs are registry ids (see lut_export_colorspace.h);
// both the transfer curve and the primaries are honoured.
//
// the style replaces the pipe's history outright and colorin / colorout are
// pinned, so the result carries the style's color transform and no baseline
// processing. items a LUT cannot represent are dropped, and one that is
// per-pixel apart from a spatially-acting option has that option switched
// off instead. a style with nothing representable yields NULL rather than an
// identity LUT.
//
// report, if non-NULL, receives what was dropped, adjusted and clipped; free
// it with dt_lut_export_report_free.
//
// no image need be selected or open in darkroom
float *dt_lut_export_render_style(const char *style_name,
                                  const char *input_cs,
                                  const char *output_cs,
                                  int grid_size,
                                  dt_lut_export_report_t *report);

// as above, but bakes an image's current edit rather than a saved style.
// the darkroom's in-memory history is flushed to the database first, so an
// edit still open and unsaved exports correctly.
//
// this is the source that matches a log target by construction: an edit made
// on a log frame already has its tone mapping set for that frame's range,
// where a style authored against an ordinary raw generally does not
float *dt_lut_export_render_image(dt_imgid_t imgid,
                                  const char *input_cs,
                                  const char *output_cs,
                                  int grid_size,
                                  dt_lut_export_report_t *report);

G_END_DECLS

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
