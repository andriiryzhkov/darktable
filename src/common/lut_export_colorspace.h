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

G_BEGIN_DECLS

// a color space as the LUT exporter needs it: a transfer curve saying how a
// code value maps to a brightness, and primaries saying what colors the three
// channels are. both halves are needed — converting one without the other
// leaves saturated colors wrong by a fixed amount.
//
// loaded from JSON so a new camera format needs no code. the shipped set is
// <datadir>/lut_export.json, which also holds the targets referencing these
// by id. <configdir>/lut_export.json merges on top entry by entry, so a user
// adding one space writes a file holding only that entry and still tracks
// changes to the shipped ones

typedef enum dt_lut_cs_transfer_t
{
  DT_LUT_CS_TRANSFER_LINEAR = 0,  // no curve
  DT_LUT_CS_TRANSFER_GAMMA,       // pure power curve, see `exponent`
  DT_LUT_CS_TRANSFER_SRGB,        // IEC 61966-2-1 piecewise
  DT_LUT_CS_TRANSFER_REC709,      // Rec.709 OETF
  DT_LUT_CS_TRANSFER_LOG          // manufacturer log, see the log_ fields
} dt_lut_cs_transfer_t;

typedef struct dt_lut_cs_t
{
  gchar *id;    // stable key used by target definitions
  gchar *name;  // shown in the UI, translated at display time

  dt_lut_cs_transfer_t transfer;

  // DT_LUT_CS_TRANSFER_GAMMA
  float exponent;

  // DT_LUT_CS_TRANSFER_LOG: a linear toe joined to a log segment, which
  // covers V-Log, S-Log3, F-Log2, D-Log and BMD Film Gen 5.
  //
  //   encode(lin)   lin  < cut      ->  e * lin + f
  //                 else            ->  c * log10(lin + b) + d
  //   decode(code)  code < cut_code ->  (code - f) / e
  //                 else            ->  10^((code - d) / c) - b
  //
  // manufacturers publish the log branch as c*log10(a*x + b) + d; fold `a` in
  // as b/a and d + c*log10(a). C-Log (a third branch for negative input) and
  // N-Log (a cube-root toe) do not fit and need a new type, not a new field
  float log_cut, log_cut_code;
  float log_b, log_c, log_d, log_e, log_f;

  float primaries[3][2];  // xy of red, green, blue
  float whitepoint[2];    // xy
} dt_lut_cs_t;

// look up a color space by id, loading the registry on first use. returns
// NULL if no definition carries that id. the returned pointer is owned by the
// registry and stays valid until dt_lut_cs_cleanup
const dt_lut_cs_t *dt_lut_cs_find(const char *id);

// every known color space, for populating a UI. the list is owned by the
// caller (g_list_free, not free_full — the elements belong to the registry)
GList *dt_lut_cs_list(void);

// scene-linear <-> code value in [0,1]
float dt_lut_cs_encode(const dt_lut_cs_t *cs, float linear);
float dt_lut_cs_decode(const dt_lut_cs_t *cs, float code);

// drop the registry. called at shutdown; safe to call when never loaded
void dt_lut_cs_cleanup(void);

G_END_DECLS

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
