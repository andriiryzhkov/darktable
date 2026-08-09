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

#include "common/lut_export_target.h"

#include <glib.h>

G_BEGIN_DECLS

// ask what the LUT is being made for, before asking where to put it.
//
// returns the chosen target, or NULL if the user cancelled or no usable
// target is defined. the target is owned by the registry. *grid_size
// receives the chosen sampling, which may differ from the target's own
// default — the target says what a Panasonic body expects, the user may
// still want a finer grid for an NLE.
//
// the choices are remembered in dt_conf as the next run's defaults. every
// entry point shares this so the two cannot drift apart
const dt_lut_target_t *dt_lut_export_dialog_run(int *grid_size);

G_END_DECLS

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
