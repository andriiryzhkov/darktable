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

// Call before config is loaded to detect fresh install (no darktablerc-common yet).
// Stores the result internally for later use by dt_gui_welcome_schedule().
void dt_gui_welcome_detect_first_launch(const char *configdir);

// Schedule the welcome wizard to appear once the GTK main loop starts.
// Only shows if dt_gui_welcome_detect_first_launch() determined this is a fresh install.
// Call after dt_init() completes, before gtk_main().
void dt_gui_welcome_schedule(void);

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
