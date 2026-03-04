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

#include <webview/api.h>

// remove native OS titlebar chrome, keeping native window controls
// must be called after webview_create() and before webview_run()
void dt_titlebar_init(webview_t w);

// initiate window drag from a mousedown event
void dt_titlebar_start_drag(webview_t w);

// toggle zoom/maximize (respects system preferences on macOS)
void dt_titlebar_zoom(webview_t w);
