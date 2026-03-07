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

#include <gtk/gtk.h>
#include <webview/api.h>

static GtkWindow *g_window = NULL;

void dt_titlebar_init(webview_t w)
{
  g_window = (GtkWindow *)webview_get_native_handle(
      w, WEBVIEW_NATIVE_HANDLE_KIND_UI_WINDOW);

  // Keep native GNOME titlebar with standard window controls.
  // Custom CSD titlebar can be revisited later.
}

void dt_titlebar_start_drag(webview_t w)
{
  (void)w;
}

void dt_titlebar_zoom(webview_t w)
{
  (void)w;
  if(gtk_window_is_maximized(g_window))
    gtk_window_unmaximize(g_window);
  else
    gtk_window_maximize(g_window);
}
