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

// --- window control button callbacks ---

static void _on_close_clicked(GtkButton *btn, gpointer data)
{
  (void)btn;
  gtk_window_close((GtkWindow *)data);
}

static void _on_minimize_clicked(GtkButton *btn, gpointer data)
{
  (void)btn;
  gtk_window_iconify((GtkWindow *)data);
}

static void _on_maximize_clicked(GtkButton *btn, gpointer data)
{
  (void)btn;
  GtkWindow *win = (GtkWindow *)data;
  if(gtk_window_is_maximized(win))
    gtk_window_unmaximize(win);
  else
    gtk_window_maximize(win);
}

static GtkWidget *_make_window_button(const char *icon_name,
                                      const char *css_class,
                                      GCallback callback,
                                      gpointer data)
{
  GtkWidget *btn = gtk_button_new_from_icon_name(icon_name, GTK_ICON_SIZE_MENU);
  gtk_widget_set_focus_on_click(btn, FALSE);
  gtk_widget_set_can_focus(btn, FALSE);
  // use native GNOME theme styling for window buttons
  GtkStyleContext *ctx = gtk_widget_get_style_context(btn);
  gtk_style_context_add_class(ctx, "titlebutton");
  gtk_style_context_add_class(ctx, css_class);
  g_signal_connect(btn, "clicked", callback, data);
  return btn;
}

// Poll until the webview library has added the WebKitWebView, then reparent.
static gboolean _inject_overlay(gpointer data)
{
  (void)data;
  GtkWidget *window = GTK_WIDGET(g_window);

  // find the existing child (WebKitWebView) added by the webview library
  GList *children = gtk_container_get_children(GTK_CONTAINER(window));
  if(!children) return G_SOURCE_CONTINUE; // not added yet, try again

  GtkWidget *webview_widget = (GtkWidget *)children->data;
  g_list_free(children);

  // reparent: remove webview from window, wrap in overlay
  g_object_ref(webview_widget);
  gtk_container_remove(GTK_CONTAINER(window), webview_widget);

  GtkWidget *overlay = gtk_overlay_new();
  gtk_container_add(GTK_CONTAINER(window), overlay);

  // webview as main child — fills entire window including top edge
  gtk_container_add(GTK_CONTAINER(overlay), webview_widget);
  g_object_unref(webview_widget);

  // wrapper matches the 52px headerbar height, buttons centered within
  GtkWidget *wrapper = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
  gtk_widget_set_halign(wrapper, GTK_ALIGN_END);
  gtk_widget_set_valign(wrapper, GTK_ALIGN_START);
  gtk_widget_set_size_request(wrapper, -1, 52);
  gtk_widget_set_margin_end(wrapper, 12);

  GtkWidget *box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  gtk_style_context_add_class(gtk_widget_get_style_context(box), "windowcontrols");
  gtk_widget_set_valign(box, GTK_ALIGN_CENTER);
  gtk_box_pack_start(GTK_BOX(wrapper), box, TRUE, FALSE, 0);

  gtk_box_pack_start(GTK_BOX(box),
    _make_window_button("window-minimize-symbolic", "minimize",
                        G_CALLBACK(_on_minimize_clicked), g_window), FALSE, FALSE, 0);
  gtk_box_pack_start(GTK_BOX(box),
    _make_window_button("window-maximize-symbolic", "maximize",
                        G_CALLBACK(_on_maximize_clicked), g_window), FALSE, FALSE, 0);
  gtk_box_pack_start(GTK_BOX(box),
    _make_window_button("window-close-symbolic", "close",
                        G_CALLBACK(_on_close_clicked), g_window), FALSE, FALSE, 0);

  gtk_overlay_add_overlay(GTK_OVERLAY(overlay), wrapper);
  gtk_overlay_set_overlay_pass_through(GTK_OVERLAY(overlay), wrapper, FALSE);
  gtk_widget_show_all(overlay);

  return G_SOURCE_REMOVE; // done, don't call again
}

void dt_titlebar_init(webview_t w)
{
  g_window = (GtkWindow *)webview_get_native_handle(
      w, WEBVIEW_NATIVE_HANDLE_KIND_UI_WINDOW);

  // Remove native decorations — webview extends to top edge.
  // An idle callback will inject an overlay with window control
  // buttons once the webview library adds the WebKitWebView.
  gtk_window_set_decorated(g_window, FALSE);
  g_idle_add(_inject_overlay, NULL);
}

void dt_titlebar_start_drag(webview_t w)
{
  (void)w;

  // When called from a JS binding, gtk_get_current_event() is NULL
  // because WebKit consumed the mouse event. Query pointer directly.
  GdkDisplay *display = gdk_display_get_default();
  GdkSeat *seat = gdk_display_get_default_seat(display);
  GdkDevice *pointer = gdk_seat_get_pointer(seat);
  gint x, y;
  gdk_device_get_position(pointer, NULL, &x, &y);
  gtk_window_begin_move_drag(g_window, 1, x, y, GDK_CURRENT_TIME);
}

void dt_titlebar_zoom(webview_t w)
{
  (void)w;
  if(gtk_window_is_maximized(g_window))
    gtk_window_unmaximize(g_window);
  else
    gtk_window_maximize(g_window);
}
