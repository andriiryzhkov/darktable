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

#include <gdk/gdk.h>
#include <gtk/gtk.h>
#include <webview/api.h>

#define RESIZE_BORDER 4

static GtkWindow *g_window = NULL;

// edge resize on mouse motion near window borders
static gboolean _on_motion(GtkWidget *widget, GdkEventMotion *event, gpointer data)
{
  (void)data;
  int w = gtk_widget_get_allocated_width(widget);
  int h = gtk_widget_get_allocated_height(widget);
  double x = event->x;
  double y = event->y;

  GdkCursorType cursor = GDK_LEFT_PTR;
  if(y < RESIZE_BORDER)
  {
    if(x < RESIZE_BORDER) cursor = GDK_TOP_LEFT_CORNER;
    else if(x > w - RESIZE_BORDER) cursor = GDK_TOP_RIGHT_CORNER;
    else cursor = GDK_TOP_SIDE;
  }
  else if(y > h - RESIZE_BORDER)
  {
    if(x < RESIZE_BORDER) cursor = GDK_BOTTOM_LEFT_CORNER;
    else if(x > w - RESIZE_BORDER) cursor = GDK_BOTTOM_RIGHT_CORNER;
    else cursor = GDK_BOTTOM_SIDE;
  }
  else if(x < RESIZE_BORDER)
  {
    cursor = GDK_LEFT_SIDE;
  }
  else if(x > w - RESIZE_BORDER)
  {
    cursor = GDK_RIGHT_SIDE;
  }

  GdkWindow *gdk_win = gtk_widget_get_window(widget);
  if(cursor != GDK_LEFT_PTR)
  {
    GdkCursor *c = gdk_cursor_new_for_display(gdk_display_get_default(), cursor);
    gdk_window_set_cursor(gdk_win, c);
    g_object_unref(c);
  }
  else
  {
    gdk_window_set_cursor(gdk_win, NULL);
  }

  return FALSE;
}

// start resize on button press near edges
static gboolean _on_button_press(GtkWidget *widget, GdkEventButton *event, gpointer data)
{
  (void)data;
  if(event->button != 1) return FALSE;

  int w = gtk_widget_get_allocated_width(widget);
  int h = gtk_widget_get_allocated_height(widget);
  double x = event->x;
  double y = event->y;

  GdkWindowEdge edge = -1;
  if(y < RESIZE_BORDER)
  {
    if(x < RESIZE_BORDER) edge = GDK_WINDOW_EDGE_NORTH_WEST;
    else if(x > w - RESIZE_BORDER) edge = GDK_WINDOW_EDGE_NORTH_EAST;
    else edge = GDK_WINDOW_EDGE_NORTH;
  }
  else if(y > h - RESIZE_BORDER)
  {
    if(x < RESIZE_BORDER) edge = GDK_WINDOW_EDGE_SOUTH_WEST;
    else if(x > w - RESIZE_BORDER) edge = GDK_WINDOW_EDGE_SOUTH_EAST;
    else edge = GDK_WINDOW_EDGE_SOUTH;
  }
  else if(x < RESIZE_BORDER)
  {
    edge = GDK_WINDOW_EDGE_WEST;
  }
  else if(x > w - RESIZE_BORDER)
  {
    edge = GDK_WINDOW_EDGE_EAST;
  }

  if(edge != (GdkWindowEdge)-1)
  {
    gtk_window_begin_resize_drag(g_window, edge, event->button,
                                 (gint)event->x_root, (gint)event->y_root,
                                 event->time);
    return TRUE;
  }

  return FALSE;
}

void dt_titlebar_init(webview_t w)
{
  g_window = (GtkWindow *)webview_get_native_handle(
      w, WEBVIEW_NATIVE_HANDLE_KIND_UI_WINDOW);

  gtk_window_set_decorated(g_window, FALSE);

  // add edge resize handlers since decorations are removed
  GtkWidget *widget = GTK_WIDGET(g_window);
  gtk_widget_add_events(widget, GDK_POINTER_MOTION_MASK | GDK_BUTTON_PRESS_MASK);
  g_signal_connect(widget, "motion-notify-event", G_CALLBACK(_on_motion), NULL);
  g_signal_connect(widget, "button-press-event", G_CALLBACK(_on_button_press), NULL);
}

void dt_titlebar_start_drag(webview_t w)
{
  (void)w;
  GdkEvent *event = gtk_get_current_event();
  if(event)
  {
    gdouble x_root, y_root;
    gdk_event_get_root_coords(event, &x_root, &y_root);
    gtk_window_begin_move_drag(g_window, 1,
                               (gint)x_root, (gint)y_root,
                               gdk_event_get_time(event));
    gdk_event_free(event);
  }
}

void dt_titlebar_zoom(webview_t w)
{
  (void)w;
  if(gtk_window_is_maximized(g_window))
    gtk_window_unmaximize(g_window);
  else
    gtk_window_maximize(g_window);
}
