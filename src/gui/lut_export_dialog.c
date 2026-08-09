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

#include "gui/lut_export_dialog.h"
#include "bauhaus/bauhaus.h"
#include "common/darktable.h"
#include "common/lut_export_colorspace.h"
#include "control/conf.h"
#include "control/control.h"
#include "gui/gtk.h"

#ifdef GDK_WINDOWING_QUARTZ
#include "osx/osx.h"
#endif

#include <gtk/gtk.h>

#define CONF_TARGET "plugins/lut_export/target"
#define CONF_GRID "plugins/lut_export/grid_size"

// 17 is coarse enough to show banding on a gradient, 65 is four times the
// data for detail no camera LUT engine resolves. 33 is what the .cube
// consumers in cameras and NLEs all expect
static const int _grid_sizes[] = { 17, 33, 65 };

typedef struct _dialog_t
{
  GtkWidget *target;
  GtkWidget *grid;
  GtkWidget *input, *output;   // the color spaces it implies, read-only
  GList *targets;              // registry-owned entries, in combo order
} _dialog_t;

// derived from the target rather than chosen. no cursor and no focus says
// that without dimming the text, which the user still has to read
static GtkWidget *_readonly_field(void)
{
  GtkWidget *entry = dt_ui_entry_new(24);
  gtk_editable_set_editable(GTK_EDITABLE(entry), FALSE);
  gtk_widget_set_can_focus(entry, FALSE);
  return entry;
}

static const dt_lut_target_t *_selected_target(const _dialog_t *d)
{
  const int row = dt_bauhaus_combobox_get(d->target);
  return (row < 0) ? NULL : g_list_nth_data(d->targets, row);
}

// one label column, one widget column, as preferences.c:343 lays them out
static void _attach_row(GtkWidget *grid,
                        int *line,
                        const char *text,
                        GtkWidget *widget)
{
  GtkWidget *label = dt_ui_label_new(text);

  // fill one shared column so the controls end at the same edge
  gtk_widget_set_hexpand(widget, TRUE);
  gtk_widget_set_halign(widget, GTK_ALIGN_FILL);

  gtk_grid_attach(GTK_GRID(grid), label, 0, (*line)++, 1, 1);
  gtk_grid_attach_next_to(GTK_GRID(grid), widget, label, GTK_POS_RIGHT, 1, 1);
}

// enter accepts. dt_handle_dialog_enter's job, but via an event controller
// rather than the "key-press-event" signal GTK4 drops
static gboolean _key_pressed(GtkEventControllerKey *controller,
                             const guint keyval,
                             const guint keycode,
                             const GdkModifierType state,
                             gpointer data)
{
  if(keyval != GDK_KEY_Return && keyval != GDK_KEY_KP_Enter) return FALSE;

  gtk_dialog_response(GTK_DIALOG(dt_gui_get_widget(controller)),
                      GTK_RESPONSE_ACCEPT);
  return TRUE;
}

// the input space is what the user has to match in camera; getting it wrong
// gives a plausible-looking but wrong result, so show it rather than imply it
static void _update_detail(GtkWidget *combo, _dialog_t *d)
{
  const dt_lut_target_t *t = _selected_target(d);

  if(!t)
  {
    gtk_entry_set_text(GTK_ENTRY(d->input), "");
    gtk_entry_set_text(GTK_ENTRY(d->output), "");
    return;
  }

  // a tooltip, not a visible block: it restates the fields below and was the
  // only widget that made the dialog change height
  gtk_widget_set_tooltip_text(d->target, t->description);

  // fall back to the id: a key they can grep for beats an empty field
  const dt_lut_cs_t *in = dt_lut_cs_find(t->input_cs);
  const dt_lut_cs_t *out = dt_lut_cs_find(t->output_cs);

  gtk_entry_set_text(GTK_ENTRY(d->input),
                     in && in->name ? in->name : t->input_cs);
  gtk_entry_set_text(GTK_ENTRY(d->output),
                     out && out->name ? out->name : t->output_cs);
}

const dt_lut_target_t *dt_lut_export_dialog_run(int *grid_size)
{
  _dialog_t d = { 0 };
  d.targets = dt_lut_target_list();

  if(!d.targets)
  {
    dt_control_log(_("no usable LUT target is defined"));
    return NULL;
  }

  GtkWidget *win = dt_ui_main_window(darktable.gui->ui);
  GtkWidget *dialog = gtk_dialog_new_with_buttons
    (_("export LUT"), GTK_WINDOW(win), GTK_DIALOG_DESTROY_WITH_PARENT,
     _("_cancel"), GTK_RESPONSE_NONE,
     _("_export"), GTK_RESPONSE_ACCEPT, NULL);

  // adds the "?" to the start of the button box and points it at the manual
  dt_gui_dialog_add_help(GTK_DIALOG(dialog), "lut_export_dialog");

  gtk_dialog_set_default_response(GTK_DIALOG(dialog), GTK_RESPONSE_ACCEPT);
  dt_gui_connect_key(dialog, _key_pressed, (gpointer)NULL);

  // bauhaus draws no background of its own; the theme supplies one per
  // container, so the content area needs a name (as preferences.c:600 does)
  gtk_widget_set_name(gtk_dialog_get_content_area(GTK_DIALOG(dialog)),
                      "lut-export-box");
#ifdef GDK_WINDOWING_QUARTZ
  dt_osx_disallow_fullscreen(dialog);
#endif

  d.target = dt_bauhaus_combobox_new(NULL);
  dt_bauhaus_combobox_set_selected_text_align(d.target,
                                              DT_BAUHAUS_COMBOBOX_ALIGN_LEFT);

  const gchar *want = dt_conf_get_string_const(CONF_TARGET);
  int row = 0, active = 0;

  for(GList *l = d.targets; l; l = g_list_next(l), row++)
  {
    const dt_lut_target_t *t = l->data;
    dt_bauhaus_combobox_add_aligned(d.target, t->name,
                                    DT_BAUHAUS_COMBOBOX_ALIGN_LEFT);
    if(want && !g_strcmp0(want, t->id)) active = row;
  }
  dt_bauhaus_combobox_set(d.target, active);

  d.grid = dt_bauhaus_combobox_new(NULL);
  dt_bauhaus_combobox_set_selected_text_align(d.grid,
                                              DT_BAUHAUS_COMBOBOX_ALIGN_LEFT);

  const int want_grid = dt_conf_get_int(CONF_GRID);
  int active_grid = 1;  // 33

  for(int i = 0; i < (int)G_N_ELEMENTS(_grid_sizes); i++)
  {
    gchar *label = g_strdup_printf("%d³", _grid_sizes[i]);
    dt_bauhaus_combobox_add_aligned(d.grid, label,
                                    DT_BAUHAUS_COMBOBOX_ALIGN_LEFT);
    g_free(label);
    if(_grid_sizes[i] == want_grid) active_grid = i;
  }
  dt_bauhaus_combobox_set(d.grid, active_grid);

  d.input = _readonly_field();
  d.output = _readonly_field();

  g_signal_connect(G_OBJECT(d.target), "value-changed",
                   G_CALLBACK(_update_detail), &d);
  _update_detail(d.target, &d);

  GtkWidget *layout = gtk_grid_new();
  gtk_grid_set_row_spacing(GTK_GRID(layout), DT_PIXEL_APPLY_DPI(3));
  gtk_grid_set_column_spacing(GTK_GRID(layout), DT_PIXEL_APPLY_DPI(10));
  gtk_widget_set_valign(layout, GTK_ALIGN_START);
  gtk_widget_set_margin_start(layout, DT_PIXEL_APPLY_DPI(8));
  gtk_widget_set_margin_end(layout, DT_PIXEL_APPLY_DPI(8));
  gtk_widget_set_margin_top(layout, DT_PIXEL_APPLY_DPI(8));
  gtk_widget_set_margin_bottom(layout, DT_PIXEL_APPLY_DPI(8));

  // no group separators: the read-only fields already read differently
  int line = 0;
  _attach_row(layout, &line, _("target"), d.target);
  _attach_row(layout, &line, _("input"), d.input);
  _attach_row(layout, &line, _("output"), d.output);
  _attach_row(layout, &line, _("grid size"), d.grid);

  dt_gui_dialog_add(dialog, layout);
  gtk_widget_show_all(dialog);

  const dt_lut_target_t *chosen = NULL;

  if(gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT)
  {
    chosen = _selected_target(&d);

    const int g = dt_bauhaus_combobox_get(d.grid);
    const int grid = (g >= 0 && g < (int)G_N_ELEMENTS(_grid_sizes))
      ? _grid_sizes[g]
      : 33;

    if(grid_size) *grid_size = grid;

    // remembered as the next run's defaults
    if(chosen) dt_conf_set_string(CONF_TARGET, chosen->id);
    dt_conf_set_int(CONF_GRID, grid);
  }

  gtk_widget_destroy(dialog);
  g_list_free(d.targets);
  return chosen;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
