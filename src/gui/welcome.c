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

#include "gui/welcome.h"
#include "gui/gtk.h"
#include "control/conf.h"
#include "common/darktable.h"

extern const char darktable_package_version[];
#ifdef GDK_WINDOWING_QUARTZ
#include "osx/osx.h"
#endif

#define WELCOME_WIDTH 900
#define WELCOME_HEIGHT 400
#define LOGO_SIZE 96
#define MODULE_SEP "\xc2\xb7"  // UTF-8 middle dot ·

// CSS injected while the welcome dialog is visible, removed on close
static const char _welcome_css[] =
  "#welcome-dialog {"
  "  font-size: 12pt;"
  "  background-color: @bg_color;"
  "  border: none;"
  "}"
  "#welcome-dialog > box {"
  "  background-color: @bg_color;"
  "  border: none;"
  "}"
  "#welcome-dialog > box > buttonbox {"
  "  background-color: shade(@bg_color, 0.85);"
  "  padding: 8px 16px;"
  "}"
  "#welcome-dialog > box > buttonbox button {"
  "  background-color: @button_bg;"
  "  border: 1px solid shade(@button_bg, 1.1);"
  "  padding: 4px 16px;"
  "}"
  "#welcome-card {"
  "  background-color: @plugin_bg_color;"
  "  border: 2px solid shade(@plugin_bg_color, 1.15);"
  "  border-radius: 6px;"
  "  padding: 0;"
  "}"
  "#welcome-card.welcome-card-selected {"
  "  border-color: @selected_bg_color;"
  "  background-color: @collapsible_bg_color;"
  "}"
  "#welcome-card-badge {"
  "  background-color: shade(@plugin_bg_color, 0.75);"
  "  color: @fg_color;"
  "  border-radius: 0 4px 0 4px;"
  "  padding: 2px 10px;"
  "  font-size: 9pt;"
  "}"
  "#welcome-tip {"
  "  background-color: @plugin_bg_color;"
  "  border-radius: 4px;"
  "  border-left: 3px solid @selected_bg_color;"
  "}"
  "#welcome-skip {"
  "  background: none;"
  "  border: none;"
  "  box-shadow: none;"
  "  opacity: 0.5;"
  "  font-size: 10pt;"
  "}"
  "#welcome-skip:hover {"
  "  opacity: 0.8;"
  "}"
  "#welcome-panel {"
  "  background-color: shade(@bg_color, 0.85);"
  "  border: 1px solid @plugin_bg_color;"
  "  border-radius: 6px;"
  "  padding: 0;"
  "}"
  "#welcome-panel-dt {"
  "  background-color: shade(@bg_color, 0.85);"
  "  border: 1px solid @selected_bg_color;"
  "  border-radius: 6px;"
  "  padding: 0;"
  "}"
  "#welcome-stage {"
  "  background-color: @bg_color;"
  "  border: 1px solid @plugin_bg_color;"
  "  border-radius: 4px;"
  "  padding: 0;"
  "}"
  "#welcome-stage-warn {"
  "  background-color: alpha(@fg_color, 0.06);"
  "  border: 1px solid alpha(@fg_color, 0.15);"
  "  border-radius: 4px;"
  "  padding: 0;"
  "}"
  "#welcome-info {"
  "  background-color: @bg_color;"
  "  border: none;"
  "  border-radius: 4px;"
  "  padding: 0;"
  "}"
  "#welcome-step-num {"
  "  background-color: @selected_bg_color;"
  "  border-radius: 14px;"
  "  padding: 2px;"
  "  border: none;"
  "}"
  "#welcome-module-tag {"
  "  border: 1px solid @disabled_fg_color;"
  "  border-radius: 3px;"
  "  padding: 1px 6px;"
  "  font-size: 8pt;"
  "}";

static const char *_theme_names[] = {
  "darktable-elegant-grey",
  "darktable-elegant-dark",
  "darktable-elegant-darker"
};
static const int _num_themes = sizeof(_theme_names) / sizeof(_theme_names[0]);

// set by dt_gui_welcome_detect_first_launch()
static gboolean _is_first_launch = FALSE;

// page identifiers
enum {
  PAGE_LANDING,
  PAGE_THEME,
  PAGE_MODULES,
  PAGE_IMPORT,
  PAGE_QUICKSTART,
  PAGE_FIRSTEDIT,
  PAGE_RESOURCES,
  PAGE_COUNT
};

static const char *_page_stack_names[] = {
  "landing", "theme", "modules", "import", "quickstart", "firstedit", "resources"
};

// step labels for breadcrumb (landing page has no step)
static const char * const _step_labels[] = {
  N_("appearance"), N_("modules"), N_("library"), N_("quick start"),
  N_("first edit"), N_("resources")
};
static const int _num_steps = sizeof(_step_labels) / sizeof(_step_labels[0]);

// maps step index to page enum
static const int _step_to_page[] = {
  PAGE_THEME, PAGE_MODULES, PAGE_IMPORT, PAGE_QUICKSTART,
  PAGE_FIRSTEDIT, PAGE_RESOURCES
};

typedef struct dt_welcome_t
{
  GtkWidget *dialog;
  GtkWidget *stack;
  GtkWidget *btn_back;
  GtkWidget *btn_next;
  GtkWidget *btn_skip;
  GtkWidget *top_bar;
  GtkWidget *breadcrumb_box;
  GtkWidget *breadcrumb_labels[6];
  GtkWidget *breadcrumb_seps[6];   // separator before each label (index 0 unused)
  int current_page;

  // theme
  int theme_choice;          // 0=elegant-grey, 1=elegant-dark, 2=elegant-darker
  GtkWidget *theme_cards[3];

  // modules
  int modules_choice;        // 0=beginner, 1=workflow-matched, 2=all
  GtkWidget *modules_cards[3];

  // import
  GtkWidget *import_entry;
} dt_welcome_t;


// gtk_container_set_border_width is removed in GTK4; use per-side margins
static inline void _set_margin_all(GtkWidget *w, int margin)
{
  gtk_widget_set_margin_start(w, margin);
  gtk_widget_set_margin_end(w, margin);
  gtk_widget_set_margin_top(w, margin);
  gtk_widget_set_margin_bottom(w, margin);
}


// ── card visuals ─────────────────────────────────────────────────────

static void _cairo_rounded_rect(cairo_t *cr, double x, double y,
                                double w, double h, double r)
{
  cairo_new_path(cr);
  cairo_arc(cr, x + r, y + r, r, G_PI, 1.5 * G_PI);
  cairo_arc(cr, x + w - r, y + r, r, 1.5 * G_PI, 2.0 * G_PI);
  cairo_arc(cr, x + w - r, y + h - r, r, 0, 0.5 * G_PI);
  cairo_arc(cr, x + r, y + h - r, r, 0.5 * G_PI, G_PI);
  cairo_close_path(cr);
}

// theme: miniature UI layout with theme-appropriate grey levels
static gboolean _draw_theme_mockup(GtkWidget *widget, cairo_t *cr, gpointer data)
{
  const int type = GPOINTER_TO_INT(data);
  const int w = gtk_widget_get_allocated_width(widget);
  const int h = gtk_widget_get_allocated_height(widget);

  _cairo_rounded_rect(cr, 0, 0, w, h, 4.0);
  cairo_clip(cr);

  // actual theme colors from darktable CSS (hex → 0..1)
  // center darkroom area is always @grey_50 = #777 across all themes
  double bg, panel, bar;
  const double canvas = 0.467;  // @grey_50 = #777777 — neutral darkroom bg
  switch(type)
  {
    case 0: // elegant grey: bg=#6a6a6a, panel=shade(grey_50,0.95), bar=#5e5e5e
      bg = 0.416; panel = 0.444; bar = 0.369;
      break;
    case 1: // elegant dark: bg=#474747, panel=#525252, bar=#3b3b3b
      bg = 0.278; panel = 0.322; bar = 0.231;
      break;
    default: // elegant darker: bg=#303030, panel=#303030, bar=#1b1b1b
      bg = 0.188; panel = 0.188; bar = 0.106;
      break;
  }

  const double gap = 2.0;
  const double top_h = 8.0;
  const double bottom_h = 14.0;
  const double side_w = w * 0.22;
  const double mid_y = top_h + gap;
  const double mid_h = h - top_h - bottom_h - 2 * gap;

  // background
  cairo_set_source_rgb(cr, bg, bg, bg);
  cairo_paint(cr);

  // top bar
  cairo_set_source_rgb(cr, bar, bar, bar);
  cairo_rectangle(cr, 0, 0, w, top_h);
  cairo_fill(cr);

  // left panel (module sidebar)
  cairo_set_source_rgb(cr, panel, panel, panel);
  cairo_rectangle(cr, 0, mid_y, side_w, mid_h);
  cairo_fill(cr);

  // right panel (module sidebar)
  cairo_rectangle(cr, w - side_w, mid_y, side_w, mid_h);
  cairo_fill(cr);

  // center darkroom canvas — neutral grey background behind image
  cairo_set_source_rgb(cr, canvas, canvas, canvas);
  cairo_rectangle(cr, side_w + gap, mid_y,
                  w - 2 * side_w - 2 * gap, mid_h);
  cairo_fill(cr);

  // small image placeholder in center
  const double img_w = (w - 2 * side_w - 2 * gap) * 0.6;
  const double img_h = mid_h * 0.65;
  const double img_x = side_w + gap + (w - 2 * side_w - 2 * gap - img_w) / 2.0;
  const double img_y = mid_y + (mid_h - img_h) / 2.0;
  cairo_set_source_rgb(cr, canvas * 0.7, canvas * 0.7, canvas * 0.7);
  cairo_rectangle(cr, img_x, img_y, img_w, img_h);
  cairo_fill(cr);

  // bottom filmstrip
  cairo_set_source_rgb(cr, bar, bar, bar);
  cairo_rectangle(cr, 0, h - bottom_h, w, bottom_h);
  cairo_fill(cr);

  // thumbnail placeholders
  const int n_thumbs = 5;
  const double thumb_gap = 2.0;
  const double thumb_h = bottom_h - 4.0;
  const double thumb_w = (w - (n_thumbs + 1) * thumb_gap) / n_thumbs;
  cairo_set_source_rgb(cr, bar + 0.06, bar + 0.06, bar + 0.06);
  for(int i = 0; i < n_thumbs; i++)
  {
    const double tx = thumb_gap + i * (thumb_w + thumb_gap);
    cairo_rectangle(cr, tx, h - bottom_h + 2, thumb_w, thumb_h);
    cairo_fill(cr);
  }

  return FALSE;
}

// modules: dot grid showing approximate module count
static gboolean _draw_module_dots(GtkWidget *widget, cairo_t *cr, gpointer data)
{
  const int count = GPOINTER_TO_INT(data);
  const int w = gtk_widget_get_allocated_width(widget);
  const int h = gtk_widget_get_allocated_height(widget);

  // read theme colors for background and dots
  GtkStyleContext *ctx = gtk_widget_get_style_context(widget);
  GdkRGBA bg = { 0.12, 0.12, 0.12, 1.0 };
  GdkRGBA fg = { 0.55, 0.55, 0.55, 1.0 };
  gtk_style_context_lookup_color(ctx, "bg_color", &bg);
  gtk_style_context_lookup_color(ctx, "fg_color", &fg);

  _cairo_rounded_rect(cr, 0, 0, w, h, 4.0);
  cairo_clip(cr);

  // background from theme — slightly darker than bg_color
  cairo_set_source_rgb(cr, bg.red * 0.8, bg.green * 0.8, bg.blue * 0.8);
  cairo_paint(cr);

  const double dot_r = 2.5;
  const double dot_stride = 2 * dot_r + 3.0;
  const double pad = 8.0;

  int cols = (int)((w - 2 * pad) / dot_stride);
  if(cols < 1) cols = 1;
  const int rows = (count + cols - 1) / cols;

  const double grid_w = cols * dot_stride - 3.0;
  const double grid_h = rows * dot_stride - 3.0;
  const double start_x = (w - grid_w) / 2.0;
  const double start_y = (h - grid_h) / 2.0;

  cairo_set_source_rgba(cr, fg.red, fg.green, fg.blue, 0.5);

  for(int i = 0; i < count; i++)
  {
    const int col = i % cols;
    const int row = i / cols;
    const double cx = start_x + col * dot_stride + dot_r;
    const double cy = start_y + row * dot_stride + dot_r;
    cairo_arc(cr, cx, cy, dot_r, 0, 2 * G_PI);
    cairo_fill(cr);
  }

  return FALSE;
}

// GTK4: replace "draw" signal with gtk_drawing_area_set_draw_func()
static GtkWidget *_make_visual(GCallback draw_func, int type, int height)
{
  GtkWidget *da = gtk_drawing_area_new();
  gtk_widget_set_size_request(da, -1, height);
  g_signal_connect_data(da, "draw", draw_func, GINT_TO_POINTER(type), NULL, 0);
  return da;
}


// ── cairo icon helpers ────────────────────────────────────────────────

// common setup for all small icon drawing areas: returns center coords
static void _icon_setup(GtkWidget *widget, cairo_t *cr,
                         double *cx, double *cy)
{
  *cx = gtk_widget_get_allocated_width(widget) / 2.0;
  *cy = gtk_widget_get_allocated_height(widget) / 2.0;
  cairo_set_source_rgba(cr, 1, 1, 1, 0.5);
  cairo_set_line_width(cr, 1.2);
  cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
  cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
}

// ── tip helper ───────────────────────────────────────────────────────

static gboolean _draw_tip_icon(GtkWidget *widget, cairo_t *cr, gpointer data)
{
  (void)data;
  double cx, cy;
  _icon_setup(widget, cr, &cx, &cy);

  // bulb
  cairo_arc(cr, cx, cy - 2, 6, -G_PI, -0.15);
  cairo_stroke(cr);
  cairo_arc(cr, cx, cy - 2, 6, -G_PI + 0.15, 0);
  cairo_stroke(cr);

  // neck
  cairo_move_to(cr, cx - 3, cy + 4);
  cairo_line_to(cr, cx + 3, cy + 4);
  cairo_stroke(cr);
  cairo_move_to(cr, cx - 2.5, cy + 6);
  cairo_line_to(cr, cx + 2.5, cy + 6);
  cairo_stroke(cr);

  // filament
  cairo_move_to(cr, cx - 2, cy - 1);
  cairo_line_to(cr, cx, cy - 3);
  cairo_line_to(cr, cx + 2, cy - 1);
  cairo_stroke(cr);

  return FALSE;
}

static GtkWidget *_make_tip(const char *text)
{
  GtkWidget *frame = gtk_frame_new(NULL);
  gtk_widget_set_name(frame, "welcome-tip");
  gtk_frame_set_shadow_type(GTK_FRAME(frame), GTK_SHADOW_NONE); // GTK4: removed, use CSS

  GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
  _set_margin_all(hbox, 8);

  // icon
  GtkWidget *icon_da = gtk_drawing_area_new();
  gtk_widget_set_size_request(icon_da, 20, 20);
  gtk_widget_set_valign(icon_da, GTK_ALIGN_START);
  gtk_widget_set_margin_top(icon_da, 2);
  g_signal_connect(icon_da, "draw", G_CALLBACK(_draw_tip_icon), NULL);
  gtk_box_pack_start(GTK_BOX(hbox), icon_da, FALSE, FALSE, 0);

  // text
  GtkWidget *label = gtk_label_new(NULL);
  gchar *markup = g_strdup_printf("<b>tip:</b>  %s", text);
  gtk_label_set_markup(GTK_LABEL(label), markup);
  g_free(markup);
  gtk_label_set_line_wrap(GTK_LABEL(label), TRUE);
  gtk_label_set_xalign(GTK_LABEL(label), 0.0);
  gtk_box_pack_start(GTK_BOX(hbox), label, TRUE, TRUE, 0);

  gtk_container_add(GTK_CONTAINER(frame), hbox);
  gtk_widget_set_opacity(frame, 0.6);
  gtk_widget_set_margin_top(frame, 8);

  return frame;
}


// ── card helper ──────────────────────────────────────────────────────

static GtkWidget *_make_card(const char *title,
                             const char *subtitle,
                             const char *description,
                             gboolean recommended,
                             GtkWidget *visual)
{
  GtkWidget *frame = gtk_frame_new(NULL);
  gtk_widget_set_name(frame, "welcome-card");
  gtk_frame_set_shadow_type(GTK_FRAME(frame), GTK_SHADOW_NONE);

  GtkWidget *overlay = gtk_overlay_new();

  // content box — uniform padding for all cards so content aligns
  // top margin clears the recommended badge (~20px) on all cards
  GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
  const int pad = visual ? 10 : 16;
  gtk_widget_set_margin_start(box, pad);
  gtk_widget_set_margin_end(box, pad);
  gtk_widget_set_margin_bottom(box, pad);
  gtk_widget_set_margin_top(box, 24);

  // visual at top of card
  if(visual)
    gtk_box_pack_start(GTK_BOX(box), visual, FALSE, FALSE, 0);

  GtkWidget *title_label = gtk_label_new(NULL);
  gchar *markup = g_strdup_printf("<b>%s</b>", title);
  gtk_label_set_markup(GTK_LABEL(title_label), markup);
  g_free(markup);
  gtk_label_set_xalign(GTK_LABEL(title_label), 0.0);
  gtk_box_pack_start(GTK_BOX(box), title_label, FALSE, FALSE, 0);

  if(subtitle)
  {
    GtkWidget *tag = gtk_label_new(subtitle);
    gtk_widget_set_name(tag, "welcome-module-tag");
    gtk_widget_set_halign(tag, GTK_ALIGN_START);
    gtk_box_pack_start(GTK_BOX(box), tag, FALSE, FALSE, 0);
  }

  if(description)
  {
    GtkWidget *desc = gtk_label_new(description);
    gtk_label_set_line_wrap(GTK_LABEL(desc), TRUE);
    gtk_label_set_xalign(GTK_LABEL(desc), 0.0);
    gtk_widget_set_opacity(desc, 0.7);
    gtk_box_pack_start(GTK_BOX(box), desc, TRUE, TRUE, 0);
  }

  gtk_container_add(GTK_CONTAINER(overlay), box);

  // recommended badge — flush with card top-right corner
  if(recommended)
  {
    GtkWidget *badge = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(badge), _("★ recommended"));
    gtk_widget_set_name(badge, "welcome-card-badge");
    gtk_widget_set_halign(badge, GTK_ALIGN_END);
    gtk_widget_set_valign(badge, GTK_ALIGN_START);
    gtk_overlay_add_overlay(GTK_OVERLAY(overlay), badge);
  }

  gtk_container_add(GTK_CONTAINER(frame), overlay);
  return frame;
}

// ── card selection helper ────────────────────────────────────────────

typedef void (*_card_changed_cb)(dt_welcome_t *d);
static void _apply_theme_live(dt_welcome_t *d);

typedef struct {
  dt_welcome_t *d;
  int *choice_ptr;
  GtkWidget **cards;
  int n_cards;
  int index;
  _card_changed_cb on_changed;
} _card_click_data_t;

static void _update_card_selection(GtkWidget **cards, int n_cards, int selected)
{
  for(int i = 0; i < n_cards; i++)
  {
    // GTK4: gtk_widget_add_css_class / gtk_widget_remove_css_class
    GtkStyleContext *ctx = gtk_widget_get_style_context(cards[i]);
    if(i == selected)
      gtk_style_context_add_class(ctx, "welcome-card-selected");
    else
      gtk_style_context_remove_class(ctx, "welcome-card-selected");
  }
}

static void _card_pressed(GtkGestureSingle *gesture, int n_press,
                           double x, double y, _card_click_data_t *data)
{
  *data->choice_ptr = data->index;
  _update_card_selection(data->cards, data->n_cards, data->index);
  if(data->on_changed) data->on_changed(data->d);
}

static GtkWidget *_make_card_row(GtkWidget **cards, int n_cards,
                                 int *choice_ptr, int default_choice,
                                 dt_welcome_t *d,
                                 _card_changed_cb on_changed)
{
  GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 16);
  gtk_widget_set_halign(hbox, GTK_ALIGN_FILL);
  gtk_box_set_homogeneous(GTK_BOX(hbox), TRUE);

  for(int i = 0; i < n_cards; i++)
  {
    _card_click_data_t *data = g_malloc0(sizeof(_card_click_data_t));
    data->d = d;
    data->choice_ptr = choice_ptr;
    data->cards = cards;
    data->n_cards = n_cards;
    data->index = i;
    data->on_changed = on_changed;

    // GTK4: remove GtkEventBox, attach gesture directly to cards[i]
    GtkWidget *ebox = gtk_event_box_new();
    gtk_event_box_set_above_child(GTK_EVENT_BOX(ebox), TRUE);
    gtk_container_add(GTK_CONTAINER(ebox), cards[i]);
    dt_gui_connect_click(ebox, _card_pressed, NULL, data);
    g_signal_connect_swapped(ebox, "destroy", G_CALLBACK(g_free), data);

    gtk_box_pack_start(GTK_BOX(hbox), ebox, TRUE, TRUE, 0);
  }

  *choice_ptr = default_choice;
  _update_card_selection(cards, n_cards, default_choice);

  return hbox;
}

// ── page layout helpers ─────────────────────────────────────────────

// creates a top-aligned page with title, subtitle, and a content area
static GtkWidget *_page_with_header(const char *title_text,
                                    const char *explain_text,
                                    GtkWidget **out_content_box)
{
  GtkWidget *outer = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
  gtk_widget_set_valign(outer, GTK_ALIGN_START);
  gtk_widget_set_halign(outer, GTK_ALIGN_FILL);
  _set_margin_all(outer, 24);

  // title
  GtkWidget *title = gtk_label_new(NULL);
  gchar *markup = g_strdup_printf("<span size='x-large'><b>%s</b></span>", title_text);
  gtk_label_set_markup(GTK_LABEL(title), markup);
  g_free(markup);
  gtk_label_set_xalign(GTK_LABEL(title), 0.0);
  gtk_box_pack_start(GTK_BOX(outer), title, FALSE, FALSE, 0);

  // explanation
  if(explain_text)
  {
    GtkWidget *explain = gtk_label_new(explain_text);
    gtk_label_set_line_wrap(GTK_LABEL(explain), TRUE);
    gtk_label_set_max_width_chars(GTK_LABEL(explain), 80);
    gtk_label_set_xalign(GTK_LABEL(explain), 0.0);
    gtk_widget_set_opacity(explain, 0.7);
    gtk_widget_set_margin_top(explain, 6);
    gtk_box_pack_start(GTK_BOX(outer), explain, FALSE, FALSE, 0);
  }

  // content area for cards / widgets
  GtkWidget *content_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
  gtk_widget_set_margin_top(content_box, 16);
  gtk_box_pack_start(GTK_BOX(outer), content_box, FALSE, FALSE, 0);

  if(out_content_box) *out_content_box = content_box;
  return outer;
}


// ── page builders ────────────────────────────────────────────────────

static GtkWidget *_build_landing_page(dt_welcome_t *d)
{
  GtkWidget *page = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
  gtk_widget_set_valign(page, GTK_ALIGN_CENTER);
  gtk_widget_set_halign(page, GTK_ALIGN_CENTER);
  _set_margin_all(page, 40);

  // logo
  const dt_logo_season_t season = dt_util_get_logo_season();
  gchar *logo_file =
    season == DT_LOGO_SEASON_NONE
    ? g_strdup_printf("%s/pixmaps/idbutton.svg", darktable.datadir)
    : g_strdup_printf("%s/pixmaps/idbutton-%d.svg", darktable.datadir, season);
  // GTK4: GdkTexture + GtkPicture replaces GdkPixbuf + GtkImage
  GdkPixbuf *logo_pixbuf =
    gdk_pixbuf_new_from_file_at_size(logo_file, LOGO_SIZE, -1, NULL);
  g_free(logo_file);

  if(logo_pixbuf)
  {
    GtkWidget *logo = gtk_image_new_from_pixbuf(logo_pixbuf);
    g_object_unref(logo_pixbuf);
    gtk_widget_set_halign(logo, GTK_ALIGN_CENTER);
    gtk_widget_set_margin_bottom(logo, 24);
    gtk_box_pack_start(GTK_BOX(page), logo, FALSE, FALSE, 0);
  }

  // heading
  GtkWidget *heading = gtk_label_new(NULL);
  gtk_label_set_markup(GTK_LABEL(heading),
    _("<span size='30000'><b>welcome to darktable</b></span>"));
  gtk_widget_set_halign(heading, GTK_ALIGN_CENTER);
  gtk_box_pack_start(GTK_BOX(page), heading, FALSE, FALSE, 0);

  // callout description
  GtkWidget *desc = gtk_label_new(NULL);
  gtk_label_set_markup(GTK_LABEL(desc),
    _("<span size='large'>this short guide will help you set up darktable "
      "for your first use\n"
      "we'll walk you through a few key choices that affect\n"
      "how the application looks and processes your photos</span>"));
  gtk_label_set_line_wrap(GTK_LABEL(desc), TRUE);
  gtk_label_set_max_width_chars(GTK_LABEL(desc), 60);
  gtk_label_set_justify(GTK_LABEL(desc), GTK_JUSTIFY_CENTER);
  gtk_widget_set_halign(desc, GTK_ALIGN_CENTER);
  gtk_widget_set_opacity(desc, 0.6);
  gtk_widget_set_margin_top(desc, 20);
  gtk_box_pack_start(GTK_BOX(page), desc, FALSE, FALSE, 0);

  // reassurance
  GtkWidget *note = gtk_label_new(NULL);
  gtk_label_set_markup(GTK_LABEL(note),
    _("<span size='large'>every setting can be <b>changed later "
      "in preferences</b></span>"));
  gtk_widget_set_halign(note, GTK_ALIGN_CENTER);
  gtk_widget_set_opacity(note, 0.4);
  gtk_widget_set_margin_top(note, 28);
  gtk_box_pack_start(GTK_BOX(page), note, FALSE, FALSE, 0);

  return page;
}

static GtkWidget *_build_theme_page(dt_welcome_t *d)
{
  GtkWidget *area;
  GtkWidget *page = _page_with_header(
    _("choose your interface theme"),
    _("photo editors use dark themes so the interface doesn't influence "
      "how you perceive brightness and colour in your images. "
      "you can change this anytime in preferences"),
    &area);

  d->theme_cards[0] = _make_card(
    _("elegant grey"), NULL,
    _("balanced midtone grey \u2013 a neutral starting point that works "
      "well in most lighting conditions"),
    TRUE,
    _make_visual(G_CALLBACK(_draw_theme_mockup), 0, 80));

  d->theme_cards[1] = _make_card(
    _("elegant dark"), NULL,
    _("darker interface with higher contrast. good for dimly-lit "
      "editing environments"),
    FALSE,
    _make_visual(G_CALLBACK(_draw_theme_mockup), 1, 80));

  d->theme_cards[2] = _make_card(
    _("elegant darker"), NULL,
    _("near-black interface. maximum image focus with minimal "
      "UI distraction"),
    FALSE,
    _make_visual(G_CALLBACK(_draw_theme_mockup), 2, 80));

  GtkWidget *row = _make_card_row(d->theme_cards, 3,
                                  &d->theme_choice, 0, d,
                                  _apply_theme_live);
  gtk_box_pack_start(GTK_BOX(area), row, FALSE, FALSE, 0);

  gtk_box_pack_start(GTK_BOX(area),
    _make_tip(_("a grey neutral theme helps you <b>judge colours more accurately</b> "
                "by not biasing your perception. "
                "you can switch themes anytime in preferences")),
    FALSE, FALSE, 0);

  gtk_box_pack_start(GTK_BOX(area),
    _make_tip(_("darktable follows your <b>system language</b>. "
                "to change it, visit preferences \u2192 general")),
    FALSE, FALSE, 0);

  return page;
}

static GtkWidget *_build_modules_page(dt_welcome_t *d)
{
  GtkWidget *area;
  GtkWidget *page = _page_with_header(
    _("choose your module layout"),
    _("darktable has many image processing modules. module groups "
      "control which ones are shown in the darkroom sidebar \u2013 fewer "
      "visible modules means a simpler, less overwhelming interface"),
    &area);

  d->modules_cards[0] = _make_card(
    _("beginner"), _("~15 modules"),
    _("only the essential modules in 3 simple tabs. the easiest way "
      "to learn darktable without being overwhelmed by choices"),
    TRUE,
    _make_visual(G_CALLBACK(_draw_module_dots), 15, 50));

  d->modules_cards[1] = _make_card(
    _("workflow-matched"), _("~35 modules"),
    _("all modules organised into 5 tabs matching your chosen workflow. "
      "good once you're comfortable with the basics"),
    FALSE,
    _make_visual(G_CALLBACK(_draw_module_dots), 35, 50));

  d->modules_cards[2] = _make_card(
    _("all modules"), _("~70 modules"),
    _("every available module in a flat list. for experienced users "
      "who already know what they need"),
    FALSE,
    _make_visual(G_CALLBACK(_draw_module_dots), 70, 50));

  GtkWidget *row = _make_card_row(d->modules_cards, 3,
                                  &d->modules_choice, 0, d,
                                  NULL);
  gtk_box_pack_start(GTK_BOX(area), row, FALSE, FALSE, 0);

  gtk_box_pack_start(GTK_BOX(area),
    _make_tip(_("hidden modules are still available via the <b>search bar</b> "
                "in the darkroom. you can change the layout anytime")),
    FALSE, FALSE, 0);

  return page;
}

static void _browse_clicked(GtkWidget *entry)
{
  GtkWidget *chooser = gtk_file_chooser_dialog_new(
    _("select photo directory"),
    GTK_WINDOW(gtk_widget_get_toplevel(entry)),
    GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
    _("_cancel"), GTK_RESPONSE_CANCEL,
    _("_select"), GTK_RESPONSE_ACCEPT,
    NULL);
  const char *current = gtk_entry_get_text(GTK_ENTRY(entry));
  if(current && current[0])
    gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(chooser), current);
  // GTK4: GtkFileDialog replaces GtkFileChooserDialog + gtk_dialog_run
  if(gtk_dialog_run(GTK_DIALOG(chooser)) == GTK_RESPONSE_ACCEPT)
  {
    char *folder = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(chooser));
    if(folder)
    {
      gtk_entry_set_text(GTK_ENTRY(entry), folder);
      g_free(folder);
    }
  }
  gtk_widget_destroy(chooser);
}

static GtkWidget *_build_import_page(dt_welcome_t *d)
{
  GtkWidget *area;
  GtkWidget *page = _page_with_header(
    _("choose your photo library location"),
    _("in darktable, every folder you import becomes a film roll \u2013 "
      "the metaphor for a physical roll of film. choose the root folder "
      "where you keep your photos and darktable will open it by default "
      "when you click import. each subfolder becomes its own film roll"),
    &area);

  // path entry + browse button
  GtkWidget *path_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);

  const char *pictures = g_get_user_special_dir(G_USER_DIRECTORY_PICTURES);
  gchar *default_path = g_build_filename(
    pictures ? pictures : g_get_home_dir(), "Darktable", NULL);
  d->import_entry = gtk_entry_new();
  gtk_entry_set_text(GTK_ENTRY(d->import_entry), default_path);
  g_free(default_path);
  gtk_widget_set_hexpand(d->import_entry, TRUE);
  gtk_box_pack_start(GTK_BOX(path_box), d->import_entry, TRUE, TRUE, 0);

  GtkWidget *browse_btn = gtk_button_new_with_label(_("browse..."));
  g_signal_connect_swapped(browse_btn, "clicked",
                           G_CALLBACK(_browse_clicked), d->import_entry);
  gtk_box_pack_start(GTK_BOX(path_box), browse_btn, FALSE, FALSE, 0);

  gtk_box_pack_start(GTK_BOX(area), path_box, FALSE, FALSE, 0);

  GtkWidget *hint = gtk_label_new(
    _("leave blank if your photos live in different places \u2013 "
      "darktable will open the system default location each time"));
  gtk_label_set_xalign(GTK_LABEL(hint), 0.0);
  gtk_label_set_line_wrap(GTK_LABEL(hint), TRUE);
  gtk_widget_set_opacity(hint, 0.5);
  gtk_widget_set_margin_top(hint, 4);
  gtk_box_pack_start(GTK_BOX(area), hint, FALSE, FALSE, 0);

  gtk_box_pack_start(GTK_BOX(area),
    _make_tip(_("this is just a <b>convenience starting point</b> for the "
                "file browser \u2013 it doesn't scan, index, or restrict where "
                "you can import from. darktable <b>never moves or copies</b> "
                "your originals")),
    FALSE, FALSE, 0);

  return page;
}


// ── pipeline comparison page ─────────────────────────────────────────

// icon types for pipeline stages
enum {
  STAGE_ICON_CAMERA,
  STAGE_ICON_WARN,
  STAGE_ICON_SLIDERS,
  STAGE_ICON_DISPLAY,
  STAGE_ICON_CURVE
};

static gboolean _draw_stage_icon(GtkWidget *widget, cairo_t *cr, gpointer data)
{
  const int type = GPOINTER_TO_INT(data);
  double cx, cy;
  _icon_setup(widget, cr, &cx, &cy);
  cairo_set_source_rgba(cr, 1, 1, 1, 0.45);  // slightly dimmer than default

  switch(type)
  {
    case STAGE_ICON_CAMERA:
    {
      // camera body
      _cairo_rounded_rect(cr, cx - 7, cy - 3, 14, 9, 1.5);
      cairo_stroke(cr);
      // lens
      cairo_arc(cr, cx, cy + 1.5, 2.5, 0, 2 * G_PI);
      cairo_stroke(cr);
      // viewfinder bump
      cairo_rectangle(cr, cx - 2, cy - 5, 4, 2.5);
      cairo_fill(cr);
      break;
    }
    case STAGE_ICON_WARN:
    {
      // triangle
      cairo_move_to(cr, cx, cy - 6);
      cairo_line_to(cr, cx + 7, cy + 5);
      cairo_line_to(cr, cx - 7, cy + 5);
      cairo_close_path(cr);
      cairo_stroke(cr);
      // exclamation
      cairo_set_line_width(cr, 1.5);
      cairo_move_to(cr, cx, cy - 2);
      cairo_line_to(cr, cx, cy + 1);
      cairo_stroke(cr);
      cairo_arc(cr, cx, cy + 3, 0.8, 0, 2 * G_PI);
      cairo_fill(cr);
      break;
    }
    case STAGE_ICON_SLIDERS:
    {
      // three slider lines with handles at different positions
      const double offsets[] = { -2, 3, -4 };
      for(int i = 0; i < 3; i++)
      {
        const double y = cy - 4 + i * 4;
        cairo_move_to(cr, cx - 7, y);
        cairo_line_to(cr, cx + 7, y);
        cairo_stroke(cr);
        cairo_arc(cr, cx + offsets[i], y, 1.5, 0, 2 * G_PI);
        cairo_fill(cr);
      }
      break;
    }
    case STAGE_ICON_DISPLAY:
    {
      // screen
      _cairo_rounded_rect(cr, cx - 7, cy - 5, 14, 9, 1);
      cairo_stroke(cr);
      // stand
      cairo_move_to(cr, cx, cy + 4);
      cairo_line_to(cr, cx, cy + 6);
      cairo_stroke(cr);
      cairo_move_to(cr, cx - 4, cy + 6);
      cairo_line_to(cr, cx + 4, cy + 6);
      cairo_stroke(cr);
      break;
    }
    case STAGE_ICON_CURVE:
    {
      // sigmoid S-curve
      cairo_move_to(cr, cx - 7, cy + 5);
      cairo_curve_to(cr, cx - 2, cy + 5,
                     cx + 2, cy - 5,
                     cx + 7, cy - 5);
      cairo_stroke(cr);
      break;
    }
    default:
      break;
  }

  return FALSE;
}

// pipeline stage box with icon
static GtkWidget *_make_stage(int icon_type, const char *title,
                               const char *subtitle, gboolean warn)
{
  GtkWidget *frame = gtk_frame_new(NULL);
  gtk_widget_set_name(frame, warn ? "welcome-stage-warn" : "welcome-stage");
  gtk_frame_set_shadow_type(GTK_FRAME(frame), GTK_SHADOW_NONE);

  GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 1);
  _set_margin_all(box, 4);
  gtk_widget_set_valign(box, GTK_ALIGN_CENTER);

  // icon
  if(icon_type >= 0)
  {
    GtkWidget *icon_da = gtk_drawing_area_new();
    gtk_widget_set_size_request(icon_da, 18, 18);
    gtk_widget_set_halign(icon_da, GTK_ALIGN_CENTER);
    g_signal_connect_data(icon_da, "draw", G_CALLBACK(_draw_stage_icon),
                          GINT_TO_POINTER(icon_type), NULL, 0);
    gtk_box_pack_start(GTK_BOX(box), icon_da, FALSE, FALSE, 0);
  }

  GtkWidget *t = gtk_label_new(NULL);
  gchar *m = g_strdup_printf("<small><b>%s</b></small>", _(title));
  gtk_label_set_markup(GTK_LABEL(t), m);
  g_free(m);
  gtk_box_pack_start(GTK_BOX(box), t, FALSE, FALSE, 0);

  if(subtitle)
  {
    GtkWidget *s = gtk_label_new(NULL);
    gchar *sm;
    if(warn)
      sm = g_strdup_printf("<span size='x-small' alpha='70%%'>%s</span>",
                           _(subtitle));
    else
      sm = g_strdup_printf("<span size='x-small' alpha='50%%'>%s</span>",
                           _(subtitle));
    gtk_label_set_markup(GTK_LABEL(s), sm);
    g_free(sm);
    gtk_box_pack_start(GTK_BOX(box), s, FALSE, FALSE, 0);
  }

  gtk_container_add(GTK_CONTAINER(frame), box);
  return frame;
}

static GtkWidget *_make_pipe_arrow(void)
{
  GtkWidget *lbl = gtk_label_new("\u25b8");
  gtk_widget_set_opacity(lbl, 0.3);
  gtk_widget_set_valign(lbl, GTK_ALIGN_CENTER);
  return lbl;
}

// pipeline stage descriptor — data-driven pipeline row construction
typedef struct {
  int icon;
  const char *title;
  const char *subtitle;
  gboolean warn;
} _pipe_stage_t;

static GtkWidget *_build_pipeline_row(const _pipe_stage_t *stages, int n_stages)
{
  GtkWidget *pipe = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
  for(int i = 0; i < n_stages; i++)
  {
    if(i > 0)
      gtk_box_pack_start(GTK_BOX(pipe), _make_pipe_arrow(), FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(pipe),
      _make_stage(stages[i].icon, stages[i].title, stages[i].subtitle,
                  stages[i].warn),
      TRUE, TRUE, 0);
  }
  return pipe;
}

// dynamic range bar: 0=display-referred (lost), 1=scene-referred (full)
static gboolean _draw_range_bar(GtkWidget *widget, cairo_t *cr, gpointer data)
{
  const int type = GPOINTER_TO_INT(data);
  const int w = gtk_widget_get_allocated_width(widget);
  const int h = gtk_widget_get_allocated_height(widget);

  _cairo_rounded_rect(cr, 0, 0, w, h, 3.0);
  cairo_clip(cr);

  // base gradient — shared by both bar types
  cairo_pattern_t *pat = cairo_pattern_create_linear(0, 0, w, 0);
  cairo_pattern_add_color_stop_rgb(pat, 0.0, 0.15, 0.15, 0.15);
  cairo_pattern_add_color_stop_rgb(pat, 1.0, 0.65, 0.65, 0.65);
  cairo_set_source(cr, pat);
  cairo_paint(cr);
  cairo_pattern_destroy(pat);

  if(type == 0)
  {
    // display-referred: angled stripes over lost section
    const double cut = w * 0.72;
    cairo_save(cr);
    cairo_rectangle(cr, cut, 0, w - cut, h);
    cairo_clip(cr);

    const double stripe_w = 4.0;
    const double step = stripe_w + 4.0;
    cairo_set_source_rgba(cr, 0, 0, 0, 0.25);
    cairo_set_line_width(cr, stripe_w);

    for(double x = cut - h; x < w + h; x += step)
    {
      cairo_move_to(cr, x, h);
      cairo_line_to(cr, x + h, 0);
      cairo_stroke(cr);
    }
    cairo_restore(cr);
  }

  return FALSE;
}

static GtkWidget *_make_info_card(const char *title, const char *text)
{
  GtkWidget *frame = gtk_frame_new(NULL);
  gtk_widget_set_name(frame, "welcome-info");
  gtk_frame_set_shadow_type(GTK_FRAME(frame), GTK_SHADOW_NONE);

  GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
  _set_margin_all(box, 8);

  GtkWidget *t = gtk_label_new(NULL);
  gchar *m = g_strdup_printf("<b>%s</b>", _(title));
  gtk_label_set_markup(GTK_LABEL(t), m);
  g_free(m);
  gtk_label_set_xalign(GTK_LABEL(t), 0.0);
  gtk_box_pack_start(GTK_BOX(box), t, FALSE, FALSE, 0);

  GtkWidget *d = gtk_label_new(_(text));
  gtk_label_set_line_wrap(GTK_LABEL(d), TRUE);
  gtk_label_set_xalign(GTK_LABEL(d), 0.0);
  gtk_widget_set_opacity(d, 0.7);
  gtk_box_pack_start(GTK_BOX(box), d, FALSE, FALSE, 0);

  gtk_container_add(GTK_CONTAINER(frame), box);
  return frame;
}

static GtkWidget *_build_quickstart_page(dt_welcome_t *d)
{
  GtkWidget *area;
  GtkWidget *page = _page_with_header(
    _("how darktable is different"),
    _("most RAW editors apply a gamma curve to your file immediately \u2013 "
      "before you touch anything. darktable keeps you in linear light "
      "the whole time and converts last"),
    &area);

  // === two comparison panels ===
  GtkWidget *panels = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);

  // --- left panel: most editors (display-referred) ---
  {
    GtkWidget *panel = gtk_frame_new(NULL);
    gtk_widget_set_name(panel, "welcome-panel");
    gtk_frame_set_shadow_type(GTK_FRAME(panel), GTK_SHADOW_NONE);

    GtkWidget *pbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    _set_margin_all(pbox, 8);

    // header
    GtkWidget *hdr = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    GtkWidget *htitle = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(htitle),
      _("<small><b>MOST EDITORS</b></small>"));
    gtk_widget_set_opacity(htitle, 0.6);
    gtk_box_pack_start(GTK_BOX(hdr), htitle, FALSE, FALSE, 0);
    GtkWidget *htag = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(htag), _("<small>display-referred</small>"));
    gtk_widget_set_opacity(htag, 0.5);
    gtk_box_pack_end(GTK_BOX(hdr), htag, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(pbox), hdr, FALSE, FALSE, 0);

    // pipeline: RAW → GAMMA BAKE → YOUR EDITS → DISPLAY
    static const _pipe_stage_t display_stages[] = {
      { STAGE_ICON_CAMERA,  N_("RAW"),        N_("linear light"),       FALSE },
      { STAGE_ICON_WARN,    N_("GAMMA BAKE"), N_("highlights crushed"), TRUE },
      { STAGE_ICON_SLIDERS, N_("YOUR EDITS"), N_("on baked data"),      FALSE },
      { STAGE_ICON_DISPLAY, N_("DISPLAY"),    NULL,                     FALSE },
    };
    gtk_box_pack_start(GTK_BOX(pbox),
      _build_pipeline_row(display_stages,
                          sizeof(display_stages) / sizeof(display_stages[0])),
      FALSE, FALSE, 0);

    // range bar labels (above)
    GtkWidget *bar_labels = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_widget_set_margin_top(bar_labels, 2);
    GtkWidget *bar_lbl_edit = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(bar_lbl_edit),
      _("<span size='small' alpha='50%%'>dynamic range you can edit</span>"));
    gtk_box_pack_start(GTK_BOX(bar_labels), bar_lbl_edit, FALSE, FALSE, 0);
    GtkWidget *bar_lbl_lost = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(bar_lbl_lost),
      _("<span size='small' alpha='50%%'>lost</span>"));
    gtk_box_pack_end(GTK_BOX(bar_labels), bar_lbl_lost, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(pbox), bar_labels, FALSE, FALSE, 0);

    // range bar
    GtkWidget *bar = gtk_drawing_area_new();
    gtk_widget_set_size_request(bar, -1, 14);
    g_signal_connect_data(bar, "draw", G_CALLBACK(_draw_range_bar),
                          GINT_TO_POINTER(0), NULL, 0);
    gtk_box_pack_start(GTK_BOX(pbox), bar, FALSE, FALSE, 0);

    // note
    GtkWidget *note = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(note),
      _("<b>~1\u20132 stops</b> of highlight data baked away "
        "before you touch anything"));
    gtk_label_set_xalign(GTK_LABEL(note), 0.0);
    gtk_label_set_line_wrap(GTK_LABEL(note), TRUE);
    gtk_widget_set_opacity(note, 0.5);
    gtk_box_pack_start(GTK_BOX(pbox), note, FALSE, FALSE, 0);

    gtk_container_add(GTK_CONTAINER(panel), pbox);
    gtk_box_pack_start(GTK_BOX(panels), panel, TRUE, TRUE, 0);
  }

  // --- right panel: darktable (scene-referred) ---
  {
    GtkWidget *panel = gtk_frame_new(NULL);
    gtk_widget_set_name(panel, "welcome-panel-dt");
    gtk_frame_set_shadow_type(GTK_FRAME(panel), GTK_SHADOW_NONE);

    GtkWidget *pbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    _set_margin_all(pbox, 8);

    // header
    GtkWidget *hdr = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    GtkWidget *htitle = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(htitle), _("<small><b>DARKTABLE</b></small>"));
    gtk_box_pack_start(GTK_BOX(hdr), htitle, FALSE, FALSE, 0);
    GtkWidget *htag = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(htag), _("<small>scene-referred</small>"));
    gtk_widget_set_opacity(htag, 0.7);
    gtk_box_pack_end(GTK_BOX(hdr), htag, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(pbox), hdr, FALSE, FALSE, 0);

    // pipeline: RAW → YOUR EDITS → TONE MAP → DISPLAY
    static const _pipe_stage_t scene_stages[] = {
      { STAGE_ICON_CAMERA,  N_("RAW"),        N_("linear light"),      FALSE },
      { STAGE_ICON_SLIDERS, N_("YOUR EDITS"), N_("on linear data"),    FALSE },
      { STAGE_ICON_CURVE,   N_("TONE MAP"),   N_("sigmoid / filmic"),  FALSE },
      { STAGE_ICON_DISPLAY, N_("DISPLAY"),    NULL,                    FALSE },
    };
    gtk_box_pack_start(GTK_BOX(pbox),
      _build_pipeline_row(scene_stages,
                          sizeof(scene_stages) / sizeof(scene_stages[0])),
      FALSE, FALSE, 0);

    // range bar labels (above)
    GtkWidget *bar_labels = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_widget_set_margin_top(bar_labels, 2);
    GtkWidget *bar_lbl_full = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(bar_lbl_full),
      _("<span size='small' alpha='50%%'>full scene data \u2013 you decide what to compress</span>"));
    gtk_box_pack_start(GTK_BOX(bar_labels), bar_lbl_full, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(pbox), bar_labels, FALSE, FALSE, 0);

    // range bar
    GtkWidget *bar = gtk_drawing_area_new();
    gtk_widget_set_size_request(bar, -1, 14);
    g_signal_connect_data(bar, "draw", G_CALLBACK(_draw_range_bar),
                          GINT_TO_POINTER(1), NULL, 0);
    gtk_box_pack_start(GTK_BOX(pbox), bar, FALSE, FALSE, 0);

    // note
    GtkWidget *note = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(note),
      _("highlight recovery works because the data is "
        "<b>actually still there</b>"));
    gtk_label_set_xalign(GTK_LABEL(note), 0.0);
    gtk_label_set_line_wrap(GTK_LABEL(note), TRUE);
    gtk_widget_set_opacity(note, 0.5);
    gtk_box_pack_start(GTK_BOX(pbox), note, FALSE, FALSE, 0);

    gtk_container_add(GTK_CONTAINER(panel), pbox);
    gtk_box_pack_start(GTK_BOX(panels), panel, TRUE, TRUE, 0);
  }

  gtk_box_pack_start(GTK_BOX(area), panels, FALSE, FALSE, 0);

  // === info cards ===
  GtkWidget *cards = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
  gtk_box_set_homogeneous(GTK_BOX(cards), TRUE);
  gtk_widget_set_margin_top(cards, 8);

  gtk_box_pack_start(GTK_BOX(cards),
    _make_info_card(
      N_("flat RAW? that\u2019s normal"),
      N_("your image will look washed-out until you add sigmoid or "
         "filmic. that module is the contrast curve \u2013 "
         "it\u2019s just made explicit here")),
    TRUE, TRUE, 0);

  gtk_box_pack_start(GTK_BOX(cards),
    _make_info_card(
      N_("highlight recovery is real"),
      N_("the overexposed data is preserved in linear light. you choose "
         "how to compress it, not a bake step you never saw")),
    TRUE, TRUE, 0);

  gtk_box_pack_start(GTK_BOX(cards),
    _make_info_card(
      N_("new module names"),
      N_("exposure \u00b7 colour calibration \u00b7 sigmoid replace "
         "highlights/shadows/whites \u2013 same ideas, physically "
         "correct vocabulary")),
    TRUE, TRUE, 0);

  gtk_box_pack_start(GTK_BOX(area), cards, FALSE, FALSE, 0);

  return page;
}


// helper: make a numbered step card for the first-edit page
static GtkWidget *_make_step_card(int number, const char *title,
                                   const char *modules, const char *desc,
                                   const char *note)
{
  GtkWidget *row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
  gtk_widget_set_margin_top(row, 8);
  gtk_widget_set_margin_bottom(row, 8);

  // number circle
  GtkWidget *num_frame = gtk_frame_new(NULL);
  gtk_widget_set_name(num_frame, "welcome-step-num");
  gtk_frame_set_shadow_type(GTK_FRAME(num_frame), GTK_SHADOW_NONE);
  gtk_widget_set_valign(num_frame, GTK_ALIGN_START);
  gtk_widget_set_size_request(num_frame, 28, 28);

  GtkWidget *num_label = gtk_label_new(NULL);
  gchar *num_m = g_strdup_printf("<b>%d</b>", number);
  gtk_label_set_markup(GTK_LABEL(num_label), num_m);
  g_free(num_m);
  gtk_container_add(GTK_CONTAINER(num_frame), num_label);
  gtk_box_pack_start(GTK_BOX(row), num_frame, FALSE, FALSE, 0);

  // text column
  GtkWidget *col = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);

  // title row: title on the left, module tags on the right
  {
    GtkWidget *title_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_widget_set_valign(title_row, GTK_ALIGN_CENTER);

    GtkWidget *t = gtk_label_new(NULL);
    gchar *esc_title = g_markup_escape_text(_(title), -1);
    gchar *tm = g_strdup_printf("<b>%s</b>", esc_title);
    gtk_label_set_markup(GTK_LABEL(t), tm);
    g_free(tm);
    g_free(esc_title);
    gtk_box_pack_start(GTK_BOX(title_row), t, FALSE, FALSE, 0);

    // module tags
    gchar **parts = g_strsplit(modules, MODULE_SEP, -1);
    for(int i = 0; parts && parts[i]; i++)
    {
      gchar *trimmed = g_strstrip(g_strdup(parts[i]));
      if(trimmed[0])
      {
        GtkWidget *tag = gtk_label_new(NULL);
        gchar *esc = g_markup_escape_text(trimmed, -1);
        gchar *tag_m = g_strdup_printf("<span size='small'>%s</span>", esc);
        gtk_label_set_markup(GTK_LABEL(tag), tag_m);
        g_free(tag_m);
        g_free(esc);
        gtk_widget_set_name(tag, "welcome-module-tag");
        gtk_box_pack_start(GTK_BOX(title_row), tag, FALSE, FALSE, 0);
      }
      g_free(trimmed);
    }
    g_strfreev(parts);
    gtk_box_pack_start(GTK_BOX(col), title_row, FALSE, FALSE, 0);
  }

  // description
  GtkWidget *d_lbl = gtk_label_new(_(desc));
  gtk_label_set_line_wrap(GTK_LABEL(d_lbl), TRUE);
  gtk_label_set_max_width_chars(GTK_LABEL(d_lbl), 70);
  gtk_label_set_xalign(GTK_LABEL(d_lbl), 0.0);
  gtk_widget_set_opacity(d_lbl, 0.7);
  gtk_widget_set_margin_top(d_lbl, 2);
  gtk_box_pack_start(GTK_BOX(col), d_lbl, FALSE, FALSE, 0);

  // optional note
  if(note)
  {
    GtkWidget *n = gtk_label_new(NULL);
    gchar *nm = g_strdup_printf("<i>%s</i>", _(note));
    gtk_label_set_markup(GTK_LABEL(n), nm);
    g_free(nm);
    gtk_label_set_line_wrap(GTK_LABEL(n), TRUE);
    gtk_label_set_max_width_chars(GTK_LABEL(n), 70);
    gtk_label_set_xalign(GTK_LABEL(n), 0.0);
    gtk_widget_set_opacity(n, 0.5);
    gtk_widget_set_margin_top(n, 2);
    gtk_box_pack_start(GTK_BOX(col), n, FALSE, FALSE, 0);
  }

  gtk_box_pack_start(GTK_BOX(row), col, TRUE, TRUE, 0);
  return row;
}


static GtkWidget *_build_firstedit_page(dt_welcome_t *d)
{
  GtkWidget *area;
  GtkWidget *page = _page_with_header(
    _("processing your first RAW"),
    _("follow these steps in order. everything else in darktable "
      "is refinement on top of this"),
    &area);

  // module names are not wrapped in N_() — they are darktable module
  // names translated in their own .po entries, and the · separator must
  // stay constant for _make_step_card to split them into tags
  static const struct { const char *title; const char *modules;
                        const char *desc; } steps[] = {
    { N_("set exposure"),
      "exposure",
      N_("push EV until midtones look right \u2013 don\u2019t worry about "
         "blown highlights yet, the tone mapper handles them") },
    { N_("set white balance"),
      "colour calibration",
      N_("use colour calibration, not the legacy white balance module \u2013 "
         "grey-pick a neutral area or enter a temperature manually") },
    { N_("apply tone mapper"),
      "sigmoid \u00b7 filmic rgb",
      N_("this maps scene-linear light to what your display can show \u2013 "
         "sigmoid needs almost no configuration, filmic gives more control") },
    { N_("balance tones"),
      "tone equalizer",
      N_("selectively brighten or darken zones (shadows, midtones, highlights) "
         "without halos \u2013 use the on-canvas picker to target specific "
         "luminosity regions") },
    { N_("adjust colour"),
      "colour balance rgb",
      N_("control global saturation, brilliance, and per-range colour "
         "lift/gain/gamma \u2013 the most powerful colour grading tool in "
         "darktable, start with global saturation") },
    { N_("sharpen & denoise"),
      "diffuse or sharpen \u00b7 denoise (profiled)",
      N_("apply after tone mapping \u2013 denoise profiled has camera-specific "
         "profiles, select your model for best results") },
  };
  const int n_steps = sizeof(steps) / sizeof(steps[0]);

  // two-column layout: first half left, second half right
  GtkWidget *columns = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 20);
  GtkWidget *col_left = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
  GtkWidget *col_right = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);

  for(int i = 0; i < n_steps; i++)
  {
    GtkWidget *col = (i < n_steps / 2) ? col_left : col_right;
    gtk_box_pack_start(GTK_BOX(col),
      _make_step_card(i + 1, steps[i].title, steps[i].modules,
                      steps[i].desc, NULL),
      FALSE, FALSE, 0);
  }

  gtk_box_pack_start(GTK_BOX(columns), col_left, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(columns), col_right, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(area), columns, FALSE, FALSE, 0);

  return page;
}

static void _resource_link_pressed(GtkGestureSingle *gesture, int n_press,
                                    double x, double y, gpointer user_data)
{
  GtkWidget *widget = gtk_event_controller_get_widget(GTK_EVENT_CONTROLLER(gesture));
  const char *url = g_object_get_data(G_OBJECT(widget), "url");
  // GTK4: gtk_show_uri(parent, url, GDK_CURRENT_TIME)
  if(url) gtk_show_uri_on_window(NULL, url, GDK_CURRENT_TIME, NULL);
}

// icon types for resource links
enum {
  RES_ICON_BOOK,
  RES_ICON_FORUM,
  RES_ICON_BUG
};

static gboolean _draw_resource_icon(GtkWidget *widget, cairo_t *cr, gpointer data)
{
  const int type = GPOINTER_TO_INT(data);
  double cx, cy;
  _icon_setup(widget, cr, &cx, &cy);

  switch(type)
  {
    case RES_ICON_BOOK:
    {
      // closed book with bookmark
      _cairo_rounded_rect(cr, cx - 8, cy - 9, 16, 18, 2);
      cairo_stroke(cr);
      // spine
      cairo_move_to(cr, cx - 5, cy - 9);
      cairo_line_to(cr, cx - 5, cy + 9);
      cairo_stroke(cr);
      // lines on page
      cairo_move_to(cr, cx - 2, cy - 4);
      cairo_line_to(cr, cx + 5, cy - 4);
      cairo_stroke(cr);
      cairo_move_to(cr, cx - 2, cy - 1);
      cairo_line_to(cr, cx + 5, cy - 1);
      cairo_stroke(cr);
      cairo_move_to(cr, cx - 2, cy + 2);
      cairo_line_to(cr, cx + 3, cy + 2);
      cairo_stroke(cr);
      break;
    }
    case RES_ICON_FORUM:
    {
      // single speech bubble with tail and dots
      _cairo_rounded_rect(cr, cx - 10, cy - 8, 20, 14, 3);
      cairo_stroke(cr);
      // tail (triangle pointing bottom-left)
      cairo_move_to(cr, cx - 4, cy + 6);
      cairo_line_to(cr, cx - 8, cy + 11);
      cairo_line_to(cr, cx, cy + 6);
      cairo_fill(cr);
      // three dots
      cairo_arc(cr, cx - 4, cy - 1, 1.2, 0, 2 * G_PI);
      cairo_fill(cr);
      cairo_arc(cr, cx, cy - 1, 1.2, 0, 2 * G_PI);
      cairo_fill(cr);
      cairo_arc(cr, cx + 4, cy - 1, 1.2, 0, 2 * G_PI);
      cairo_fill(cr);
      break;
    }
    case RES_ICON_BUG:
    {
      // circle with exclamation (issue/bug report)
      cairo_arc(cr, cx, cy, 9, 0, 2 * G_PI);
      cairo_stroke(cr);
      // exclamation mark
      cairo_set_line_width(cr, 1.5);
      cairo_move_to(cr, cx, cy - 5);
      cairo_line_to(cr, cx, cy + 1);
      cairo_stroke(cr);
      cairo_arc(cr, cx, cy + 4, 1.0, 0, 2 * G_PI);
      cairo_fill(cr);
      break;
    }
    default:
      break;
  }
  return FALSE;
}

static GtkWidget *_build_resources_page(dt_welcome_t *d)
{
  GtkWidget *area;
  GtkWidget *page = _page_with_header(
    _("learn and connect"),
    _("bookmark these \u2013 they're the best places to go when you have "
      "questions or want to contribute"),
    &area);

  const struct { const char *title; const char *subtitle; const char *url; int icon; } links[] = {
    { N_("official user manual"), "docs.darktable.org/usermanual",
      "https://docs.darktable.org/", RES_ICON_BOOK },
    { N_("pixls.us community forum"), "discuss.pixls.us",
      "https://discuss.pixls.us/c/software/darktable/", RES_ICON_FORUM },
    { N_("GitHub \u2013 report an issue"), "github.com/darktable-org/darktable",
      "https://github.com/darktable-org/darktable/issues", RES_ICON_BUG },
  };

  for(size_t i = 0; i < sizeof(links) / sizeof(links[0]); i++)
  {
    GtkWidget *link_frame = gtk_frame_new(NULL);
    gtk_widget_set_name(link_frame, "welcome-card");
    gtk_frame_set_shadow_type(GTK_FRAME(link_frame), GTK_SHADOW_NONE);

    GtkWidget *link_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
    _set_margin_all(link_box, 12);

    // icon
    GtkWidget *icon_da = gtk_drawing_area_new();
    gtk_widget_set_size_request(icon_da, 28, 28);
    gtk_widget_set_valign(icon_da, GTK_ALIGN_CENTER);
    g_signal_connect_data(icon_da, "draw", G_CALLBACK(_draw_resource_icon),
                          GINT_TO_POINTER(links[i].icon), NULL, 0);
    gtk_box_pack_start(GTK_BOX(link_box), icon_da, FALSE, FALSE, 0);

    // text
    GtkWidget *text_col = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);

    GtkWidget *link_title = gtk_label_new(NULL);
    gchar *link_markup = g_strdup_printf("<b>%s</b>", _(links[i].title));
    gtk_label_set_markup(GTK_LABEL(link_title), link_markup);
    g_free(link_markup);
    gtk_label_set_xalign(GTK_LABEL(link_title), 0.0);
    gtk_box_pack_start(GTK_BOX(text_col), link_title, FALSE, FALSE, 0);

    GtkWidget *link_sub = gtk_label_new(links[i].subtitle);
    gtk_label_set_xalign(GTK_LABEL(link_sub), 0.0);
    gtk_widget_set_opacity(link_sub, 0.5);
    gtk_box_pack_start(GTK_BOX(text_col), link_sub, FALSE, FALSE, 0);

    gtk_box_pack_start(GTK_BOX(link_box), text_col, TRUE, TRUE, 0);

    gtk_container_add(GTK_CONTAINER(link_frame), link_box);

    // GTK4: remove GtkEventBox, attach gesture directly to link_frame
    GtkWidget *ebox = gtk_event_box_new();
    gtk_event_box_set_above_child(GTK_EVENT_BOX(ebox), TRUE);
    gtk_container_add(GTK_CONTAINER(ebox), link_frame);
    g_object_set_data(G_OBJECT(ebox), "url", (gpointer)links[i].url);
    dt_gui_connect_click(ebox, _resource_link_pressed, NULL, NULL);

    gtk_box_pack_start(GTK_BOX(area), ebox, FALSE, FALSE, 0);
  }

  // done message
  GtkWidget *done_msg = gtk_label_new(NULL);
  gtk_label_set_markup(GTK_LABEL(done_msg),
    _("you're all set \u2013 these settings can be revisited in <b>preferences</b>"));
  gtk_label_set_line_wrap(GTK_LABEL(done_msg), TRUE);
  gtk_label_set_xalign(GTK_LABEL(done_msg), 0.0);
  gtk_widget_set_opacity(done_msg, 0.7);
  gtk_widget_set_margin_top(done_msg, 16);
  gtk_box_pack_start(GTK_BOX(area), done_msg, FALSE, FALSE, 0);

  return page;
}


// ── breadcrumb ───────────────────────────────────────────────────────

static void _update_breadcrumb(dt_welcome_t *d)
{
  if(d->current_page == PAGE_LANDING)
  {
    gtk_widget_hide(d->top_bar);
    return;
  }
  gtk_widget_show(d->top_bar);
  gtk_widget_show(d->breadcrumb_box);

  // find which step index corresponds to current page
  int current_step = -1;
  for(int i = 0; i < _num_steps; i++)
    if(_step_to_page[i] == d->current_page)
    {
      current_step = i;
      break;
    }

  for(int i = 0; i < _num_steps; i++)
  {
    gchar *markup;
    if(i == current_step)
      markup = g_strdup_printf("<b>%s</b>", _(_step_labels[i]));
    else
      markup = g_strdup_printf("<span alpha='40%%'>%s</span>", _(_step_labels[i]));
    gtk_label_set_markup(GTK_LABEL(d->breadcrumb_labels[i]), markup);
    g_free(markup);
  }
}


// ── navigation ───────────────────────────────────────────────────────

static int _next_page(int current)
{
  const int next = current + 1;
  return next < PAGE_COUNT ? next : -1;
}

static int _prev_page(int current)
{
  const int prev = current - 1;
  return prev >= 0 ? prev : -1;
}

static void _skip_clicked(GtkButton *button, gpointer user_data)
{
  dt_welcome_t *d = user_data;
  gtk_dialog_response(GTK_DIALOG(d->dialog), GTK_RESPONSE_YES);
}

static void _navigate_to_page(dt_welcome_t *d)
{
  gtk_stack_set_visible_child_name(GTK_STACK(d->stack),
                                   _page_stack_names[d->current_page]);

  const gboolean is_landing = (d->current_page == PAGE_LANDING);
  const gboolean is_last = (_next_page(d->current_page) == -1);

  gtk_widget_set_visible(d->btn_back, !is_landing);
  gtk_widget_set_sensitive(d->btn_back, _prev_page(d->current_page) >= 0);

  if(is_landing)
    gtk_button_set_label(GTK_BUTTON(d->btn_next), _("set up darktable →"));
  else if(is_last)
    gtk_button_set_label(GTK_BUTTON(d->btn_next), _("launch darktable"));
  else
    gtk_button_set_label(GTK_BUTTON(d->btn_next), _("continue"));

  // show skip button on all pages except last (keep space allocation to prevent jump)
  if(!is_last)
  {
    gtk_widget_set_opacity(d->btn_skip, 1.0);
    gtk_widget_set_sensitive(d->btn_skip, TRUE);
    gtk_widget_show(d->btn_skip);
  }
  else
  {
    gtk_widget_set_opacity(d->btn_skip, 0.0);
    gtk_widget_set_sensitive(d->btn_skip, FALSE);
  }

  _update_breadcrumb(d);
}

static void _apply_theme_live(dt_welcome_t *d)
{
  if(d->theme_choice >= 0 && d->theme_choice < _num_themes)
    dt_gui_load_theme(_theme_names[d->theme_choice]);
}


// ── apply settings ───────────────────────────────────────────────────

static void _apply_settings(dt_welcome_t *d)
{
  // workflow — always scene-referred (sigmoid) for new users
  dt_conf_set_string("plugins/darkroom/workflow", "scene-referred (sigmoid)");

  // module groups
  static const char *module_presets[] = {
    "workflow: beginner",
    "workflow: scene-referred",
    "modules: all"
  };
  if(d->modules_choice >= 0 && d->modules_choice < 3)
    dt_conf_set_string("plugins/darkroom/modulegroups_preset",
                       module_presets[d->modules_choice]);

  // theme
  if(d->theme_choice >= 0 && d->theme_choice < _num_themes)
    dt_conf_set_string("ui_last/theme", _theme_names[d->theme_choice]);

  // base film roll directory
  const char *import_path = gtk_entry_get_text(GTK_ENTRY(d->import_entry));
  if(import_path && import_path[0])
    dt_conf_set_string("session/base_directory_pattern", import_path);
}


// ── main dialog ──────────────────────────────────────────────────────

static gboolean _welcome_idle_callback(gpointer data)
{
  dt_welcome_t *d = g_malloc0(sizeof(dt_welcome_t));

  GtkWindow *main_win = GTK_WINDOW(dt_ui_main_window(darktable.gui->ui));

  d->dialog = gtk_dialog_new_with_buttons(
    _("welcome to darktable"),
    main_win,
    GTK_DIALOG_DESTROY_WITH_PARENT | GTK_DIALOG_MODAL,
    NULL, NULL);

#ifdef GDK_WINDOWING_QUARTZ
  dt_osx_disallow_fullscreen(d->dialog);
#endif

  gtk_window_set_default_size(GTK_WINDOW(d->dialog), WELCOME_WIDTH, WELCOME_HEIGHT);
  gtk_window_set_position(GTK_WINDOW(d->dialog), GTK_WIN_POS_CENTER_ON_PARENT);
  gtk_widget_set_name(d->dialog, "welcome-dialog");

  // action area buttons (back + next only)
  d->btn_back = gtk_dialog_add_button(GTK_DIALOG(d->dialog),
                                      _("back"), GTK_RESPONSE_REJECT);
  d->btn_next = gtk_dialog_add_button(GTK_DIALOG(d->dialog),
                                      _("continue"), GTK_RESPONSE_ACCEPT);

  // top bar: breadcrumb (centered) + skip button (right)
  d->top_bar = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  GtkWidget *top_bar = d->top_bar;
  gtk_widget_set_margin_top(top_bar, 12);
  gtk_widget_set_margin_bottom(top_bar, 4);
  gtk_widget_set_margin_start(top_bar, 16);
  gtk_widget_set_margin_end(top_bar, 16);

  // breadcrumb in center
  d->breadcrumb_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 16);
  gtk_widget_set_halign(d->breadcrumb_box, GTK_ALIGN_CENTER);
  for(int i = 0; i < _num_steps; i++)
  {
    if(i > 0)
    {
      d->breadcrumb_seps[i] = gtk_label_new("›");
      gtk_widget_set_opacity(d->breadcrumb_seps[i], 0.3);
      gtk_box_pack_start(GTK_BOX(d->breadcrumb_box), d->breadcrumb_seps[i],
                         FALSE, FALSE, 0);
    }
    d->breadcrumb_labels[i] = gtk_label_new(_(_step_labels[i]));
    gtk_box_pack_start(GTK_BOX(d->breadcrumb_box), d->breadcrumb_labels[i],
                       FALSE, FALSE, 0);
  }
  gtk_box_pack_start(GTK_BOX(top_bar), d->breadcrumb_box, TRUE, TRUE, 0);

  // skip button at top right
  d->btn_skip = gtk_button_new_with_label(_("use defaults and start →"));
  gtk_widget_set_name(d->btn_skip, "welcome-skip");
  gtk_widget_set_valign(d->btn_skip, GTK_ALIGN_CENTER);
  g_signal_connect_data(d->btn_skip, "clicked",
                        G_CALLBACK(_skip_clicked), d, NULL, 0);
  gtk_box_pack_end(GTK_BOX(top_bar), d->btn_skip, FALSE, FALSE, 0);

  // build pages
  d->stack = gtk_stack_new();
  gtk_stack_set_transition_type(GTK_STACK(d->stack),
                                GTK_STACK_TRANSITION_TYPE_SLIDE_LEFT_RIGHT);
  gtk_widget_set_vexpand(d->stack, TRUE);

  gtk_stack_add_named(GTK_STACK(d->stack), _build_landing_page(d), "landing");
  gtk_stack_add_named(GTK_STACK(d->stack), _build_theme_page(d), "theme");
  gtk_stack_add_named(GTK_STACK(d->stack), _build_modules_page(d), "modules");
  gtk_stack_add_named(GTK_STACK(d->stack), _build_import_page(d), "import");
  gtk_stack_add_named(GTK_STACK(d->stack), _build_quickstart_page(d), "quickstart");
  gtk_stack_add_named(GTK_STACK(d->stack), _build_firstedit_page(d), "firstedit");
  gtk_stack_add_named(GTK_STACK(d->stack), _build_resources_page(d), "resources");

  GtkWidget *content = gtk_dialog_get_content_area(GTK_DIALOG(d->dialog));
  gtk_box_pack_start(GTK_BOX(content), top_bar, FALSE, FALSE, 0);
  gtk_box_pack_start(GTK_BOX(content), d->stack, TRUE, TRUE, 0);

  // welcome CSS — higher priority than darktable themes (USER + 1)
  GtkCssProvider *css = gtk_css_provider_new();
  // GTK4: gtk_css_provider_load_from_string(css, _welcome_css)
  gtk_css_provider_load_from_data(css, _welcome_css, -1, NULL);
  // GTK4: GdkDisplay + gtk_style_context_add_provider_for_display()
  GdkScreen *screen = gdk_screen_get_default();
  gtk_style_context_add_provider_for_screen(
    screen, GTK_STYLE_PROVIDER(css),
    GTK_STYLE_PROVIDER_PRIORITY_USER + 2);

  // GTK4: widgets visible by default, remove show_all
  gtk_widget_show_all(d->dialog);

  d->current_page = PAGE_LANDING;
  _navigate_to_page(d);

  // GTK4: gtk_dialog_run removed; convert to async response signal
  gboolean finished = FALSE;
  while(!finished)
  {
    const int resp = gtk_dialog_run(GTK_DIALOG(d->dialog));

    if(resp == GTK_RESPONSE_ACCEPT)
    {
      const int next = _next_page(d->current_page);
      if(next >= 0)
      {
        d->current_page = next;
        _navigate_to_page(d);
      }
      else
      {
        _apply_settings(d);
        finished = TRUE;
      }
    }
    else if(resp == GTK_RESPONSE_REJECT)
    {
      const int prev = _prev_page(d->current_page);
      if(prev >= 0)
      {
        d->current_page = prev;
        _navigate_to_page(d);
      }
    }
    else if(resp == GTK_RESPONSE_YES)
    {
      // use defaults and launch — apply current state and finish
      _apply_settings(d);
      finished = TRUE;
    }
    else if(resp == GTK_RESPONSE_DELETE_EVENT)
    {
      finished = TRUE;
    }
  }

  gtk_widget_destroy(d->dialog); // GTK4: gtk_window_destroy()
  gtk_style_context_remove_provider_for_screen(screen, GTK_STYLE_PROVIDER(css)); // GTK4: ..._for_display()
  g_object_unref(css);
  g_free(d);

  return G_SOURCE_REMOVE;
}

void dt_gui_welcome_detect_first_launch(const char *configdir)
{
  gchar *rc_path = g_build_filename(configdir, "darktablerc-common", NULL);
  _is_first_launch = !g_file_test(rc_path, G_FILE_TEST_EXISTS);
  g_free(rc_path);
}

void dt_gui_welcome_schedule(void)
{
  if(_is_first_launch)
    g_idle_add(_welcome_idle_callback, NULL);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
