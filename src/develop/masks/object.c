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

#include "common/debug.h"
#include "control/conf.h"
#include "control/control.h"
#include "gui/gtk.h"
#include "ai/segmentation.h"
#include "develop/blend.h"
#include "develop/imageop.h"
#include "develop/masks.h"
#include "develop/openmp_maths.h"
#include "common/ras2vect.h"

#include <math.h>
#include <string.h>

// --- Per-session segmentation state (stored in gui->scratchpad) ---

typedef struct _object_data_t
{
  dt_ai_environment_t *env;  // AI environment for model registry
  dt_seg_context_t *seg;     // SAM context (encoder+decoder)
  float *mask;               // current mask buffer (preview pipe size)
  int mask_w, mask_h;        // mask dimensions
  gboolean model_loaded;     // whether the model was loaded
  int encode_state;          // 0=idle, 1=msg shown, 2=ready, -1=error
  dt_imgid_t encoded_imgid;  // image ID that was encoded
  guint modifier_poll_id;    // timer to detect shift key changes
} _object_data_t;

static _object_data_t *_get_data(dt_masks_form_gui_t *gui)
{
  return (gui && gui->scratchpad) ? (_object_data_t *)gui->scratchpad : NULL;
}

static void _free_data(dt_masks_form_gui_t *gui)
{
  _object_data_t *d = _get_data(gui);
  if(!d) return;
  if(d->modifier_poll_id)
    g_source_remove(d->modifier_poll_id);
  if(d->seg) dt_seg_free(d->seg);
  if(d->env) dt_ai_env_destroy(d->env);
  g_free(d->mask);
  g_free(d);
  gui->scratchpad = NULL;
}

// Ensure the model is loaded and the preview image is encoded.
// Returns the _object_data_t or NULL on failure.
static _object_data_t *_ensure_encoded(dt_masks_form_gui_t *gui)
{
  _object_data_t *d = _get_data(gui);
  if(!d)
  {
    d = g_new0(_object_data_t, 1);
    gui->scratchpad = d;
  }

  if(!d->model_loaded)
  {
    if(!d->env)
      d->env = dt_ai_env_init(NULL);

    d->seg = dt_seg_load(d->env, "mask-light-hq-sam");

    if(!d->seg)
    {
      dt_print(DT_DEBUG_AI, "[object mask] Failed to load segmentation model");
      dt_control_log(_("AI segmentation model not found"));
      return NULL;
    }
    d->model_loaded = TRUE;
  }

  // Encode preview image if not done yet
  if(d->seg && !dt_seg_is_encoded(d->seg))
  {
    dt_dev_pixelpipe_t *preview = darktable.develop->preview_pipe;
    if(!preview || !preview->backbuf)
    {
      dt_print(DT_DEBUG_AI, "[object mask] Preview buffer not available");
      return NULL;
    }

    // Convert preview backbuf (uint8 RGBA) to uint8 RGB
    dt_pthread_mutex_lock(&preview->backbuf_mutex);
    const int pw = preview->backbuf_width;
    const int ph = preview->backbuf_height;
    const uint8_t *src = preview->backbuf;
    uint8_t *rgb = NULL;

    if(src && pw > 0 && ph > 0)
    {
      rgb = g_try_malloc((size_t)pw * ph * 3);
      if(rgb)
      {
        for(int i = 0; i < pw * ph; i++)
        {
          rgb[i * 3 + 0] = src[i * 4 + 2]; // R (backbuf is BGRA)
          rgb[i * 3 + 1] = src[i * 4 + 1]; // G
          rgb[i * 3 + 2] = src[i * 4 + 0]; // B
        }
      }
    }
    dt_pthread_mutex_unlock(&preview->backbuf_mutex);

    if(!rgb)
      return NULL;

    const gboolean ok = dt_seg_encode_image(d->seg, rgb, pw, ph);
    g_free(rgb);

    if(!ok)
    {
      dt_control_log(_("AI image encoding failed"));
      return NULL;
    }
  }

  return d;
}

// Run the decoder with accumulated points and update the cached mask
static void _run_decoder(dt_masks_form_gui_t *gui)
{
  _object_data_t *d = _get_data(gui);
  if(!d || !d->seg || !dt_seg_is_encoded(d->seg)) return;
  if(gui->guipoints_count <= 0) return;

  const float *gp = dt_masks_dynbuf_buffer(gui->guipoints);
  const float *gpp = dt_masks_dynbuf_buffer(gui->guipoints_payload);

  dt_seg_point_t *points = g_new(dt_seg_point_t, gui->guipoints_count);
  for(int i = 0; i < gui->guipoints_count; i++)
  {
    points[i].x = gp[i * 2 + 0];
    points[i].y = gp[i * 2 + 1];
    points[i].label = (int)gpp[i];
  }

  int mw, mh;
  float *mask = dt_seg_compute_mask(d->seg, points, gui->guipoints_count,
                                     &mw, &mh);
  g_free(points);

  if(mask)
  {
    g_free(d->mask);
    d->mask = mask;
    d->mask_w = mw;
    d->mask_h = mh;
  }
}

// Finalize: vectorize the mask and register as a group of path forms
static void _finalize_mask(dt_iop_module_t *module,
                           dt_masks_form_t *form,
                           dt_masks_form_gui_t *gui)
{
  _object_data_t *d = _get_data(gui);
  if(!d || !d->mask) return;

  const dt_image_t *const image = &(darktable.develop->image_storage);

  // Invert mask for potrace (potrace traces dark regions)
  // Our mask: 1.0 = foreground; potrace expects: 0.0 = foreground
  const size_t n = (size_t)d->mask_w * d->mask_h;
  float *inv_mask = g_try_malloc(n * sizeof(float));
  if(!inv_mask) return;

  for(size_t i = 0; i < n; i++)
    inv_mask[i] = 1.0f - d->mask[i];

  GList *forms = ras2forms(inv_mask, d->mask_w, d->mask_h, image);
  g_free(inv_mask);

  const int nbform = g_list_length(forms);
  if(nbform == 0)
  {
    dt_control_log(_("no mask extracted from AI segmentation"));
    return;
  }

  // Name the forms with AI prefix
  static int ai_group_counter = 0;
  ai_group_counter++;

  for(GList *l = forms; l; l = g_list_next(l))
  {
    dt_masks_form_t *f = l->data;
    snprintf(f->name, sizeof(f->name), "AI mask %d", ai_group_counter);
  }

  dt_masks_register_forms(darktable.develop, forms);

  dt_control_log(ngettext("%d AI mask created",
                          "%d AI masks created", nbform), nbform);
}

// --- Mask Event Handlers ---

static void _object_get_distance(const float x,
                                 const float y,
                                 const float as,
                                 dt_masks_form_gui_t *gui,
                                 const int index,
                                 const int num_points,
                                 gboolean *inside,
                                 gboolean *inside_border,
                                 int *near,
                                 gboolean *inside_source,
                                 float *dist)
{
  (void)x; (void)y; (void)as; (void)gui;
  (void)index; (void)num_points;
  (void)inside; (void)inside_border; (void)near;
  (void)inside_source; (void)dist;
}

static int _object_events_mouse_scrolled(dt_iop_module_t *module,
                                         const float pzx,
                                         const float pzy,
                                         const gboolean up,
                                         const uint32_t state,
                                         dt_masks_form_t *form,
                                         const dt_imgid_t parentid,
                                         dt_masks_form_gui_t *gui,
                                         const int index)
{
  if(dt_modifier_is(state, GDK_CONTROL_MASK))
  {
    dt_masks_form_change_opacity(form, parentid, up ? 0.05f : -0.05f);
    return 1;
  }
  return 0;
}

// Clear accumulated points, mask preview, and iterative refinement state
static void _clear_selection(dt_masks_form_gui_t *gui)
{
  _object_data_t *d = _get_data(gui);
  if(!d) return;

  if(gui->guipoints)
    dt_masks_dynbuf_reset(gui->guipoints);
  if(gui->guipoints_payload)
    dt_masks_dynbuf_reset(gui->guipoints_payload);
  gui->guipoints_count = 0;

  g_free(d->mask);
  d->mask = NULL;
  d->mask_w = d->mask_h = 0;

  if(d->seg) dt_seg_reset_prev_mask(d->seg);

  dt_control_queue_redraw_center();
}

static int _object_events_button_pressed(dt_iop_module_t *module,
                                         float pzx, float pzy,
                                         const double pressure,
                                         const int which,
                                         const int type,
                                         const uint32_t state,
                                         dt_masks_form_t *form,
                                         const dt_imgid_t parentid,
                                         dt_masks_form_gui_t *gui,
                                         const int index)
{
  (void)pressure; (void)parentid; (void)index;
  if(type == GDK_2BUTTON_PRESS || type == GDK_3BUTTON_PRESS) return 1;
  if(!gui) return 0;

  if(gui->creation && which == 1 && dt_modifier_is(state, GDK_MOD1_MASK))
  {
    // Alt+click: clear selection
    _object_data_t *d = _get_data(gui);
    if(d && d->encode_state == 2 && gui->guipoints_count > 0)
      _clear_selection(gui);
    return 1;
  }
  else if(gui->creation && which == 1)
  {
    // Only accept clicks after encoding is complete
    _object_data_t *d = _get_data(gui);
    if(!d || d->encode_state != 2) return 1;

    // Initialize point buffers
    if(!gui->guipoints)
      gui->guipoints = dt_masks_dynbuf_init(200000, "object guipoints");
    if(!gui->guipoints) return 1;
    if(!gui->guipoints_payload)
      gui->guipoints_payload = dt_masks_dynbuf_init(100000, "object guipoints_payload");
    if(!gui->guipoints_payload) return 1;

    // Coordinates in preview pipe pixel space
    float wd, ht, iwidth, iheight;
    dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);

    dt_masks_dynbuf_add_2(gui->guipoints, pzx * wd, pzy * ht);
    dt_masks_dynbuf_add(gui->guipoints_payload,
                        dt_modifier_is(state, GDK_SHIFT_MASK) ? 0.0f : 1.0f);
    gui->guipoints_count++;

    // Run decoder to update live preview
    _run_decoder(gui);

    dt_control_queue_redraw_center();
    return 1;
  }
  else if(gui->creation && which == 3)
  {
    // Right-click: finalize mask
    if(gui->guipoints_count > 0)
      _finalize_mask(module, form, gui);

    // Cleanup and exit creation mode
    _free_data(gui);

    dt_masks_dynbuf_free(gui->guipoints);
    dt_masks_dynbuf_free(gui->guipoints_payload);
    gui->guipoints = NULL;
    gui->guipoints_payload = NULL;
    gui->guipoints_count = 0;

    gui->creation_continuous = FALSE;
    gui->creation_continuous_module = NULL;

    dt_masks_set_edit_mode(module, DT_MASKS_EDIT_FULL);
    dt_masks_iop_update(module);
    dt_control_queue_redraw_center();
    return 1;
  }

  return 0;
}

static int _object_events_button_released(dt_iop_module_t *module,
                                          const float pzx,
                                          const float pzy,
                                          const int which,
                                          const uint32_t state,
                                          dt_masks_form_t *form,
                                          const dt_imgid_t parentid,
                                          dt_masks_form_gui_t *gui,
                                          const int index)
{
  (void)module; (void)pzx; (void)pzy; (void)which; (void)state;
  (void)form; (void)parentid; (void)gui; (void)index;
  return 0;
}

static int _object_events_mouse_moved(dt_iop_module_t *module,
                                      const float pzx,
                                      const float pzy,
                                      const double pressure,
                                      const int which,
                                      const float zoom_scale,
                                      dt_masks_form_t *form,
                                      const dt_imgid_t parentid,
                                      dt_masks_form_gui_t *gui,
                                      const int index)
{
  (void)module; (void)pressure; (void)which; (void)zoom_scale;
  (void)form; (void)parentid; (void)index;

  if(!gui) return 0;

  gui->form_selected = FALSE;
  gui->border_selected = FALSE;
  gui->source_selected = FALSE;
  gui->feather_selected = -1;
  gui->point_selected = -1;
  gui->seg_selected = -1;
  gui->point_border_selected = -1;

  if(gui->creation)
    dt_control_queue_redraw_center();

  return 1;
}

// Timer callback: periodically redraw center so +/- cursor tracks shift key
static gboolean _modifier_poll(gpointer data)
{
  (void)data;
  dt_control_queue_redraw_center();
  return G_SOURCE_CONTINUE;
}

static void _object_events_post_expose(cairo_t *cr,
                                       const float zoom_scale,
                                       dt_masks_form_gui_t *gui,
                                       const int index,
                                       const int num_points)
{
  (void)index; (void)num_points;
  if(!gui) return;
  if(!gui->creation) return;

  // Ensure scratchpad exists
  _object_data_t *d = _get_data(gui);
  if(!d)
  {
    d = g_new0(_object_data_t, 1);
    gui->scratchpad = d;
  }

  // Detect image change: reset encoding if we switched to a different image
  const dt_imgid_t cur_imgid = darktable.develop->image_storage.id;
  if(d->encode_state == 2 && d->encoded_imgid != cur_imgid)
  {
    if(d->seg) dt_seg_reset_encoding(d->seg);
    g_free(d->mask);
    d->mask = NULL;
    d->mask_w = d->mask_h = 0;
    d->encode_state = 0;
  }

  // Eager encoding: load model and encode image as soon as tool opens
  if(d->encode_state == 0)
  {
    // Frame 1: show log message and return so it renders before we block
    dt_control_log(_("preparing AI mask..."));
    d->encode_state = 1;
    dt_control_queue_redraw_center();
    return;
  }

  if(d->encode_state == 1)
  {
    // Frame 2: log message is visible from frame 1, now block on encoding
    _ensure_encoded(gui);
    if(d->model_loaded && d->seg && dt_seg_is_encoded(d->seg))
    {
      d->encode_state = 2;
      d->encoded_imgid = cur_imgid;
      // Start modifier polling so +/- cursor tracks shift key changes
      if(!d->modifier_poll_id)
        d->modifier_poll_id = g_timeout_add(100, _modifier_poll, NULL);
    }
    else
      d->encode_state = -1;
    dt_control_queue_redraw_center();
    return;
  }

  if(d->encode_state != 2) return;

  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);

  // Draw reddish mask overlay
  if(d->mask && d->mask_w > 0 && d->mask_h > 0)
  {
    const int mw = d->mask_w;
    const int mh = d->mask_h;
    const int stride = cairo_format_stride_for_width(CAIRO_FORMAT_ARGB32, mw);
    unsigned char *buf = g_try_malloc0((size_t)stride * mh);
    if(buf)
    {
      for(int y = 0; y < mh; y++)
      {
        unsigned char *row = buf + y * stride;
        for(int x = 0; x < mw; x++)
        {
          const float val = d->mask[y * mw + x];
          if(val > 0.5f)
          {
            const unsigned char alpha = 80;
            row[x * 4 + 0] = 0;            // B
            row[x * 4 + 1] = 0;            // G
            row[x * 4 + 2] = alpha;        // R (premultiplied)
            row[x * 4 + 3] = alpha;        // A
          }
        }
      }

      cairo_surface_t *surface = cairo_image_surface_create_for_data(
          buf, CAIRO_FORMAT_ARGB32, mw, mh, stride);

      if(surface)
      {
        cairo_save(cr);
        cairo_scale(cr, wd / mw, ht / mh);
        cairo_set_source_surface(cr, surface, 0, 0);
        cairo_paint(cr);
        cairo_restore(cr);
        cairo_surface_destroy(surface);
      }
      g_free(buf);
    }
  }

  // Query pointer modifier state reliably (gdk_keymap doesn't work on macOS)
  GtkWidget *cw = dt_ui_center(darktable.gui->ui);
  GdkWindow *win = gtk_widget_get_window(cw);
  GdkDevice *pointer = gdk_seat_get_pointer(
      gdk_display_get_default_seat(gdk_display_get_default()));
  GdkModifierType mod = 0;
  if(win && pointer)
    gdk_window_get_device_position(win, pointer, NULL, NULL, &mod);
  const gboolean shift_held = (mod & GDK_SHIFT_MASK) != 0;

  // Draw +/- cursor indicator (white, like other mask controls)
  const float r = DT_PIXEL_APPLY_DPI(8.0f) / zoom_scale;
  const float lw = DT_PIXEL_APPLY_DPI(2.0f) / zoom_scale;
  cairo_set_line_width(cr, lw);
  cairo_set_source_rgba(cr, 0.9, 0.9, 0.9, 0.9);

  // Horizontal line (common to both + and -)
  cairo_move_to(cr, gui->posx - r, gui->posy);
  cairo_line_to(cr, gui->posx + r, gui->posy);
  cairo_stroke(cr);

  if(!shift_held)
  {
    // Add mode: vertical line to form "+"
    cairo_move_to(cr, gui->posx, gui->posy - r);
    cairo_line_to(cr, gui->posx, gui->posy + r);
    cairo_stroke(cr);
  }
}

// --- Stub functions (object is transient — result is path masks) ---

static int _object_get_points(dt_develop_t *dev,
                              const float x, const float y,
                              const float radius, const float radius2,
                              const float rotation,
                              float **points, int *points_count)
{
  (void)dev; (void)x; (void)y; (void)radius; (void)radius2;
  (void)rotation;
  *points = NULL;
  *points_count = 0;
  return 0;
}

static int _object_get_points_border(dt_develop_t *dev,
                                     struct dt_masks_form_t *form,
                                     float **points, int *points_count,
                                     float **border, int *border_count,
                                     const int source,
                                     const dt_iop_module_t *module)
{
  (void)dev; (void)form; (void)points; (void)points_count;
  (void)border; (void)border_count; (void)source; (void)module;
  return 0;
}

static int _object_get_source_area(dt_iop_module_t *module,
                                   dt_dev_pixelpipe_iop_t *piece,
                                   dt_masks_form_t *form,
                                   int *width, int *height,
                                   int *posx, int *posy)
{
  (void)module; (void)piece; (void)form;
  (void)width; (void)height; (void)posx; (void)posy;
  return 1;
}

static int _object_get_area(const dt_iop_module_t *const restrict module,
                            const dt_dev_pixelpipe_iop_t *const restrict piece,
                            dt_masks_form_t *const restrict form,
                            int *width, int *height, int *posx, int *posy)
{
  (void)module; (void)piece; (void)form;
  (void)width; (void)height; (void)posx; (void)posy;
  return 1;
}

static int _object_get_mask(const dt_iop_module_t *const restrict module,
                            const dt_dev_pixelpipe_iop_t *const restrict piece,
                            dt_masks_form_t *const restrict form,
                            float **buffer,
                            int *width, int *height, int *posx, int *posy)
{
  (void)module; (void)piece; (void)form; (void)buffer;
  (void)width; (void)height; (void)posx; (void)posy;
  return 1;
}

static int _object_get_mask_roi(const dt_iop_module_t *const restrict module,
                                const dt_dev_pixelpipe_iop_t *const restrict piece,
                                dt_masks_form_t *const form,
                                const dt_iop_roi_t *const roi,
                                float *const restrict buffer)
{
  (void)module; (void)piece; (void)form; (void)roi; (void)buffer;
  return 1;
}

static GSList *_object_setup_mouse_actions(const struct dt_masks_form_t *const form)
{
  (void)form;
  GSList *lm = NULL;
  lm = dt_mouse_action_create_simple(lm, DT_MOUSE_ACTION_LEFT,
                                     0, _("[OBJECT] add foreground point"));
  lm = dt_mouse_action_create_simple(lm, DT_MOUSE_ACTION_LEFT,
                                     GDK_SHIFT_MASK,
                                     _("[OBJECT] add background point"));
  lm = dt_mouse_action_create_simple(lm, DT_MOUSE_ACTION_RIGHT,
                                     0, _("[OBJECT] apply mask"));
  lm = dt_mouse_action_create_simple(lm, DT_MOUSE_ACTION_SCROLL,
                                     GDK_CONTROL_MASK,
                                     _("[OBJECT] change opacity"));
  return lm;
}

static void _object_sanitize_config(dt_masks_type_t type)
{
  (void)type;
}

static void _object_set_form_name(dt_masks_form_t *const form,
                                  const size_t nb)
{
  snprintf(form->name, sizeof(form->name), _("object #%d"), (int)nb);
}

static void _object_set_hint_message(const dt_masks_form_gui_t *const gui,
                                     const dt_masks_form_t *const form,
                                     const int opacity,
                                     char *const restrict msgbuf,
                                     const size_t msgbuf_len)
{
  (void)form;
  if(gui->creation)
    g_snprintf(msgbuf, msgbuf_len,
               _("<b>add</b>: click, <b>subtract</b>: shift+click, "
                 "<b>clear</b>: alt+click, <b>apply</b>: right-click\n"
                 "<b>opacity</b>: ctrl+scroll (%d%%)"), opacity);
}

static void _object_duplicate_points(dt_develop_t *dev,
                                     dt_masks_form_t *const base,
                                     dt_masks_form_t *const dest)
{
  (void)dev; (void)base; (void)dest;
}

static void _object_modify_property(dt_masks_form_t *const form,
                                    const dt_masks_property_t prop,
                                    const float old_val, const float new_val,
                                    float *sum, int *count,
                                    float *min, float *max)
{
  (void)form; (void)prop; (void)old_val; (void)new_val;
  (void)sum; (void)count; (void)min; (void)max;
}

static void _object_initial_source_pos(const float iwd, const float iht,
                                       float *x, float *y)
{
  (void)iwd; (void)iht; (void)x; (void)y;
}

// The function table for object masks
const dt_masks_functions_t dt_masks_functions_object = {
  .point_struct_size = sizeof(struct dt_masks_point_object_t),
  .sanitize_config = _object_sanitize_config,
  .setup_mouse_actions = _object_setup_mouse_actions,
  .set_form_name = _object_set_form_name,
  .set_hint_message = _object_set_hint_message,
  .modify_property = _object_modify_property,
  .duplicate_points = _object_duplicate_points,
  .initial_source_pos = _object_initial_source_pos,
  .get_distance = _object_get_distance,
  .get_points = _object_get_points,
  .get_points_border = _object_get_points_border,
  .get_mask = _object_get_mask,
  .get_mask_roi = _object_get_mask_roi,
  .get_area = _object_get_area,
  .get_source_area = _object_get_source_area,
  .mouse_moved = _object_events_mouse_moved,
  .mouse_scrolled = _object_events_mouse_scrolled,
  .button_pressed = _object_events_button_pressed,
  .button_released = _object_events_button_released,
  .post_expose = _object_events_post_expose
};

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
