/*
    This file is part of darktable,
    Copyright (C) 2025 darktable developers.

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

#include "bauhaus/bauhaus.h"
#include "common/debug.h"
#include "common/undo.h"
#include "common/segmentation.h"
#include "control/conf.h"
#include "control/control.h"
#include "develop/blend.h"
#include "develop/imageop.h"
#include "develop/masks.h"
#include "develop/openmp_maths.h"

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
  (void)x; // unused arg, keep compiler from complaining
  (void)y;
  (void)as;
  (void)index;
  (void)num_points;
}

static int _object_events_mouse_scrolled(dt_iop_module_t *module,
                                         const float pzx,
                                         const float pzy,
                                         const int up,
                                         const uint32_t state,
                                         dt_masks_form_t *form,
                                         const dt_mask_id_t parentid,
                                         dt_masks_form_gui_t *gui,
                                         const int index)
{
  if(dt_modifier_is(state, GDK_CONTROL_MASK))
  {
    // we try to change the opacity
    dt_masks_form_change_opacity(form, parentid, up ? 0.05f : -0.05f);

    return 1;
  }

return 0;
}

static int _object_events_button_pressed(dt_iop_module_t *module,
                                         float pzx, float pzy,
                                         const double pressure,
                                         const int which,
                                         const int type,
                                         const uint32_t state,
                                         dt_masks_form_t *form,
                                         const dt_mask_id_t parentid,
                                         dt_masks_form_gui_t *gui,
                                         const int index)
{
  if(type == GDK_2BUTTON_PRESS || type == GDK_3BUTTON_PRESS) return 1;
  if(!gui) return 0;

  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);

  if(gui->creation && which == 1)
  {
    if(!gui->guipoints)
      gui->guipoints = dt_masks_dynbuf_init(200000, "object guipoints");
    if(!gui->guipoints)
      return 1;
    if(!gui->guipoints_payload)
      gui->guipoints_payload = dt_masks_dynbuf_init(100000, "object guipoints_payload");
    if(!gui->guipoints_payload)
      return 1;
    dt_masks_dynbuf_add_2(gui->guipoints, pzx * wd, pzy * ht);
    dt_masks_dynbuf_add(gui->guipoints_payload, dt_modifier_is(state, GDK_SHIFT_MASK) ? 0 : 1);
    gui->guipoints_count++;

    dt_control_queue_redraw_center();
    return 1;
  }
  else if(gui->creation && which == 3)
  {
    dt_print(DT_DEBUG_ALWAYS, "[object] right click");
    // get the image preview
    dt_dev_pixelpipe_t *preview = darktable.develop->preview_pipe;
    dt_print(DT_DEBUG_ALWAYS, "[object] backbuf %ix%i", preview->backbuf_width, preview->backbuf_height);
    seg_image_t img;
    img.nx = preview->backbuf_width;
    img.ny = preview->backbuf_height;
    img.data = preview->backbuf;

    seg_params_t params;
    int n_masks = 0;
    seg_params_init(&params);
    params.model = dt_conf_get_string("plugins/darkroom/masks/segmentation/model");

    // load the GGML model
    seg_context_t* ctx = seg_load_model(&params);
    if (!ctx) {
      dt_print(DT_DEBUG_ALWAYS, "[object] failed to load segmentation model");  
      return 1;
    }

    // encode image
    if (!seg_compute_image_embeddings(ctx, &img, params.n_threads)) {
      dt_print(DT_DEBUG_ALWAYS, "[object] failed to encode image");
      seg_free(ctx);
      return 1;
    }

    // create array of seg_point_t from click_points
    const float *guipoints = dt_masks_dynbuf_buffer(gui->guipoints);
    const float *guipoints_payload = dt_masks_dynbuf_buffer(gui->guipoints_payload);
    seg_point_t* seg_points = malloc(gui->guipoints_count * sizeof(seg_point_t));
    for (int i = 0; i < gui->guipoints_count; i++) {
        seg_points[i].x = guipoints[i * 2];
        seg_points[i].y = guipoints[i * 2 + 1];
        seg_points[i].label = guipoints_payload[i * 2];
    }

    // decode prompt
    seg_image_t* masks = seg_compute_masks(ctx, &img, params.n_threads, seg_points,
                                           gui->guipoints_count, &n_masks, 255, 0);
    if (!masks || n_masks == 0) {
      dt_print(DT_DEBUG_ALWAYS, "[object] failed to compute masks");
      seg_free(ctx);
      return 1;
    }
  
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
                                          const dt_mask_id_t parentid,
                                          dt_masks_form_gui_t *gui,
                                          const int index)
{
  return 0;
}

static int _object_events_mouse_moved(dt_iop_module_t *module,
                                      const float pzx,
                                      const float pzy,
                                      const double pressure,
                                      const int which,
                                      const float zoom_scale,
                                      dt_masks_form_t *form,
                                      const dt_mask_id_t parentid,
                                      dt_masks_form_gui_t *gui,
                                      const int index)
{
  gui->form_selected = FALSE;
  gui->border_selected = FALSE;
  gui->source_selected = FALSE;
  gui->feather_selected = -1;
  gui->point_selected = -1;
  gui->seg_selected = -1;
  gui->point_border_selected = -1;
    
  if(gui->creation)
  {
    dt_control_queue_redraw_center();
  }

  return 1;
}

static int _object_get_points(dt_develop_t *dev,
                              const float x,
                              const float y,
                              const float radius,
                              const float radius2,
                              const float rotation,
                              float **points,
                              int *points_count)
{
  (void)dev; // keep compiler from complaining about unused arg
  (void)x;
  (void)y;
  (void)radius;
  (void)radius2;
  (void)rotation;
  *points = NULL;
  *points_count = 0;
  return 0;
}

static void _object_events_post_expose(cairo_t *cr,
                                       const float zoom_scale,
                                       dt_masks_form_gui_t *gui,
                                       const int index,
                                       const int num_points)
{
  if(!gui) return;

  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);

  // in creation mode
  if(gui->creation)
  {
    const float opacity = 1.0f;
    const float radius = DT_PIXEL_APPLY_DPI(5.0f) / zoom_scale;
    const float sign_size = DT_PIXEL_APPLY_DPI(8.0f) / zoom_scale;

    cairo_save(cr);

    dt_gui_gtk_set_source_rgba(cr, DT_GUI_COLOR_BRUSH_CURSOR, opacity);
    cairo_arc(cr, gui->posx, gui->posy, radius, 0, 2.0 * M_PI);
    cairo_fill_preserve(cr);

    cairo_restore(cr);

    if(gui->guipoints_count > 0)
    {
      const float *guipoints = dt_masks_dynbuf_buffer(gui->guipoints);
      const float *guipoints_payload = dt_masks_dynbuf_buffer(gui->guipoints_payload);

      for(int i = 0; i < gui->guipoints_count; i++)
      {
        dt_draw_set_color_overlay(cr, TRUE, 0.8);
        cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(2.0) / zoom_scale);
        
        if(guipoints_payload[i] > 0) // Positive point - draw "+"
        {
          cairo_move_to(cr, guipoints[i * 2] - sign_size, guipoints[i * 2 + 1]);
          cairo_line_to(cr, guipoints[i * 2] + sign_size, guipoints[i * 2 + 1]);
          cairo_stroke(cr);
          
          cairo_move_to(cr, guipoints[i * 2], guipoints[i * 2 + 1] - sign_size);
          cairo_line_to(cr, guipoints[i * 2], guipoints[i * 2 + 1] + sign_size);
          cairo_stroke(cr);
        }
        else // Negative point - draw "-"
        {
          cairo_move_to(cr, guipoints[i * 2] - sign_size, guipoints[i * 2 + 1]);
          cairo_line_to(cr, guipoints[i * 2] + sign_size, guipoints[i * 2 + 1]);
          cairo_stroke(cr);
        }
      }
    }

    return;
  } // creation
}

static int _object_get_points_border(dt_develop_t *dev,
                                     struct dt_masks_form_t *form,
                                     float **points,
                                     int *points_count,
                                     float **border,
                                     int *border_count,
                                     const int source,
                                     const dt_iop_module_t *module)
{
  return 0;
}

static int _object_get_source_area(dt_iop_module_t *module,
                                   dt_dev_pixelpipe_iop_t *piece,
                                   dt_masks_form_t *form,
                                   int *width,
                                   int *height,
                                   int *posx,
                                   int *posy)
{
  return 1;
}

static int _object_get_area(const dt_iop_module_t *const restrict module,
                            const dt_dev_pixelpipe_iop_t *const restrict piece,
                            dt_masks_form_t *const restrict form,
                            int *width,
                            int *height,
                            int *posx,
                            int *posy)
{
  (void)module; // unused arg, keep compiler from complaining
  (void)piece;
  (void)form;
  return 1;
}

static int _object_get_mask(const dt_iop_module_t *const restrict module,
                            const dt_dev_pixelpipe_iop_t *const restrict piece,
                            dt_masks_form_t *const restrict form,
                            float **buffer,
                            int *width,
                            int *height,
                            int *posx,
                            int *posy)
{
  (void)module; // unused arg, keep compiler from complaining
  (void)piece;
  (void)form;
  (void)buffer;
  return 1;
}


static int _object_get_mask_roi(const dt_iop_module_t *const restrict module,
                                const dt_dev_pixelpipe_iop_t *const restrict piece,
                                dt_masks_form_t *const form,
                                const dt_iop_roi_t *const roi,
                                float *const restrict buffer)
{
  (void)module; // unused arg, keep compiler from complaining
  (void)piece;
  (void)form;
  (void)roi;
  (void)buffer;
  return 1;
}

static GSList *_object_setup_mouse_actions(const struct dt_masks_form_t *const form)
{
  GSList *lm = NULL;
  lm = dt_mouse_action_create_simple(lm, DT_MOUSE_ACTION_SCROLL,
                                     GDK_CONTROL_MASK, _("[OBJECT] change opacity"));
  return lm;
}

static void _object_sanitize_config(dt_masks_type_t type)
{
  (void)type; // unused arg, keep compiler from complaining
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
  if(gui->creation)
    g_snprintf(msgbuf, msgbuf_len,
              _("<b>add</b>: click, <b>substract</b>: shift+click\n"
                "<b>opacity</b>: ctrl+scroll (%d%%)"), opacity);
}

static void _object_duplicate_points(dt_develop_t *dev,
                                     dt_masks_form_t *const base,
                                     dt_masks_form_t *const dest)
{
  (void)dev; // unused arg, keep compiler from complaining
  (void)base;
  (void)dest;
}

static void _object_modify_property(dt_masks_form_t *const form,
                                    const dt_masks_property_t prop,
                                    const float old_val,
                                    const float new_val,
                                    float *sum,
                                    int *count,
                                    float *min,
                                    float *max)
{
  (void)form; // unused arg, keep compiler from complaining
  (void)prop;
  (void)old_val;
  (void)new_val;
}

static void _object_initial_source_pos(const float iwd,
                                       const float iht,
                                       float *x,
                                       float *y)
{
  (void)iwd; // unused arg, keep compiler from complaining
  (void)iht;
}

// The function table for objects.  This must be public, i.e. no "static" keyword.
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
