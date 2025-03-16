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
  return 0;
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
  (void)zoom_scale; // unused arg, keep compiler from complaining
  (void)index;
  (void)num_points;
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
                                     0, _("[OBJECT] change size"));
  lm = dt_mouse_action_create_simple(lm, DT_MOUSE_ACTION_SCROLL,
                                     GDK_SHIFT_MASK, _("[OBJECT] change feather size"));
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
  // object has same controls on creation and on edit
  g_snprintf(msgbuf, msgbuf_len,
             _("<b>size</b>: scroll, <b>feather size</b>: shift+scroll\n"
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
