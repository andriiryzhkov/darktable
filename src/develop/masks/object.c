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

#define MIN_POINT_RADIUS 0.005f

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
  // initialise returned values
  *inside_source = FALSE;
  *inside = FALSE;
  *inside_border = FALSE;
  *near = -1;
  *dist = FLT_MAX;

  if(!gui) return;

  dt_masks_form_gui_points_t *gpt = g_list_nth_data(gui->points, index);
  if(!gpt) return;

  // we first check if we are inside the source form
  if(dt_masks_point_in_form_exact(x, y, gpt->source, 1, gpt->source_count))
  {
    // Object masks cannot be used with clone, so this should never happen
    return;
  }

  // Check if we are close to a point
  for(int i = 0; i < num_points; i++)
  {
    // points are in a flat array with 2 elements per point
    const float px = gpt->points[i * 2];
    const float py = gpt->points[i * 2 + 1];
    
    const float d = (x - px) * (x - px) + (y - py) * (y - py);
    
    if(d < as * as)
    {
      *inside = TRUE;
      *dist = d;
      *near = i;
      return;
    }
    
    if(d < *dist)
    {
      *dist = d;
    }
  }
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
  if(gui->form_selected)
  {
    // we register the current position
    if(gui->scrollx == 0.0f && gui->scrolly == 0.0f)
    {
      gui->scrollx = pzx;
      gui->scrolly = pzy;
    }
    if(dt_modifier_is(state, GDK_CONTROL_MASK))
    {
      // we try to change the opacity
      dt_masks_form_change_opacity(form, parentid, up ? 0.05f : -0.05f);
    }
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
  if(!gui) return 0;
  
  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);

  // dt_masks_form_gui_points_t *gpt = g_list_nth_data(gui->points, index);
  
  if(which == 1 && type == GDK_BUTTON_PRESS)
  {
    // Add a new point
    dt_masks_point_object_t *point = malloc(sizeof(dt_masks_point_object_t));
    
    // Get the point coordinates
    float pts[2] = { pzx * wd, pzy * ht };
    dt_dev_distort_backtransform(darktable.develop, pts, 1);
    
    point->point[0] = pts[0] / iwidth;
    point->point[1] = pts[1] / iheight;
    
    // Set the label (1 for positive, 0 for negative)
    point->label = dt_modifier_is(state, GDK_SHIFT_MASK) ? 0 : 1;
    
    // Add the point to form
    form->points = g_list_append(form->points, point);
    
    // If this is a clone mask, set source position if it's the first point
    if((form->type & DT_MASKS_CLONE) && g_list_length(form->points) == 1)
    {
      dt_masks_set_source_pos_initial_value(gui, DT_MASKS_OBJECT, form, pzx, pzy);
    }
    
    // We save the move
    dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
    
    // We recreate the form points
    dt_masks_gui_form_create(form, gui, index, module);
    
    dt_control_queue_redraw_center();
    return 1;
  }
  else if(which == 3 && type == GDK_BUTTON_PRESS)
  {
    // Right click - clear all points
    while(form->points)
    {
      dt_masks_point_object_t *point = form->points->data;
      form->points = g_list_remove(form->points, point);
      free(point);
    }
    
    // We save the change
    dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
    
    // We recreate the form points
    dt_masks_gui_form_create(form, gui, index, module);
    
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
  if(!gui) return 0;
  if(gui->form_dragging)
  {
    // We end the form dragging
    gui->form_dragging = FALSE;
    
    // Save the changes
    dt_dev_add_masks_history_item(darktable.develop, module, TRUE);
    
    return 1;
  }
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
  if(!gui) return 0;
  
  float wd, ht;
  dt_masks_get_image_size(&wd, &ht, NULL, NULL);
  
  // Record mouse position even if there are no masks visible
  if(gui)
  {
    gui->posx = pzx * wd;
    gui->posy = pzy * ht;
  }
  
  dt_masks_form_gui_points_t *gpt = g_list_nth_data(gui->points, index);
  if(!gpt) return 0;
  
  // Check if we are near a point
  const float as = dt_masks_sensitive_dist(zoom_scale);
  
  // get number of points in the form
  const int nb_points = g_list_length(form->points);
  
  gboolean in, inb, ins;
  int near = -1;
  float dist = 0;
  
  _object_get_distance(pzx * wd, pzy * ht, as, gui, index, nb_points, &in, &inb, &near, &ins, &dist);
  
  if(near >= 0)
  {
    gui->point_selected = near;
    gui->form_selected = TRUE;
  }
  else
  {
    gui->point_selected = -1;
    
    // Check if we're still inside the form and can drag it
    if(in)
    {
      gui->form_selected = TRUE;
    }
    else
    {
      gui->form_selected = FALSE;
    }
  }
  
  dt_control_queue_redraw_center();
  if(!gui->form_selected) return 0;
  if(gui->edit_mode != DT_MASKS_EDIT_FULL) return 0;
  
  return 1;
}

static void _object_draw_points(cairo_t *cr,
                              float zoom_scale,
                              dt_masks_form_gui_t *gui,
                              int index,
                              dt_masks_form_t *form)
{
  double dashed[] = { 4.0, 4.0 };
  dashed[0] /= zoom_scale;
  dashed[1] /= zoom_scale;
  
  dt_masks_form_gui_points_t *gpt = g_list_nth_data(gui->points, index);
  if(!gpt) return;
  
  // Get number of points
  const int nb = gpt->points_count / 2; // Each point has 2 coordinates (x,y)
  
  // Draw each point as an anchor
  for(int i = 0; i < nb; i++)
  {
    // Draw the anchor
    dt_masks_draw_anchor(cr,
                       gui->point_selected == i,
                       zoom_scale,
                       gpt->points[i * 2],
                       gpt->points[i * 2 + 1]);
    
    // Draw the +/- sign above the anchor
    const float x = gpt->points[i * 2];
    const float y = gpt->points[i * 2 + 1];
    const float sign_size = DT_PIXEL_APPLY_DPI(5.0f) / zoom_scale;
    
    // Get the point from the form to check its label
    dt_masks_point_object_t *pt = g_list_nth_data(form->points, i);
    
    if(pt)
    {
      if(pt->label == 1) // Positive point
      {
        // Draw '+' sign
        cairo_move_to(cr, x - sign_size, y - sign_size - sign_size);
        cairo_line_to(cr, x + sign_size, y - sign_size - sign_size);
        cairo_move_to(cr, x, y - sign_size - sign_size - sign_size);
        cairo_line_to(cr, x, y - sign_size - sign_size + sign_size);
      }
      else // Negative point
      {
        // Draw '-' sign
        cairo_move_to(cr, x - sign_size, y - sign_size - sign_size);
        cairo_line_to(cr, x + sign_size, y - sign_size - sign_size);
      }
      
      dt_masks_line_stroke(cr, FALSE, FALSE, gui->point_selected == i, zoom_scale);
    }
  }
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

  // This function is needed for the API but isn't actually used for object masks
  // since they don't have a specific geometric shape to calculate
  *points = NULL;
  points_count = 0;

  // Return failure to indicate this isn't implemented for object masks
  return 0;
}

static void _object_events_post_expose(cairo_t *cr,
                                       const float zoom_scale,
                                       dt_masks_form_gui_t *gui,
                                       const int index,
                                       const int num_points)
{
  if(!gui) return;
  
  dt_masks_form_gui_points_t *gpt = g_list_nth_data(gui->points, index);
  if(!gpt) return;
  
  // Draw the points
  dt_masks_form_t *form = darktable.develop->form_visible;
  _object_draw_points(cr, zoom_scale, gui, index, form);
}

// static void _bounding_box(const float *const points,
//                           const int num_points,
//                           int *width,
//                           int *height,
//                           int *posx,
//                           int *posy)
// {
//   // search for min/max X and Y coordinates
//   float xmin = FLT_MAX, xmax = FLT_MIN, ymin = FLT_MAX, ymax = FLT_MIN;
//   for(int i = 1; i < num_points; i++) // skip point[0], which is object's center
//   {
//     xmin = fminf(points[i * 2], xmin);
//     xmax = fmaxf(points[i * 2], xmax);
//     ymin = fminf(points[i * 2 + 1], ymin);
//     ymax = fmaxf(points[i * 2 + 1], ymax);
//   }
//   // set the min/max values we found
//   *posx = xmin;
//   *posy = ymin;
//   *width = (xmax - xmin);
//   *height = (ymax - ymin);
// }

static int _object_get_points_border(dt_develop_t *dev,
                                     struct dt_masks_form_t *form,
                                     float **points,
                                     int *points_count,
                                     float **border,
                                     int *border_count,
                                     const int source,
                                     const dt_iop_module_t *module)
{
  // Source is not supported for object masks
  if(source) return 0;
  
  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);
  
  // Count the number of points
  int nbp = g_list_length(form->points);
  if(nbp == 0) return 0;
  
  // Allocate arrays for points coordinates
  *points = dt_alloc_align_float((size_t)2 * nbp);
  if(border) *border = dt_alloc_align_float((size_t)2 * nbp);
  
  // Sanity check
  if(*points == NULL) return 0;
  if(border && *border == NULL)
  {
    dt_free_align(*points);
    *points = NULL;
    return 0;
  }
  
  // Fill arrays with points coordinates
  int i = 0;
  for(GList *l = form->points; l; l = g_list_next(l))
  {
    dt_masks_point_object_t *p = (dt_masks_point_object_t *)l->data;
    
    (*points)[i * 2] = p->point[0] * iwidth;
    (*points)[i * 2 + 1] = p->point[1] * iheight;
    
    // Border is same as point (simple circle around the point will be drawn)
    if(border)
    {
      (*border)[i * 2] = p->point[0] * iwidth;
      (*border)[i * 2 + 1] = p->point[1] * iheight;
    }
    
    i++;
  }
  
  *points_count = nbp;
  if(border) *border_count = nbp;
  
  // Transform the points with all distortion modules
  if(!dt_dev_distort_transform_plus(dev, dev->preview_pipe, module->iop_order, DT_DEV_TRANSFORM_DIR_ALL, *points, *points_count))
  {
    dt_free_align(*points);
    *points = NULL;
    *points_count = 0;
    
    if(border)
    {
      dt_free_align(*border);
      *border = NULL;
      *border_count = 0;
    }
    return 0;
  }
  
  if(border)
  {
    if(!dt_dev_distort_transform_plus(dev, dev->preview_pipe, module->iop_order, DT_DEV_TRANSFORM_DIR_ALL, *border, *border_count))
    {
      dt_free_align(*points);
      dt_free_align(*border);
      *points = NULL;
      *border = NULL;
      *points_count = 0;
      *border_count = 0;
      return 0;
    }
  }
  
  return 1;
}

static int _object_get_source_area(dt_iop_module_t *module,
                                   dt_dev_pixelpipe_iop_t *piece,
                                   dt_masks_form_t *form,
                                   int *width,
                                   int *height,
                                   int *posx,
                                   int *posy)
{
  // Object masks cannot be used with clone
  return 0;
}

static int _object_get_area(const dt_iop_module_t *const restrict module,
                            const dt_dev_pixelpipe_iop_t *const restrict piece,
                            dt_masks_form_t *const restrict form,
                            int *width,
                            int *height,
                            int *posx,
                            int *posy)
{
  // We get the points
  float *points = NULL, *border = NULL;
  int points_count, border_count;
  if(!_object_get_points_border(module->dev, form, &points, &points_count, &border, &border_count, FALSE, module))
  {
    return 0;
  }
  
  // Now compute the bounding box
  float xmin = FLT_MAX, ymin = FLT_MAX, xmax = FLT_MIN, ymax = FLT_MIN;
  
  for(int i = 0; i < points_count; i++)
  {
    const float x = points[i * 2];
    const float y = points[i * 2 + 1];
    
    xmin = fminf(xmin, x);
    xmax = fmaxf(xmax, x);
    ymin = fminf(ymin, y);
    ymax = fmaxf(ymax, y);
  }
  
  for(int i = 0; i < border_count; i++)
  {
    const float x = border[i * 2];
    const float y = border[i * 2 + 1];
    
    xmin = fminf(xmin, x);
    xmax = fmaxf(xmax, x);
    ymin = fminf(ymin, y);
    ymax = fmaxf(ymax, y);
  }
  
  *posx = xmin - 2;
  *posy = ymin - 2;
  *width = (xmax - xmin) + 4;
  *height = (ymax - ymin) + 4;
  
  dt_free_align(points);
  dt_free_align(border);
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
  double start = dt_get_debug_wtime();
  
  // We get the area
  if(!_object_get_area(module, piece, form, width, height, posx, posy))
  {
    return 0;
  }
  
  if(*width <= 0 || *height <= 0) return 0;
  
  // We get the points
  float *points = NULL, *border = NULL;
  int points_count, border_count;
  if(!_object_get_points_border(module->dev, form, &points, &points_count, &border, &border_count, FALSE, module))
  {
    return 0;
  }
  
  // Allocate the buffer
  *buffer = dt_alloc_align_float((size_t)*width * *height);
  if(*buffer == NULL)
  {
    dt_free_align(points);
    dt_free_align(border);
    return 0;
  }
  memset(*buffer, 0, sizeof(float) * (*width) * (*height));
  
  //TODO: update mask calculation
  // For now, just create small circles around each point
  const float radius = 20.0f; // Radius in pixels
  
  // Get the label of each point to apply positive or negative effect
  GList *pt_list = form->points;
  
  for(int i = 0; i < points_count; i++)
  {
    // Get point coordinates relative to the buffer
    const float px = points[i * 2] - *posx;
    const float py = points[i * 2 + 1] - *posy;
    
    // Get the current point from the form to check its label
    dt_masks_point_object_t *pt = g_list_nth_data(pt_list, i);
    if(!pt) continue;
    
    // Draw a circle around this point
    for(int y = MAX(0, py - radius); y < MIN(*height, py + radius); y++)
    {
      for(int x = MAX(0, px - radius); x < MIN(*width, px + radius); x++)
      {
        // Calculate distance to point center
        const float dist = sqrtf((x - px) * (x - px) + (y - py) * (y - py));
        
        // If inside radius, set opacity (positive or negative based on label)
        if(dist <= radius)
        {
          // Falloff from center to edge
          float opacity = 1.0f - (dist / radius);
          
          // Apply negative effect if label is 0
          if(pt->label == 0)
            opacity = -opacity;
            
          const int index = y * (*width) + x;
          
          // For positive points, we add opacity
          // For negative points, we subtract opacity
          if(opacity > 0)
            (*buffer)[index] = fmaxf((*buffer)[index], opacity);
          else
            (*buffer)[index] = fminf((*buffer)[index], 0.0f); // Negative points remove mask
        }
      }
    }
  }
  
  dt_free_align(points);
  dt_free_align(border);
  
  dt_print(DT_DEBUG_MASKS | DT_DEBUG_PERF,
           "[masks %s] object fill buffer took %0.04f sec", form->name,
           dt_get_lap_time(&start));
  
  return 1;
}


static int _object_get_mask_roi(const dt_iop_module_t *const restrict module,
                                const dt_dev_pixelpipe_iop_t *const restrict piece,
                                dt_masks_form_t *const form,
                                const dt_iop_roi_t *const roi,
                                float *const restrict buffer)
{
  double start = dt_get_debug_wtime();
  
  // We get the points
  float *points = NULL, *border = NULL;
  int points_count, border_count;
  if(!_object_get_points_border(module->dev, form, &points, &points_count, &border, &border_count, FALSE, module))
  {
    return 0;
  }
  
  //TODO: update mask calculation
  // For now, just create small circles around each point
  const float radius = 20.0f; // Radius in pixels
  
  // Get the label of each point to apply positive or negative effect
  GList *pt_list = form->points;
  
  for(int i = 0; i < points_count; i++)
  {
    // Adjust point coordinates to roi
    const float px = points[i * 2] - roi->x;
    const float py = points[i * 2 + 1] - roi->y;
    
    // Get the current point from the form to check its label
    dt_masks_point_object_t *pt = g_list_nth_data(pt_list, i);
    if(!pt) continue;
    
    // Draw a circle around this point
    for(int y = MAX(0, py - radius); y < MIN(roi->height, py + radius); y++)
    {
      for(int x = MAX(0, px - radius); x < MIN(roi->width, px + radius); x++)
      {
        // Calculate distance to point center
        const float dist = sqrtf((x - px) * (x - px) + (y - py) * (y - py));
        
        // If inside radius, set opacity (positive or negative based on label)
        if(dist <= radius)
        {
          // Falloff from center to edge
          float opacity = 1.0f - (dist / radius);
          
          // Apply negative effect if label is 0
          if(pt->label == 0)
            opacity = -opacity;
            
          const int index = y * roi->width + x;
          
          // For positive points, we add opacity
          // For negative points, we subtract opacity
          if(opacity > 0)
            buffer[index] = fmaxf(buffer[index], opacity);
          else
            buffer[index] = fminf(buffer[index], 0.0f); // Negative points remove mask
        }
      }
    }
  }
  
  dt_free_align(points);
  dt_free_align(border);
  
  dt_print(DT_DEBUG_MASKS | DT_DEBUG_PERF,
           "[masks %s] object roi fill took %0.04f sec", form->name,
           dt_get_lap_time(&start));
  
  return 1;
}

static GSList *_object_setup_mouse_actions(const struct dt_masks_form_t *const form)
{
  GSList *lm = NULL;
  lm = dt_mouse_action_create_simple(lm, DT_MOUSE_ACTION_LEFT, 0,
                                     _("[OBJECT] add a positive point"));
  lm = dt_mouse_action_create_simple(lm, DT_MOUSE_ACTION_LEFT, GDK_SHIFT_MASK,
                                     _("[OBJECT] add a negative point"));
  lm = dt_mouse_action_create_simple(lm, DT_MOUSE_ACTION_RIGHT, 0,
                                     _("[OBJECT] remove all points"));
  lm = dt_mouse_action_create_simple(lm, DT_MOUSE_ACTION_SCROLL, GDK_CONTROL_MASK,
                                     _("[OBJECT] change opacity"));
  return lm;
}

static void _object_sanitize_config(dt_masks_type_t type)
{
  // nothing to do (yet?)
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
    g_strlcat(msgbuf, _("<b>add point</b>: left-click, <b>add negative point</b>: shift+left-click\n"
                       "<b>remove all points</b>: right-click"), msgbuf_len);
  else if(gui->point_selected >= 0)
    g_strlcat(msgbuf, _("<b>remove point</b>: right-click"), msgbuf_len);
  else if(gui->form_selected)
    g_snprintf(msgbuf, msgbuf_len, _("<b>remove all points</b>: right-click\n"
                                     "<b>opacity</b>: ctrl+scroll (%d%%)"), opacity);
}

static void _object_duplicate_points(dt_develop_t *dev,
                                     dt_masks_form_t *const base,
                                     dt_masks_form_t *const dest)
{
  (void)dev; // unused arg, keep compiler from complaining
  for(const GList *pts = base->points; pts; pts = g_list_next(pts))
  {
    dt_masks_point_object_t *pt = pts->data;
    dt_masks_point_object_t *npt = malloc(sizeof(dt_masks_point_object_t));
    memcpy(npt, pt, sizeof(dt_masks_point_object_t));
    dest->points = g_list_append(dest->points, npt);
  }
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
  // Object mask does not support changing size/feather
  // Opacity is handled by the caller
  
  // Placeholder: If future properties are added for object masks, they should be handled here
}

static void _object_initial_source_pos(const float iwd,
                                       const float iht,
                                       float *x,
                                       float *y)
{
  // Object masks don't support clone functionality, but we still need to implement this
  // function as part of the API. Setting default values that won't be used.
  *x = 0.0f;
  *y = 0.0f;
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
