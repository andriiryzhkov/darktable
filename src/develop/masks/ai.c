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

#ifdef HAVE_AI
#include "common/ai/mask_ai.h"
#include "common/ai_models.h"
#endif
#include "common/dtdata.h"
#include "control/control.h"
#include "control/jobs.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/masks.h"
#include "dtgtk/paint.h"
#include "gui/gtk.h"
#include "views/view.h"

#include <math.h>
#include <string.h>

// an AI mask is made in one click, from a model that needs no prompts, and
// is stored as pixels in the image's .dtdata sidecar like a committed
// object (pixel_mask.c). it cannot be edited; a lost one is made again by
// clicking its icon

// --- making a mask: only where AI is compiled in ---

#ifdef HAVE_AI

static gboolean _task_usable(const char *task)
{
  GList *tasks = dt_ai_models_get_mask_ai_tasks();
  const gboolean usable = g_list_find_custom(tasks, task, (GCompareFunc)g_strcmp0) != NULL;
  g_list_free_full(tasks, g_free);
  return usable;
}

const char *dt_masks_ai_unavailable_reason(void)
{
  if(!dt_ai_registry_is_enabled())
    return _("AI is disabled, see preferences > AI");
  // the mask is pixels, and the sidecar is the only place they can go
  if(!dt_dtdata_enabled())
    return _("AI masks need XMP files: see preferences > storage > create XMP files");
  GList *tasks = dt_ai_models_get_mask_ai_tasks();
  const gboolean none = tasks == NULL;
  g_list_free_full(tasks, g_free);
  if(none)
    return _("no AI mask model is installed, see preferences > AI");
  return NULL;
}

typedef struct _ai_request_t
{
  // set at the click
  dt_imgid_t imgid;
  char task[DT_MASKS_AI_TASK_LEN];
  // the module asked for, by operation and instance rather than address:
  // it may be deleted, and another made, before the mask is done. an empty
  // op for none
  dt_dev_operation_t op;
  int multi_priority;
  dt_mask_id_t formid;       // the form to regenerate, NO_MASKID for a new one
  // set when its job is queued
  int32_t history_end;
  dt_hash_t distort_hash;
  char *model_id;
  char *producer;            // model id and version, kept in the reference
  // set by the job
  gboolean ran;
  float *mask;               // over the whole view, dt_free_align'd
  int mask_w, mask_h;
  dt_masks_pixel_grid_t grid; // the render the mask spans
} _ai_request_t;

// one job at a time, since each loads a model of its own, and the rest wait
// here rather than in a worker, where they would hold up exports. gui
// thread only
static GList *_waiting = NULL;
static _ai_request_t *_running = NULL;

static void _request_free(_ai_request_t *r)
{
  g_free(r->model_id);
  g_free(r->producer);
  dt_free_align(r->mask);
  g_free(r);
}

// canceled only when darktable quits
static gboolean _job_cancelled(dt_job_t *job)
{
  return dt_control_job_get_state(job) == DT_JOB_STATE_CANCELLED;
}

// the running request's toast, kept up for as long as it runs the way the
// object tool keeps "analyzing image" up: a repeated message only restarts
// the toast's timeout. gui thread only
static guint _toast_id = 0;
static char *_toast_message = NULL;

static gboolean _toast_repeat(gpointer data)
{
  if(dt_view_get_current() == DT_VIEW_DARKROOM)
    dt_control_log("%s", _toast_message);
  return G_SOURCE_CONTINUE;
}

static void _toast_start(const _ai_request_t *r)
{
  // TRANSLATORS: %s is the AI mask type, e.g. "ai subject"
  _toast_message = g_strdup_printf(_("%s: generating the mask..."),
                                   dt_ai_task_label(r->task));
  dt_control_log("%s", _toast_message);
  _toast_id = g_timeout_add(1000, _toast_repeat, NULL);
}

static void _toast_stop(void)
{
  if(!_toast_id) return;
  g_source_remove(_toast_id);
  _toast_id = 0;
  g_free(_toast_message);
  _toast_message = NULL;
  // elsewhere the toast is not ours to clear: it has not been repeated there
  if(dt_view_get_current() == DT_VIEW_DARKROOM) dt_control_log_ack_all();
}

// a worker thread: render the view at the model's size, then run the
// model, which loads its session only once the render's pipe is gone
static int32_t _job_run(dt_job_t *job)
{
  _ai_request_t *r = dt_control_job_get_params(job);
  r->ran = TRUE;

  dt_mask_ai_t *model = _job_cancelled(job) ? NULL : dt_mask_ai_open(r->model_id);
  if(!model) return 1;
  const int size = dt_mask_ai_input_size(model);

  int width = 0, height = 0;
  dt_masks_pixel_render_t *render = _job_cancelled(job)
    ? NULL : dt_masks_pixel_render_init(r->imgid, r->history_end, &width, &height);
  uint8_t *rgb = NULL;
  int rw = 0, rh = 0;
  if(render)
  {
    // the short side at the model's input size, so the stretch to its
    // square keeps all the detail it can take on both axes, and the long
    // side at least DT_MASK_AI_GUIDE_SIZE when the render also guides the
    // edges. never above 1: a larger render adds no detail. the epsilon
    // keeps a side scaled to exactly a size from flooring below it
    double scale = (double)size / MAX(1, MIN(width, height));
    if(dt_mask_ai_refines(model))
      scale = fmax(scale, (double)DT_MASK_AI_GUIDE_SIZE / MAX(1, MAX(width, height)));
    scale = fmin(1.0, scale);
    rw = MAX(1, (int)(scale * width + 1e-4));
    rh = MAX(1, (int)(scale * height + 1e-4));
    dt_print(DT_DEBUG_AI, "[AI mask] rendering %dx%d (scale=%.3f) for %s",
             rw, rh, scale, r->model_id);
    r->grid = (dt_masks_pixel_grid_t){ width, height, rw, rh, scale };
    rgb = _job_cancelled(job) ? NULL : dt_masks_pixel_render(render, rw, rh, scale);
    dt_masks_pixel_render_cleanup(render);
  }
  float *mask = rgb && !_job_cancelled(job)
    ? dt_mask_ai_run(model, rgb, rw, rh, &r->mask_w, &r->mask_h) : NULL;
  g_free(rgb);
  dt_mask_ai_close(model);
  r->mask = mask;
  return mask ? 0 : 1;
}

// the module the mask was asked for, if it is still there. when it is
// gone a new mask stays, as a standalone shape
static dt_iop_module_t *_live_module(const _ai_request_t *r)
{
  return r->op[0]
    ? dt_iop_get_module_by_op_priority(darktable.develop->iop, r->op, r->multi_priority)
    : NULL;
}

static gboolean _store(const _ai_request_t *r, dt_dtdata_ref_t *ref)
{
  int tw = 0, th = 0;
  return dt_masks_pixel_store_size(r->mask_w, r->mask_h, DT_MASKS_PIXEL_MAX_STORED,
                                   &tw, &th)
    && dt_masks_pixel_store(r->mask, r->mask_w, r->mask_h, &r->grid, tw, th,
                            NULL, r->producer, ref);
}

// the point, allocated before the write: failing after it would leave an
// unreferenced entry. NULL when either fails
static dt_masks_point_ai_t *_new_point(const _ai_request_t *r)
{
  dt_masks_point_ai_t *pt = calloc(1, sizeof(dt_masks_point_ai_t));
  if(!pt || !_store(r, &pt->ref))
  {
    free(pt);
    // TRANSLATORS: %s is the AI mask type, e.g. "ai subject"
    dt_control_log(_("%s: the mask could not be stored in the sidecar"),
                   dt_ai_task_label(r->task));
    return NULL;
  }
  g_strlcpy(pt->task, r->task, sizeof(pt->task));
  return pt;
}

// filed as a committed object is, and selected so the mask manager lists
// it at once (object.c's _leave_creation)
static void _add(const _ai_request_t *r)
{
  dt_develop_t *dev = darktable.develop;
  dt_iop_module_t *module = _live_module(r);

  dt_masks_form_t *form = dt_masks_create(DT_MASKS_AI);
  dt_masks_point_ai_t *pt = form ? _new_point(r) : NULL;
  if(!pt)
  {
    dt_masks_free_form(form);
    return;
  }
  form->points = g_list_append(NULL, pt);
  // a pipe copies dev->forms and the groups under history_mutex, and this
  // runs from an idle, not from a shape's event callback, which holds it
  dt_pthread_mutex_lock(&dev->history_mutex);
  dt_masks_gui_form_save_creation(dev, module, form, NULL);
  dt_pthread_mutex_unlock(&dev->history_mutex);
  if(module) dt_masks_iop_update(module);

  // a shape being drawn meanwhile keeps the canvas, and a module that lost
  // the focus since the click does not take the canvas back
  if(!(dev->form_gui && dev->form_gui->creation))
  {
    if(module && module == dev->gui_module)
      dt_masks_set_edit_mode(module, DT_MASKS_EDIT_FULL);
    dt_dev_masks_selection_change(dev, module, form->formid);
  }
  dt_control_queue_redraw_center();
}

static void _replace(const _ai_request_t *r)
{
  dt_develop_t *dev = darktable.develop;
  // by id: a history change since the click replaces forms with copies, and
  // may have removed this one
  dt_masks_form_t *live = dt_masks_get_from_id(dev, r->formid);
  if(!live || live->functions != &dt_masks_functions_ai)
  {
    // TRANSLATORS: %s is the AI mask type, e.g. "ai subject"
    dt_control_log(_("%s: the mask was removed before it was regenerated"),
                   dt_ai_task_label(r->task));
    return;
  }
  dt_masks_point_ai_t *pt = _new_point(r);
  if(!pt) return;
  // as in _add: a pipe may be copying these points
  dt_pthread_mutex_lock(&dev->history_mutex);
  g_list_free_full(live->points, free);
  live->points = g_list_append(NULL, pt);
  dt_pthread_mutex_unlock(&dev->history_mutex);
  dt_dev_add_masks_history_item(dev, _live_module(r), TRUE);
  dt_control_queue_redraw_center();
}

static gboolean _job_finish(gpointer data);

// the job's params destructor, on whichever thread disposes of it: always
// called, so the gui thread hears of every job, canceled ones included
static void _job_dispose(void *data)
{
  g_idle_add(_job_finish, data);
}

// TRUE once darktable is closing: dt_control_quit raises quitting before
// it waits for the jobs, while dt_control_running still holds
static gboolean _closing(void)
{
  return !dt_control_running() || dt_atomic_get_int(&darktable.control->quitting);
}

// the darkroom is on imgid and not leaving it: dt_dev_change_image() writes
// the history and requests the next image, and image_storage names the old
// one until that loads, freeing the forms unsaved (darkroom.c)
static gboolean _on_image(const dt_imgid_t imgid)
{
  const dt_develop_t *dev = darktable.develop;
  return dev->image_storage.id == imgid && dev->requested_id == imgid;
}

// the focused module when it distorts. crop and perspective show the image
// uncropped while they have focus (crop.c's and ashift.c's commit_params),
// while the job renders the history's view: the preview pipe that maps the
// mask into the input image would then place it wrongly
static dt_iop_module_t *_distort_focused(void)
{
  dt_iop_module_t *m = darktable.develop->gui_module;
  return m && m->enabled && (m->operation_tags() & IOP_TAG_DISTORT) ? m : NULL;
}

// at the model's own decision boundary: below it everywhere, the type is
// not in the image, and the faint rest would apply the module weakly
static gboolean _found(const _ai_request_t *r)
{
  const size_t n = (size_t)r->mask_w * r->mask_h;
  for(size_t i = 0; i < n; i++)
    if(r->mask[i] > 0.5f) return TRUE;
  return FALSE;
}

// queue the request's job, with the state of the view taken now, after the
// history is written: the job renders from the database. FALSE, the
// request freed, when it no longer applies
static gboolean _launch(_ai_request_t *r)
{
  dt_develop_t *dev = darktable.develop;
  // the darkroom left meanwhile is no news to the user, nor is quitting
  if(_closing() || dt_view_get_current() != DT_VIEW_DARKROOM)
  {
    _request_free(r);
    return FALSE;
  }
  if(!_on_image(r->imgid))
  {
    // TRANSLATORS: %s is the AI mask type, e.g. "ai subject"
    dt_control_log(_("%s: the mask is discarded, the image changed"),
                   dt_ai_task_label(r->task));
    _request_free(r);
    return FALSE;
  }
  const dt_iop_module_t *focused = _distort_focused();
  if(focused)
  {
    // TRANSLATORS: the first %s is the AI mask type, e.g. "ai subject", the
    // second a module's name, e.g. "crop"
    dt_control_log(_("%s: cannot be made while %s has focus"),
                   dt_ai_task_label(r->task), focused->name());
    _request_free(r);
    return FALSE;
  }
  char *model_id = _task_usable(r->task) ? dt_ai_models_get_active_for_task(r->task) : NULL;
  if(!model_id)
  {
    // TRANSLATORS: %s is the AI mask type, e.g. "ai subject"
    dt_control_log(_("%s: no model is installed, see preferences > AI"),
                   dt_ai_task_label(r->task));
    _request_free(r);
    return FALSE;
  }

  dt_dev_write_history(dev);
  r->history_end = dev->history_end;
  r->distort_hash = dt_masks_pixel_distort_hash(dev);
  r->model_id = model_id;
  r->producer = g_strdup_printf("%s %s", model_id, dt_ai_model_get_version(model_id));

  dt_job_t *job = dt_control_job_create(_job_run, "AI mask");
  if(!job)
  {
    _request_free(r);
    return FALSE;
  }
  dt_control_job_set_params(job, r, _job_dispose);
  dt_control_add_job(DT_JOB_QUEUE_USER_BG, job);
  return TRUE;
}

// drop everything waiting once darktable quits, else start the next request
// when none runs
static void _pump(void)
{
  if(_closing())
  {
    g_list_free_full(_waiting, (GDestroyNotify)_request_free);
    _waiting = NULL;
  }
  while(!_running && _waiting)
  {
    _ai_request_t *r = _waiting->data;
    _waiting = g_list_delete_link(_waiting, _waiting);
    if(_launch(r))
    {
      _running = r;
      _toast_start(r);
    }
  }
}

static gboolean _job_finish(gpointer data)
{
  _ai_request_t *r = data;
  if(_running == r)
  {
    _running = NULL;
    _toast_stop();
  }

  dt_develop_t *dev = darktable.develop;
  // a job canceled, never run, or done after the darkroom was left or
  // while quitting has nothing to say
  if(r->ran && !_closing()
     && dt_view_get_current() == DT_VIEW_DARKROOM)
  {
    const dt_iop_module_t *focused = NULL;
    if(!r->mask)
      // TRANSLATORS: %s is the AI mask type, e.g. "ai subject"
      dt_control_log(_("%s: the mask could not be generated"), dt_ai_task_label(r->task));
    else if(!_found(r))
      // TRANSLATORS: %s is the AI mask type, e.g. "ai subject"
      dt_control_log(_("%s: nothing was found"), dt_ai_task_label(r->task));
    else if(!_on_image(r->imgid)
            || dt_masks_pixel_distort_hash(dev) != r->distort_hash)
      // TRANSLATORS: %s is the AI mask type, e.g. "ai subject"
      dt_control_log(_("%s: the mask is discarded, the image changed while it"
                       " was generated"), dt_ai_task_label(r->task));
    else if((focused = _distort_focused()))
      // TRANSLATORS: the first %s is the AI mask type, e.g. "ai subject", the
      // second a module's name, e.g. "crop"
      dt_control_log(_("%s: the mask is discarded, %s got focus while it was"
                       " generated"), dt_ai_task_label(r->task), focused->name());
    else if(dt_is_valid_maskid(r->formid))
      _replace(r);
    else
      _add(r);
  }

  _request_free(r);
  _pump();
  return G_SOURCE_REMOVE;
}

// by image too: a history paste copies form ids to other images
static gboolean _pending(const dt_imgid_t imgid, const dt_mask_id_t formid)
{
  if(_running && _running->imgid == imgid && _running->formid == formid)
    return TRUE;
  for(const GList *l = _waiting; l; l = g_list_next(l))
  {
    const _ai_request_t *r = l->data;
    if(r->imgid == imgid && r->formid == formid) return TRUE;
  }
  return FALSE;
}

static void _request(dt_iop_module_t *module, const char *task, const dt_mask_id_t formid)
{
  dt_develop_t *dev = darktable.develop;
  const char *reason = dt_masks_ai_unavailable_reason();
  if(reason)
  {
    dt_control_log("%s", reason);
    return;
  }
  if(!dt_is_valid_imgid(dev->image_storage.id)) return;
  if(!_task_usable(task))
  {
    // TRANSLATORS: %s is the AI mask type, e.g. "ai subject"
    dt_control_log(_("%s: no model is installed, see preferences > AI"),
                   dt_ai_task_label(task));
    return;
  }

  _ai_request_t *r = g_new0(_ai_request_t, 1);
  r->imgid = dev->image_storage.id;
  g_strlcpy(r->task, task, sizeof(r->task));
  if(module)
  {
    g_strlcpy(r->op, module->op, sizeof(r->op));
    r->multi_priority = module->multi_priority;
  }
  r->formid = formid;

  _waiting = g_list_append(_waiting, r);
  _pump();
}

typedef enum _ai_state_t
{
  AI_MASK_THERE,
  AI_MASK_SHORT,   // there, but memory ran short reading it
  AI_MASK_MISSING,
} _ai_state_t;

static _ai_state_t _state(const dt_masks_form_t *form)
{
  const dt_dtdata_ref_t *ref = dt_masks_pixel_ref(form);
  if(!ref) return AI_MASK_THERE;
  gboolean transient = FALSE;
  if(dt_masks_pixel_readable(darktable.develop->image_storage.id, ref, &transient))
    return AI_MASK_THERE;
  return transient ? AI_MASK_SHORT : AI_MASK_MISSING;
}

static void _regenerate(dt_iop_module_t *module, const dt_masks_form_t *form)
{
  if(_pending(darktable.develop->image_storage.id, form->formid))
  {
    dt_control_log(_("the mask is already being regenerated"));
    return;
  }
  const dt_masks_point_ai_t *pt = form->points->data;
  // module is the focused one: the history item goes under it, enabling
  // it, only if the mask is one of its masks (as object.c's _start_edit)
  _request(dt_masks_is_in_module(form->formid, module) ? module : NULL,
           pt->task, form->formid);
}

static void _menu_activate(GSimpleAction *action, GVariant *task, gpointer module)
{
  _request(module, g_variant_get_string(task, NULL), NO_MASKID);
}

void dt_masks_ai_popup_menu(GtkWidget *button, dt_iop_module_t *module)
{
  const char *reason = dt_masks_ai_unavailable_reason();
  if(reason)
  {
    dt_control_log("%s", reason);
    return;
  }
  // a menu even for a single type, which names what the click makes
  GList *tasks = dt_ai_models_get_mask_ai_tasks();
  if(!tasks) return;

  // a shortcut can activate a button that is not shown: the menu then
  // points at the pointer over the center view
  const gboolean shown = button && gtk_widget_get_mapped(button);
  GtkWidget *parent = shown ? button : dt_ui_center(darktable.gui->ui);

  // inserted again for each menu: the mask manager's button serves
  // whichever module is selected
  const GActionEntry entries[] = { { "add", _menu_activate, "s", NULL, NULL } };
  GSimpleActionGroup *group = g_simple_action_group_new();
  g_action_map_add_action_entries(G_ACTION_MAP(group), entries, G_N_ELEMENTS(entries), module);
  gtk_widget_insert_action_group(parent, "aimask", G_ACTION_GROUP(group));
  g_object_unref(group);

  GMenu *menu = g_menu_new();
  for(GList *l = tasks; l; l = g_list_next(l))
  {
    GMenuItem *item = g_menu_item_new(dt_ai_mask_ai_type_label(l->data), NULL);
    g_menu_item_set_action_and_target_value(item, "aimask.add", g_variant_new_string(l->data));
    g_menu_append_item(menu, item);
    g_object_unref(item);
  }
  g_list_free_full(tasks, g_free);

  GtkWidget *popover = dt_gui_popover_menu_from_model(parent, menu);
  g_object_unref(menu);
  if(!shown)
  {
    int x = 0, y = 0;
    GdkDisplay *display = gtk_widget_get_display(parent);
    gdk_window_get_device_position(gtk_widget_get_window(parent),
                                   gdk_seat_get_pointer(gdk_display_get_default_seat(display)),
                                   &x, &y, NULL);
    // kept inside the view, where the pointer need not be
    const GdkRectangle at = { CLAMP(x, 0, gtk_widget_get_allocated_width(parent) - 1),
                              CLAMP(y, 0, gtk_widget_get_allocated_height(parent) - 1),
                              1, 1 };
    gtk_popover_set_pointing_to(GTK_POPOVER(popover), &at);
  }
  gtk_popover_popup(GTK_POPOVER(popover));
}

void dt_masks_ai_update_button(GtkWidget *button)
{
  if(!button) return;
  const char *reason = dt_masks_ai_unavailable_reason();
  gtk_widget_set_sensitive(button, reason == NULL);
  gtk_widget_set_tooltip_text(button, reason ? reason : _("add AI mask"));
}

// numbered per type: nb counts every AI mask and grows until the name is
// unique, so the other types' count, fixed meanwhile, is taken off it
static void _ai_set_form_name(dt_masks_form_t *const form, const size_t nb)
{
  const dt_masks_point_ai_t *pt = form->points ? form->points->data : NULL;
  const char *task = pt ? pt->task : "";
  int others = 0;
  for(const GList *l = darktable.develop->forms; l; l = g_list_next(l))
  {
    const dt_masks_form_t *f = l->data;
    if(f != form && f->type == form->type && f->points
       && strcmp(((const dt_masks_point_ai_t *)f->points->data)->task, task))
      others++;
  }
  // TRANSLATORS: the name of an AI mask: %s is its type, e.g. "ai subject",
  // and %d its number among the masks of that type
  snprintf(form->name, sizeof(form->name), _("%s #%d"),
           dt_ai_task_label(task), (int)nb - others);
}

#endif // HAVE_AI

// --- the committed mask: in every build, so one renders without AI ---

// an icon the mask gives no place, a lost mask's, goes to the middle of the
// frame: it is the way to regenerate the mask
static gboolean _ai_fallback_anchor(const dt_masks_form_t *form, float anchor[2])
{
  anchor[0] = anchor[1] = 0.5f;
  return TRUE;
}

static const dt_masks_pixel_type_t _ai_pixel_type = {
  .icon = dtgtk_cairo_paint_masks_ai,
  .fallback_anchor = _ai_fallback_anchor,
  .edit_action = N_("[AI MASK] regenerate a missing mask"),
  .opacity_action = N_("[AI MASK] change opacity"),
};

static GSList *_ai_setup_mouse_actions(const struct dt_masks_form_t *const form)
{
  return dt_masks_pixel_setup_mouse_actions(&_ai_pixel_type);
}

static void _ai_set_hint_message(const dt_masks_form_gui_t *const gui,
                                 const dt_masks_form_t *const form,
                                 const int opacity,
                                 char *const restrict msgbuf,
                                 const size_t msgbuf_len)
{
#ifdef HAVE_AI
  if(_state(form) == AI_MASK_MISSING)
  {
    g_snprintf(msgbuf, msgbuf_len,
               _("<b>regenerate</b>: click, <b>opacity</b>: ctrl+scroll (%d%%)"),
               opacity);
    return;
  }
#endif
  // there is nothing to edit
  g_snprintf(msgbuf, msgbuf_len, _("<b>opacity</b>: ctrl+scroll (%d%%)"), opacity);
}

// a click on the icon regenerates a lost mask; one that is there is final.
// one only short of memory says nothing: its struck icon already tells
static int _ai_button_pressed(dt_iop_module_t *module,
                              const float pzx,
                              const float pzy,
                              const double pressure,
                              const int which,
                              const int type,
                              const uint32_t state,
                              dt_masks_form_t *form,
                              const dt_imgid_t parentid,
                              dt_masks_form_gui_t *gui,
                              const int index)
{
#ifdef HAVE_AI
  if(gui && which == 1 && type == GDK_BUTTON_PRESS && gui->form_selected
     && gui->point_selected == 0 && dt_masks_pixel_ref(form))
  {
    const _ai_state_t st = _state(form);
    if(st == AI_MASK_MISSING)
      _regenerate(module, form);
    else if(st == AI_MASK_THERE)
      dt_control_log(_("AI masks cannot be edited"));
    return 1;
  }
#endif
  return 0;
}

static int _ai_button_released(dt_iop_module_t *module,
                               const float pzx,
                               const float pzy,
                               const int which,
                               const uint32_t state,
                               dt_masks_form_t *form,
                               const dt_imgid_t parentid,
                               dt_masks_form_gui_t *gui,
                               const int index)
{
  return dt_masks_pixel_button_released(module, which, form, parentid, gui);
}

static void _ai_post_expose(cairo_t *cr,
                            const float zoom_scale,
                            dt_masks_form_gui_t *gui,
                            const int index,
                            const int num_points)
{
  dt_masks_pixel_post_expose(&_ai_pixel_type, cr, zoom_scale, gui, index, num_points);
}

static int _ai_get_points_border(dt_develop_t *dev,
                                 dt_masks_form_t *form,
                                 float **points,
                                 int *points_count,
                                 float **border,
                                 int *border_count,
                                 const int source,
                                 const dt_iop_module_t *const module)
{
  return dt_masks_pixel_get_points_border(&_ai_pixel_type, dev, form,
                                          points, points_count, border,
                                          border_count, source, module);
}

const dt_masks_functions_t dt_masks_functions_ai = {
  .point_struct_size = sizeof(struct dt_masks_point_ai_t),
  .sanitize_config = NULL,
  .setup_mouse_actions = _ai_setup_mouse_actions,
  .set_hint_message = _ai_set_hint_message,
  .modify_property = NULL,
#ifdef HAVE_AI
  .set_form_name = _ai_set_form_name,
#else
  // nothing names a form without AI: none is made
  .set_form_name = NULL,
#endif
  .duplicate_points = dt_masks_pixel_duplicate_points,
  .initial_source_pos = NULL,
  .get_distance = dt_masks_pixel_get_distance,
  .get_points = NULL,
  .get_points_border = _ai_get_points_border,
  // never in a clone group, the only user of these
  .get_mask = NULL,
  .get_mask_roi = dt_masks_pixel_get_mask_roi,
  .get_area = NULL,
  .get_source_area = NULL,
  .mouse_moved = dt_masks_pixel_mouse_moved,
  .mouse_scrolled = dt_masks_pixel_mouse_scrolled,
  .button_pressed = _ai_button_pressed,
  .button_released = _ai_button_released,
  .post_expose = _ai_post_expose
};

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
