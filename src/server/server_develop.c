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

#include "server/server.h"
#include "common/history.h"
#include "common/image.h"
#include "common/image_cache.h"
#include "common/iop_order.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/pixelpipe.h"

#include <string.h>

// Mirror of dt_iop_exposure_params_t from iop/exposure.c
// Must match the struct layout exactly (introspection version 7).
typedef enum _server_exposure_mode_t
{
  _EXPOSURE_MODE_MANUAL = 0,
  _EXPOSURE_MODE_DEFLICKER = 1
} _server_exposure_mode_t;

typedef struct _server_exposure_params_t
{
  _server_exposure_mode_t mode;
  float black;
  float exposure;
  float deflicker_percentile;
  float deflicker_target_level;
  gboolean compensate_exposure_bias;
  gboolean compensate_hilite_pres;
} _server_exposure_params_t;

static dt_server_session_t *_find_free_session_slot(dt_server_t *server)
{
  if(server->session_count >= DT_SERVER_MAX_SESSIONS)
    return NULL;

  dt_server_session_t *session = g_new0(dt_server_session_t, 1);
  snprintf(session->session_id, sizeof(session->session_id),
           "dev-%03d", server->next_session_id++);
  server->sessions[server->session_count] = session;
  server->session_count++;
  return session;
}

char *dt_server_develop_open(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "imgid"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing imgid parameter");

  const dt_imgid_t imgid = (dt_imgid_t)json_object_get_int_member(req->params, "imgid");

  // Default preview size
  int preview_width = 1920;
  int preview_height = 1280;
  if(json_object_has_member(req->params, "width"))
    preview_width = (int)json_object_get_int_member(req->params, "width");
  if(json_object_has_member(req->params, "height"))
    preview_height = (int)json_object_get_int_member(req->params, "height");

  // Clamp preview dimensions
  if(preview_width < 320) preview_width = 320;
  if(preview_width > 4096) preview_width = 4096;
  if(preview_height < 240) preview_height = 240;
  if(preview_height > 4096) preview_height = 4096;

  // Single-client model: close all existing sessions before opening a new one.
  // This prevents orphaned sessions from accumulating when the client rapidly
  // switches images (the previous close may not have arrived yet).
  for(int i = server->session_count - 1; i >= 0; i--)
  {
    if(server->sessions[i])
    {
      dt_server_session_t *old = server->sessions[i];
      fprintf(stderr, "[server] develop.open: auto-closing session %s\n", old->session_id);
      dt_shm_destroy(&old->shm_buffers[0]);
      dt_shm_destroy(&old->shm_buffers[1]);
      pthread_mutex_destroy(&old->pipeline_mutex);
      dt_dev_cleanup(&old->dev);
      g_free(old);
      server->sessions[i] = NULL;
    }
  }
  server->session_count = 0;

  // Allocate session slot
  dt_server_session_t *session = _find_free_session_slot(server);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_BUSY,
                                 "Maximum number of develop sessions reached");

  session->imgid = imgid;
  session->preview_width = preview_width;
  session->preview_height = preview_height;
  session->frame_sequence = 0;
  session->dirty = TRUE;
  session->pipeline_seq = 0;
  session->pipeline_busy = FALSE;
  pthread_mutex_init(&session->pipeline_mutex, NULL);

  // Initialize develop context following dt_dev_image() pattern (develop.c:3847)
  dt_dev_init(&session->dev, TRUE);
  session->dev.gui_attached = FALSE;

  // The full pipe is already DT_DEV_PIXELPIPE_FULL from dt_dev_pixelpipe_init().
  // Do NOT set DT_DEV_PIXELPIPE_IMAGE — that flag disables the intermediate
  // cache (pipe->nocache=TRUE), which prevents reuse of upstream module outputs
  // when only one module's params change.
  dt_dev_pixelpipe_t *pipe = session->dev.full.pipe;

  // Load image: instantiates modules, loads history from DB
  dt_dev_load_image(&session->dev, imgid);

  // Apply history to module structs (params + enabled state).
  // dt_dev_load_image() reads history items but does not sync them to modules.
  // Without this, module->enabled stays at default_enabled and module->params
  // stay at default_params, causing set_params to create disabled history items.
  dt_pthread_mutex_lock(&session->dev.history_mutex);
  dt_dev_pop_history_items_ext(&session->dev, session->dev.history_end);
  dt_pthread_mutex_unlock(&session->dev.history_mutex);

  // Validate the image was loaded successfully
  if(!dt_is_valid_imgid(session->dev.image_storage.id))
  {
    dt_dev_cleanup(&session->dev);
    server->sessions[server->session_count - 1] = NULL;
    server->session_count--;
    g_free(session);
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND,
                                 "Failed to load image into develop");
  }

  // Configure viewport for requested preview dimensions
  session->dev.full.zoom = DT_ZOOM_FIT;
  session->dev.full.width = preview_width;
  session->dev.full.height = preview_height;
  session->dev.full.ppd = 1.0;
  session->dev.full.color_assessment = FALSE;
  session->dev.full.dev = &session->dev;
  session->dev.full.pipe = pipe;

  // Create double-buffered SHM for preview frames
  char shm_name0[32], shm_name1[32];
  snprintf(shm_name0, sizeof(shm_name0), "/dt-prev-%s-0", session->session_id);
  snprintf(shm_name1, sizeof(shm_name1), "/dt-prev-%s-1", session->session_id);

  if(!dt_shm_create(&session->shm_buffers[0], shm_name0, preview_width, preview_height)
     || !dt_shm_create(&session->shm_buffers[1], shm_name1, preview_width, preview_height))
  {
    dt_shm_destroy(&session->shm_buffers[0]);
    dt_shm_destroy(&session->shm_buffers[1]);
    dt_dev_cleanup(&session->dev);
    // Remove session from slot
    server->sessions[server->session_count - 1] = NULL;
    server->session_count--;
    g_free(session);
    return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL,
                                 "Failed to create shared memory buffers");
  }

  session->front_buffer = 0;

  fprintf(stderr, "[server] develop.open: session=%s imgid=%d preview=%dx%d\n",
          session->session_id, imgid, preview_width, preview_height);

  // Build response
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "session_id");
  json_builder_add_string_value(b, session->session_id);

  json_builder_set_member_name(b, "imgid");
  json_builder_add_int_value(b, imgid);

  json_builder_set_member_name(b, "preview_width");
  json_builder_add_int_value(b, preview_width);

  json_builder_set_member_name(b, "preview_height");
  json_builder_add_int_value(b, preview_height);

  json_builder_set_member_name(b, "shm_names");
  json_builder_begin_array(b);
  json_builder_add_string_value(b, shm_name0);
  json_builder_add_string_value(b, shm_name1);
  json_builder_end_array(b);

  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_close(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing session_id parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");
  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  fprintf(stderr, "[server] develop.close: session=%s\n", session_id);

  // Destroy SHM buffers
  dt_shm_destroy(&session->shm_buffers[0]);
  dt_shm_destroy(&session->shm_buffers[1]);

  // Cleanup pipeline mutex
  pthread_mutex_destroy(&session->pipeline_mutex);

  // Cleanup develop context
  dt_dev_cleanup(&session->dev);

  // Remove from server's session list
  for(int i = 0; i < server->session_count; i++)
  {
    if(server->sessions[i] == session)
    {
      // Shift remaining sessions down
      for(int j = i; j < server->session_count - 1; j++)
        server->sessions[j] = server->sessions[j + 1];
      server->sessions[server->session_count - 1] = NULL;
      server->session_count--;
      break;
    }
  }
  g_free(session);

  // Build response
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "closed");
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_get_modules(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing session_id parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");
  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "modules");
  json_builder_begin_array(b);

  for(GList *modules = session->dev.iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;

    json_builder_begin_object(b);

    json_builder_set_member_name(b, "op");
    json_builder_add_string_value(b, mod->op);

    json_builder_set_member_name(b, "name");
    json_builder_add_string_value(b, mod->name());

    json_builder_set_member_name(b, "enabled");
    json_builder_add_boolean_value(b, mod->enabled);

    json_builder_set_member_name(b, "instance");
    json_builder_add_int_value(b, mod->multi_priority);

    json_builder_set_member_name(b, "iop_order");
    json_builder_add_double_value(b, mod->iop_order);

    json_builder_set_member_name(b, "params_size");
    json_builder_add_int_value(b, mod->params_size);

    json_builder_end_object(b);
  }

  json_builder_end_array(b);
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_get_history(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing session_id parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");
  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "history_end");
  json_builder_add_int_value(b, session->dev.history_end);

  json_builder_set_member_name(b, "items");
  json_builder_begin_array(b);

  // Add "original" as item -1 (same as GTK darktable)
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "num");
  json_builder_add_int_value(b, -1);
  json_builder_set_member_name(b, "op");
  json_builder_add_string_value(b, "");
  json_builder_set_member_name(b, "name");
  json_builder_add_string_value(b, "original");
  json_builder_set_member_name(b, "enabled");
  json_builder_add_boolean_value(b, FALSE);
  json_builder_set_member_name(b, "mandatory");
  json_builder_add_boolean_value(b, TRUE);
  json_builder_end_object(b);

  dt_pthread_mutex_lock(&session->dev.history_mutex);

  int num = 0;
  for(GList *hist = session->dev.history; hist; hist = g_list_next(hist))
  {
    dt_dev_history_item_t *item = hist->data;
    if(!item) continue;

    // Skip mask_manager entries
    if(!strcmp(item->op_name, "mask_manager")) continue;

    json_builder_begin_object(b);

    json_builder_set_member_name(b, "num");
    json_builder_add_int_value(b, num++);

    json_builder_set_member_name(b, "op");
    json_builder_add_string_value(b, item->op_name);

    json_builder_set_member_name(b, "name");
    if(item->module && item->module->name)
    {
      const char *display_name = item->module->name();
      // Skip _builtin_ prefix and empty/default multi_name values
      const char *mname = item->multi_name;
      if(g_str_has_prefix(mname, BUILTIN_PREFIX))
        mname += strlen(BUILTIN_PREFIX);
      if(mname[0] && strcmp(mname, "0") != 0)
      {
        char full_name[256];
        snprintf(full_name, sizeof(full_name), "%s \xe2\x80\xa2 %s",
                 display_name, mname);
        json_builder_add_string_value(b, full_name);
      }
      else
        json_builder_add_string_value(b, display_name);
    }
    else
      json_builder_add_string_value(b, item->op_name);

    json_builder_set_member_name(b, "enabled");
    json_builder_add_boolean_value(b, item->enabled);

    json_builder_set_member_name(b, "mandatory");
    json_builder_add_boolean_value(b, item->module && item->module->hide_enable_button);

    json_builder_end_object(b);
  }

  dt_pthread_mutex_unlock(&session->dev.history_mutex);

  json_builder_end_array(b);
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_get_params(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "op"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id or op parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  const char *op = json_object_get_string_member(req->params, "op");
  if(!session_id || !op)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id and op must be strings");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  // Find the module
  dt_iop_module_t *target = NULL;
  for(GList *modules = session->dev.iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;
    if(dt_iop_module_is(mod->so, op))
    {
      target = mod;
      break;
    }
  }

  if(!target)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Module not found");

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "op");
  json_builder_add_string_value(b, target->op);

  json_builder_set_member_name(b, "enabled");
  json_builder_add_boolean_value(b, target->enabled);

  // Module-specific parameter serialization
  // For now, we hardcode serializers for known modules.
  json_builder_set_member_name(b, "params");
  json_builder_begin_object(b);

  if(!strcmp(op, "exposure"))
  {
    const _server_exposure_params_t *p = (const _server_exposure_params_t *)target->params;
    json_builder_set_member_name(b, "mode");
    json_builder_add_int_value(b, (int)p->mode);
    json_builder_set_member_name(b, "exposure");
    json_builder_add_double_value(b, p->exposure);
    json_builder_set_member_name(b, "black");
    json_builder_add_double_value(b, p->black);
    json_builder_set_member_name(b, "compensate_exposure_bias");
    json_builder_add_boolean_value(b, p->compensate_exposure_bias);
    json_builder_set_member_name(b, "compensate_hilite_pres");
    json_builder_add_boolean_value(b, p->compensate_hilite_pres);
    json_builder_set_member_name(b, "deflicker_percentile");
    json_builder_add_double_value(b, p->deflicker_percentile);
    json_builder_set_member_name(b, "deflicker_target_level");
    json_builder_add_double_value(b, p->deflicker_target_level);

    // Computed EXIF bias values for dynamic checkbox labels
    float exposure_bias = 0.0f;
    if(session->dev.image_storage.exif_exposure_bias != DT_EXIF_TAG_UNINITIALIZED)
      exposure_bias = CLAMPF(session->dev.image_storage.exif_exposure_bias, -5.0f, 5.0f);
    json_builder_set_member_name(b, "exposure_bias_ev");
    json_builder_add_double_value(b, exposure_bias);

    float highlight_bias = 0.0f;
    if(session->dev.image_storage.exif_highlight_preservation > 0.0f
       && session->dev.image_storage.exif_highlight_preservation != DT_EXIF_TAG_UNINITIALIZED)
      highlight_bias = CLAMPF(session->dev.image_storage.exif_highlight_preservation, -1.0f, 4.0f);
    json_builder_set_member_name(b, "highlight_bias_ev");
    json_builder_add_double_value(b, highlight_bias);
  }
  else
  {
    // Generic: return params as base64 blob for unsupported modules
    gchar *b64 = g_base64_encode((const guchar *)target->params, target->params_size);
    json_builder_set_member_name(b, "_raw_base64");
    json_builder_add_string_value(b, b64);
    json_builder_set_member_name(b, "_raw_size");
    json_builder_add_int_value(b, target->params_size);
    g_free(b64);
  }

  json_builder_end_object(b);
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

// --- Async pipeline worker (event-driven preview) ---

typedef struct _pipeline_job_t
{
  dt_server_t *server;
  dt_server_session_t *session;
} _pipeline_job_t;

static void *_preview_pipeline_worker(void *arg)
{
  _pipeline_job_t *job = arg;
  dt_server_session_t *session = job->session;
  dt_server_t *server = job->server;
  g_free(job);

  while(TRUE)
  {
    // Snapshot the current seq before processing
    pthread_mutex_lock(&session->pipeline_mutex);
    const uint64_t start_seq = session->pipeline_seq;
    pthread_mutex_unlock(&session->pipeline_mutex);

    // Process the pipeline
    const gint64 t_pipe_start = g_get_monotonic_time();
    dt_dev_pixelpipe_t *pipe = session->dev.full.pipe;
    dt_dev_process_image_job(&session->dev, &session->dev.full, pipe, -1, DT_DEVICE_CPU);
    const gint64 t_pipe_end = g_get_monotonic_time();
    fprintf(stderr, "[perf] pipeline: %.1f ms\n",
            (t_pipe_end - t_pipe_start) / 1000.0);

    // Check if new params arrived while we were processing
    pthread_mutex_lock(&session->pipeline_mutex);
    if(session->pipeline_seq != start_seq)
    {
      // New params arrived — loop and reprocess with latest values
      pthread_mutex_unlock(&session->pipeline_mutex);
      fprintf(stderr, "[perf] pipeline: stale, reprocessing\n");
      continue;
    }
    pthread_mutex_unlock(&session->pipeline_mutex);

    // Pipeline is current — write to SHM and send event
    dt_pthread_mutex_lock(&pipe->backbuf_mutex);

    if(!pipe->backbuf || pipe->backbuf_width <= 0 || pipe->backbuf_height <= 0)
    {
      dt_pthread_mutex_unlock(&pipe->backbuf_mutex);
      fprintf(stderr, "[server] async pipeline: no output\n");
      break;
    }

    const int rendered_width = pipe->backbuf_width;
    const int rendered_height = pipe->backbuf_height;

    const int back = 1 - session->front_buffer;
    dt_shm_buffer_t *shm = &session->shm_buffers[back];

    session->frame_sequence++;

    dt_shm_write_header(shm, rendered_width, rendered_height,
                         DT_SHM_FORMAT_BGRA8, session->frame_sequence);

    uint8_t *dst = dt_shm_pixel_data(shm);
    const size_t copy_size = (size_t)rendered_width * rendered_height * 4;
    memcpy(dst, pipe->backbuf, copy_size);

    dt_pthread_mutex_unlock(&pipe->backbuf_mutex);

    __atomic_store_n(&shm->mapped->ready, 1, __ATOMIC_RELEASE);
    session->front_buffer = back;
    session->dirty = FALSE;

    // Queue preview_ready event
    {
      JsonBuilder *eb = json_builder_new();
      json_builder_begin_object(eb);
      json_builder_set_member_name(eb, "session_id");
      json_builder_add_string_value(eb, session->session_id);
      json_builder_set_member_name(eb, "front_buffer");
      json_builder_add_int_value(eb, session->front_buffer);
      json_builder_set_member_name(eb, "width");
      json_builder_add_int_value(eb, rendered_width);
      json_builder_set_member_name(eb, "height");
      json_builder_add_int_value(eb, rendered_height);
      json_builder_set_member_name(eb, "sequence");
      json_builder_add_int_value(eb, session->frame_sequence);
      json_builder_end_object(eb);

      JsonNode *event_data = json_builder_get_root(eb);
      dt_server_queue_event(server, "develop.preview_ready", event_data);
      json_node_unref(event_data);
      g_object_unref(eb);
    }

    {
      const gint64 t_event = g_get_monotonic_time();
      fprintf(stderr, "[perf] shm_write+event: %.1f ms  total_since_pipe_start: %.1f ms  %dx%d seq=%llu\n",
              (t_event - t_pipe_end) / 1000.0,
              (t_event - t_pipe_start) / 1000.0,
              rendered_width, rendered_height, (unsigned long long)session->frame_sequence);
    }
    break;
  }

  // Mark pipeline as no longer busy
  pthread_mutex_lock(&session->pipeline_mutex);
  session->pipeline_busy = FALSE;
  pthread_mutex_unlock(&session->pipeline_mutex);

  return NULL;
}

static void _maybe_start_pipeline(dt_server_t *server, dt_server_session_t *session)
{
  pthread_mutex_lock(&session->pipeline_mutex);
  if(session->pipeline_busy)
  {
    // Worker is already running — it will see the bumped pipeline_seq and reprocess
    pthread_mutex_unlock(&session->pipeline_mutex);
    return;
  }
  session->pipeline_busy = TRUE;
  pthread_mutex_unlock(&session->pipeline_mutex);

  _pipeline_job_t *job = g_new0(_pipeline_job_t, 1);
  job->server = server;
  job->session = session;

  pthread_t thread;
  if(dt_pthread_create(&thread, _preview_pipeline_worker, job) != 0)
  {
    fprintf(stderr, "[server] failed to create pipeline worker thread\n");
    g_free(job);
    pthread_mutex_lock(&session->pipeline_mutex);
    session->pipeline_busy = FALSE;
    pthread_mutex_unlock(&session->pipeline_mutex);
    return;
  }
  pthread_detach(thread);
}

char *dt_server_develop_set_params(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "op")
     || !json_object_has_member(req->params, "params"))
  {
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id, op, or params");
  }

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  const char *op = json_object_get_string_member(req->params, "op");
  if(!session_id || !op)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id and op must be strings");
  JsonObject *new_params = json_object_get_object_member(req->params, "params");
  if(!new_params)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "params must be an object");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  // Find the module
  dt_iop_module_t *target = NULL;
  for(GList *modules = session->dev.iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;
    if(dt_iop_module_is(mod->so, op))
    {
      target = mod;
      break;
    }
  }

  if(!target)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Module not found");

  // Check for preview_only mode (skip history during drag)
  const gboolean preview_only = json_object_has_member(req->params, "preview_only")
    && json_object_get_boolean_member(req->params, "preview_only");

  // Apply module-specific parameter updates
  if(!strcmp(op, "exposure"))
  {
    _server_exposure_params_t *p = (_server_exposure_params_t *)target->params;

    if(json_object_has_member(new_params, "mode"))
      p->mode = (_server_exposure_mode_t)json_object_get_int_member(new_params, "mode");
    if(json_object_has_member(new_params, "exposure"))
      p->exposure = (float)json_object_get_double_member(new_params, "exposure");
    if(json_object_has_member(new_params, "black"))
      p->black = (float)json_object_get_double_member(new_params, "black");
    if(json_object_has_member(new_params, "compensate_exposure_bias"))
      p->compensate_exposure_bias = json_object_get_boolean_member(new_params, "compensate_exposure_bias");
    if(json_object_has_member(new_params, "compensate_hilite_pres"))
      p->compensate_hilite_pres = json_object_get_boolean_member(new_params, "compensate_hilite_pres");
    if(json_object_has_member(new_params, "deflicker_percentile"))
      p->deflicker_percentile = (float)json_object_get_double_member(new_params, "deflicker_percentile");
    if(json_object_has_member(new_params, "deflicker_target_level"))
      p->deflicker_target_level = (float)json_object_get_double_member(new_params, "deflicker_target_level");
  }
  else
  {
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Module params not yet supported for this operation");
  }

  // Check if we should also set enabled state
  if(json_object_has_member(req->params, "enabled"))
    target->enabled = json_object_get_boolean_member(req->params, "enabled");

  if(preview_only)
  {
    // Preview-only: commit params directly to the pipe piece, bypassing history.
    // DT_DEV_PIPE_TOP_CHANGED triggers synch_top() which reads dev->history —
    // since we skipped history write, that would revert our changes.
    // Instead, commit to the pipe piece directly and invalidate its cache.
    dt_dev_pixelpipe_t *pipe = session->dev.full.pipe;
    for(GList *nodes = pipe->nodes; nodes; nodes = g_list_next(nodes))
    {
      dt_dev_pixelpipe_iop_t *piece = nodes->data;
      if(piece->module == target)
      {
        piece->enabled = target->enabled;
        dt_iop_commit_params(target, target->params,
                             target->blend_params ? target->blend_params
                               : target->default_blendop_params,
                             pipe, piece);
        // Invalidate cache from this module onwards
        dt_dev_pixelpipe_cache_invalidate_later(pipe, target->iop_order);
        break;
      }
    }
  }
  else
  {
    // Record the change to history.
    // Use _ext variant which bypasses the GUI check in dt_dev_add_history_item()
    // (the wrapper bails early when darktable.gui is NULL, i.e. headless/server mode).
    // no_image=TRUE avoids GUI widget operations that crash without a GUI.
    dt_dev_add_history_item_ext(&session->dev, target, target->enabled, TRUE);

    // Mark pipeline dirty — no_image=TRUE above skips pipe->changed, so set manually
    session->dev.full.pipe->changed |= DT_DEV_PIPE_TOP_CHANGED;
    if(session->dev.preview_pipe)
      session->dev.preview_pipe->changed |= DT_DEV_PIPE_TOP_CHANGED;
    if(session->dev.preview2.pipe)
      session->dev.preview2.pipe->changed |= DT_DEV_PIPE_TOP_CHANGED;
  }

  dt_dev_invalidate_all(&session->dev);
  session->dirty = TRUE;

  // Bump pipeline sequence and trigger async processing
  pthread_mutex_lock(&session->pipeline_mutex);
  session->pipeline_seq++;
  pthread_mutex_unlock(&session->pipeline_mutex);
  _maybe_start_pipeline(server, session);

  fprintf(stderr, "[server] develop.set_params: session=%s op=%s enabled=%d history_end=%d\n",
          session_id, op, target->enabled, session->dev.history_end);

  // Build response
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "ok");
  json_builder_set_member_name(b, "dirty");
  json_builder_add_boolean_value(b, session->dirty);
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_request_preview(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing session_id parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");
  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  // Process the pipeline (synchronous — blocks until done)
  dt_dev_pixelpipe_t *pipe = session->dev.full.pipe;

  fprintf(stderr, "[server] develop.request_preview: session=%s processing...\n",
          session_id);

  dt_dev_process_image_job(&session->dev, &session->dev.full, pipe, -1, DT_DEVICE_CPU);

  // Lock backbuf_mutex to safely access pipeline output
  dt_pthread_mutex_lock(&pipe->backbuf_mutex);

  if(!pipe->backbuf || pipe->backbuf_width <= 0 || pipe->backbuf_height <= 0)
  {
    dt_pthread_mutex_unlock(&pipe->backbuf_mutex);
    return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL,
                                 "Pipeline processing failed — no output");
  }

  const int rendered_width = pipe->backbuf_width;
  const int rendered_height = pipe->backbuf_height;

  fprintf(stderr, "[server] develop.request_preview: rendered %dx%d\n",
          rendered_width, rendered_height);

  // Write to back buffer (the one client is NOT reading)
  const int back = 1 - session->front_buffer;
  dt_shm_buffer_t *shm = &session->shm_buffers[back];

  session->frame_sequence++;

  // Update SHM header
  dt_shm_write_header(shm, rendered_width, rendered_height,
                       DT_SHM_FORMAT_BGRA8, session->frame_sequence);

  // Copy pixel data while holding the lock
  uint8_t *dst = dt_shm_pixel_data(shm);
  const size_t copy_size = (size_t)rendered_width * rendered_height * 4;
  memcpy(dst, pipe->backbuf, copy_size);

  dt_pthread_mutex_unlock(&pipe->backbuf_mutex);

  // Mark buffer as ready and swap
  __atomic_store_n(&shm->mapped->ready, 1, __ATOMIC_RELEASE);
  session->front_buffer = back;
  session->dirty = FALSE;

  // Queue a preview_ready event for the client
  {
    JsonBuilder *eb = json_builder_new();
    json_builder_begin_object(eb);

    json_builder_set_member_name(eb, "session_id");
    json_builder_add_string_value(eb, session->session_id);

    json_builder_set_member_name(eb, "shm_name");
    json_builder_add_string_value(eb, shm->name);

    json_builder_set_member_name(eb, "width");
    json_builder_add_int_value(eb, rendered_width);

    json_builder_set_member_name(eb, "height");
    json_builder_add_int_value(eb, rendered_height);

    json_builder_set_member_name(eb, "sequence");
    json_builder_add_int_value(eb, session->frame_sequence);

    json_builder_end_object(eb);

    JsonNode *event_data = json_builder_get_root(eb);
    dt_server_queue_event(server, "develop.preview_ready", event_data);
    json_node_unref(event_data);
    g_object_unref(eb);
  }

  // Build the direct response (client also gets the event asynchronously)
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "session_id");
  json_builder_add_string_value(b, session->session_id);

  json_builder_set_member_name(b, "width");
  json_builder_add_int_value(b, rendered_width);

  json_builder_set_member_name(b, "height");
  json_builder_add_int_value(b, rendered_height);

  json_builder_set_member_name(b, "sequence");
  json_builder_add_int_value(b, session->frame_sequence);

  json_builder_set_member_name(b, "shm_name");
  json_builder_add_string_value(b, shm->name);

  json_builder_set_member_name(b, "front_buffer");
  json_builder_add_int_value(b, session->front_buffer);

  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_commit_params(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "op"))
  {
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id or op");
  }

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  const char *op = json_object_get_string_member(req->params, "op");
  if(!session_id || !op)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id and op must be strings");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  // Find the module
  dt_iop_module_t *target = NULL;
  for(GList *modules = session->dev.iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;
    if(dt_iop_module_is(mod->so, op))
    {
      target = mod;
      break;
    }
  }

  if(!target)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Module not found");

  // Write current params to history (the params were already applied by preview_only set_params calls)
  dt_dev_add_history_item_ext(&session->dev, target, target->enabled, TRUE);

  fprintf(stderr, "[server] develop.commit_params: session=%s op=%s history_end=%d\n",
          session_id, op, session->dev.history_end);

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "ok");
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_delete_history(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing session_id parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  const dt_imgid_t imgid = session->imgid;

  // Delete history (no undo in server mode).
  // Pass init_history=FALSE because the TRUE path calls
  // dt_dev_reload_history_items(darktable.develop) which uses GUI code.
  dt_history_delete_on_image_ext(imgid, FALSE, FALSE);

  // Clear auto-presets-applied flag so dt_dev_read_history re-applies
  // default modules (sigmoid, etc.). This is what _remove_preset_flag()
  // does in the init_history=TRUE path.
  {
    dt_image_t *image = dt_image_cache_get(imgid, 'w');
    if(image)
    {
      image->flags &= ~DT_IMAGE_AUTO_PRESETS_APPLIED;
      dt_image_cache_write_release_info(image, DT_IMAGE_CACHE_SAFE,
                                        "server_delete_history");
    }
  }

  // Server-safe reload: dt_dev_reload_history_items() calls GUI code,
  // so we do the non-GUI parts inline here.
  dt_develop_t *dev = &session->dev;
  dev->focus_hash = FALSE;

  dt_lock_image(imgid);

  // Reset all modules to defaults
  dt_pthread_mutex_lock(&dev->history_mutex);
  dt_dev_pop_history_items_ext(dev, 0);
  dt_pthread_mutex_unlock(&dev->history_mutex);

  // Remove unused history items from the in-memory list
  GList *history = g_list_nth(dev->history, dev->history_end);
  while(history)
  {
    GList *next = g_list_next(history);
    dt_dev_history_item_t *hist = history->data;
    hist->module->multi_name_hand_edited = FALSE;
    g_strlcpy(hist->module->multi_name, "", sizeof(hist->module->multi_name));
    dt_dev_free_history_item(hist);
    dev->history = g_list_delete_link(dev->history, history);
    history = next;
  }

  // Re-read history from the now-clean database (will re-apply auto-presets)
  dt_dev_read_history(dev);
  dt_ioppr_set_default_iop_order(dev, imgid);

  // Apply the (now clean) history
  dt_pthread_mutex_lock(&dev->history_mutex);
  dt_dev_pop_history_items_ext(dev, dev->history_end);
  dt_pthread_mutex_unlock(&dev->history_mutex);

  dt_ioppr_resync_iop_list(dev);

  // Mark pipe dirty for re-rendering
  if(dev->full.pipe)
  {
    dev->full.pipe->changed |= DT_DEV_PIPE_REMOVE;
    dev->full.pipe->status = DT_DEV_PIXELPIPE_DIRTY;
  }

  dt_unlock_image(imgid);

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "ok");
  json_builder_end_object(b);

  JsonNode *res = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, res);
  json_node_unref(res);
  g_object_unref(b);
  return resp;
}
