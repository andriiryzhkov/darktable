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

#include "server/dt_server.h"
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
           "dev-%03d", server->session_count);
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

  // Initialize develop context following dt_dev_image() pattern (develop.c:3847)
  dt_dev_init(&session->dev, TRUE);
  session->dev.gui_attached = FALSE;

  // Configure the full pipeline for image processing mode
  dt_dev_pixelpipe_t *pipe = session->dev.full.pipe;
  pipe->type |= DT_DEV_PIXELPIPE_IMAGE;

  // Load image: instantiates modules, loads history from DB
  dt_dev_load_image(&session->dev, imgid);

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
  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  fprintf(stderr, "[server] develop.close: session=%s\n", session_id);

  // Destroy SHM buffers
  dt_shm_destroy(&session->shm_buffers[0]);
  dt_shm_destroy(&session->shm_buffers[1]);

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

char *dt_server_develop_get_params(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "op"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id or op parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  const char *op = json_object_get_string_member(req->params, "op");

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

char *dt_server_develop_set_params(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "op")
     || !json_object_has_member(req->params, "params"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id, op, or params");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  const char *op = json_object_get_string_member(req->params, "op");
  JsonObject *new_params = json_object_get_object_member(req->params, "params");

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

  // Apply module-specific parameter updates
  if(!strcmp(op, "exposure"))
  {
    _server_exposure_params_t *p = (_server_exposure_params_t *)target->params;

    if(json_object_has_member(new_params, "exposure"))
      p->exposure = (float)json_object_get_double_member(new_params, "exposure");
    if(json_object_has_member(new_params, "black"))
      p->black = (float)json_object_get_double_member(new_params, "black");
    if(json_object_has_member(new_params, "compensate_exposure_bias"))
      p->compensate_exposure_bias = json_object_get_boolean_member(new_params, "compensate_exposure_bias");
  }
  else
  {
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Module params not yet supported for this operation");
  }

  // Check if we should also set enabled state
  if(json_object_has_member(req->params, "enabled"))
    target->enabled = json_object_get_boolean_member(req->params, "enabled");

  // Record the change to history and mark pipeline dirty
  dt_dev_add_history_item(&session->dev, target, target->enabled);
  session->dirty = TRUE;

  fprintf(stderr, "[server] develop.set_params: session=%s op=%s\n",
          session_id, op);

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
  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  // Process the pipeline (synchronous — blocks until done)
  dt_dev_pixelpipe_t *pipe = session->dev.full.pipe;

  fprintf(stderr, "[server] develop.request_preview: session=%s processing...\n",
          session_id);

  dt_dev_process_image_job(&session->dev, &session->dev.full, pipe, -1, DT_DEVICE_CPU);

  if(!pipe->backbuf || pipe->backbuf_width <= 0 || pipe->backbuf_height <= 0)
  {
    return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL,
                                 "Pipeline processing failed — no output");
  }

  fprintf(stderr, "[server] develop.request_preview: rendered %dx%d\n",
          pipe->backbuf_width, pipe->backbuf_height);

  // Write to back buffer (the one client is NOT reading)
  const int back = 1 - session->front_buffer;
  dt_shm_buffer_t *shm = &session->shm_buffers[back];

  session->frame_sequence++;

  // Update SHM header
  dt_shm_write_header(shm, pipe->backbuf_width, pipe->backbuf_height,
                       DT_SHM_FORMAT_BGRA8, session->frame_sequence);

  // Copy pixel data: backbuf is uint8_t RGBA (4 bytes/pixel) = BGRA8 in our SHM format
  // Note: darktable outputs sRGB after colorout. The exact channel order
  // depends on the pipeline (typically BGRA on output). Copy as-is.
  uint8_t *dst = dt_shm_pixel_data(shm);
  const size_t copy_size = (size_t)pipe->backbuf_width * pipe->backbuf_height * 4;
  memcpy(dst, pipe->backbuf, copy_size);

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
    json_builder_add_int_value(eb, pipe->backbuf_width);

    json_builder_set_member_name(eb, "height");
    json_builder_add_int_value(eb, pipe->backbuf_height);

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
  json_builder_add_int_value(b, pipe->backbuf_width);

  json_builder_set_member_name(b, "height");
  json_builder_add_int_value(b, pipe->backbuf_height);

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
