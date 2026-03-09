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

#include "bindings.h"
#include "ipc.h"
#include "path_validation.h"
#include "titlebar.h"
#include "transport.h"
#include "server/server_protocol.h"

#include <errno.h>
#include <fcntl.h>
#include <glib/gstdio.h>
#include <jpeglib.h>
#include <json-glib/json-glib.h>
#include <nfd.h>
#include <pthread.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/mman.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#endif
#include <unistd.h>

/* ── Adaptive JPEG quality ────────────────────────────────────
 * When enabled, the frame server accepts a `q=` query parameter
 * to control JPEG quality.  During interactive editing (slider
 * drag) the client requests lower quality for faster feedback;
 * on release it requests full quality for the final preview.
 * Set to 0 to always use JPEG_QUALITY_FULL.
 */
#define ADAPTIVE_JPEG          1
#define JPEG_QUALITY_FULL      92
#define JPEG_QUALITY_INTERACTIVE 60

/* ── Thread pool for binding workers ──────────────────────────
 * Instead of spawning a new pthread per binding call, we maintain
 * a fixed pool of worker threads and dispatch via GAsyncQueue.
 */
#define BINDING_POOL_SIZE 4

typedef struct _pool_work_item_t
{
  void *(*func)(void *);  // worker function
  void *arg;              // argument (typically async_req_t*)
} _pool_work_item_t;

static GAsyncQueue *_work_queue = NULL;
static pthread_t _pool_threads[BINDING_POOL_SIZE];
static int _pool_initialized = 0;

static void *_pool_worker(void *unused)
{
  (void)unused;
  while(1)
  {
    _pool_work_item_t *item = g_async_queue_pop(_work_queue);
    if(!item->func)
    {
      /* Sentinel: NULL func means shutdown */
      g_free(item);
      break;
    }
    item->func(item->arg);
    g_free(item);
  }
  return NULL;
}

static void _pool_dispatch(void *(*func)(void *), void *arg)
{
  _pool_work_item_t *item = g_new(_pool_work_item_t, 1);
  item->func = func;
  item->arg = arg;
  g_async_queue_push(_work_queue, item);
}

static void _binding_pool_init(void)
{
  if(_pool_initialized) return;
  _work_queue = g_async_queue_new();
  for(int i = 0; i < BINDING_POOL_SIZE; i++)
    pthread_create(&_pool_threads[i], NULL, _pool_worker, NULL);
  _pool_initialized = 1;
}

void dt_binding_pool_shutdown(void)
{
  if(!_pool_initialized) return;
  /* Push sentinel items to wake and stop each worker */
  for(int i = 0; i < BINDING_POOL_SIZE; i++)
  {
    _pool_work_item_t *sentinel = g_new0(_pool_work_item_t, 1);
    sentinel->func = NULL;
    g_async_queue_push(_work_queue, sentinel);
  }
  for(int i = 0; i < BINDING_POOL_SIZE; i++)
    pthread_join(_pool_threads[i], NULL);
  g_async_queue_unref(_work_queue);
  _work_queue = NULL;
  _pool_initialized = 0;
}

/* async callback infrastructure */

typedef struct async_req_t
{
  dt_webview_ctx_t *ctx;
  char *id;
  char *req;
} async_req_t;

static async_req_t *_async_req_new(dt_webview_ctx_t *ctx, const char *id, const char *req)
{
  async_req_t *ar = g_new0(async_req_t, 1);
  ar->ctx = ctx;
  ar->id = g_strdup(id);
  ar->req = g_strdup(req);
  return ar;
}

static void _async_req_free(async_req_t *ar)
{
  g_free(ar->id);
  g_free(ar->req);
  g_free(ar);
}

static void _return_ok(dt_webview_ctx_t *ctx, const char *id, const char *json_result)
{
  webview_return(ctx->webview, id, 0, json_result ? json_result : "null");
}

static void _return_error(dt_webview_ctx_t *ctx, const char *id, const char *msg)
{
  // Use json-glib to properly escape the message for JSON
  JsonNode *node = json_node_new(JSON_NODE_VALUE);
  json_node_set_string(node, msg);
  JsonGenerator *gen = json_generator_new();
  json_generator_set_root(gen, node);
  char *json = json_generator_to_data(gen, NULL);
  g_object_unref(gen);
  json_node_unref(node);
  webview_return(ctx->webview, id, 1, json);
  g_free(json);
}

// Helper: parse JSON array string into a JsonArray.
// Returns NULL on error. Caller must unref the parser.
static JsonArray *_parse_args(const char *req, JsonParser **out_parser)
{
  *out_parser = json_parser_new();
  if(!json_parser_load_from_data(*out_parser, req, -1, NULL))
  {
    g_object_unref(*out_parser);
    *out_parser = NULL;
    return NULL;
  }
  JsonNode *root = json_parser_get_root(*out_parser);
  if(!root || !JSON_NODE_HOLDS_ARRAY(root))
  {
    g_object_unref(*out_parser);
    *out_parser = NULL;
    return NULL;
  }
  return json_node_get_array(root);
}

/* Path validation is in path_validation.h (shared with tests) */
#define _path_is_allowed dt_path_is_allowed

// Helper: get an owned copy of a string argument from a parsed JSON array.
// Returns a g_strdup'd string that the caller must g_free.
// Safe to use after g_object_unref(parser).
static char *_get_string_arg(JsonArray *args, guint index)
{
  const char *s = json_array_get_string_element(args, index);
  return s ? g_strdup(s) : NULL;
}

/* ── Server event bridge ──────────────────────────────────────── */

typedef struct _event_dispatch_t
{
  dt_webview_ctx_t *ctx;
  char *event_name;
  char *data_json;
} _event_dispatch_t;

static void _event_dispatch_main(webview_t w, void *arg)
{
  (void)w;
  _event_dispatch_t *ed = arg;

  // Build JS call: window.__dt_event('event_name', {data})
  char *js = g_strdup_printf(
    "window.__dt_event && window.__dt_event('%s', %s);",
    ed->event_name, ed->data_json);
  webview_eval(ed->ctx->webview, js);
  g_free(js);

  g_free(ed->event_name);
  g_free(ed->data_json);
  g_free(ed);
}

// Called from IPC reader thread — must dispatch to webview's main thread
static void _on_server_event(const char *event_name, const char *data_json, void *user_data)
{
  dt_webview_ctx_t *ctx = user_data;
  fprintf(stderr, "[webview] event received: %s\n", event_name);

  _event_dispatch_t *ed = g_new0(_event_dispatch_t, 1);
  ed->ctx = ctx;
  ed->event_name = g_strdup(event_name);
  ed->data_json = g_strdup(data_json);
  webview_dispatch(ctx->webview, _event_dispatch_main, ed);
}

/* IPC passthrough: send method+params via IPC, return result to JS */
typedef struct ipc_passthrough_t
{
  dt_webview_ctx_t *ctx;
  char *id;
  char *method;
  char *params_json;
} ipc_passthrough_t;

static void *_ipc_passthrough_worker(void *arg)
{
  ipc_passthrough_t *pt = arg;
  char *error = NULL;

  char *result = dt_transport_call(pt->ctx->transport, pt->method, pt->params_json, &error);

  if(result)
    _return_ok(pt->ctx, pt->id, result);
  else
    _return_error(pt->ctx, pt->id, error ? error : "IPC request failed");

  g_free(result);
  g_free(error);
  g_free(pt->id);
  g_free(pt->method);
  g_free(pt->params_json);
  g_free(pt);
  return NULL;
}

static void _ipc_passthrough(dt_webview_ctx_t *ctx, const char *id,
                              const char *method, const char *params_json)
{
  ipc_passthrough_t *pt = g_new0(ipc_passthrough_t, 1);
  pt->ctx = ctx;
  pt->id = g_strdup(id);
  pt->method = g_strdup(method);
  pt->params_json = g_strdup(params_json ? params_json : "{}");

  _pool_dispatch(_ipc_passthrough_worker, pt);
}

static void on_ping(const char *id, const char *req, void *arg)
{
  (void)req;
  _ipc_passthrough(arg, id, "system.ping", "{}");
}

static void on_catalog_query(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 2)
  {
    _return_error(ctx, id, "catalogQuery requires (offset, limit[, rules])");
    if(parser) g_object_unref(parser);
    return;
  }

  gint64 offset = json_array_get_int_element(args, 0);
  gint64 limit = json_array_get_int_element(args, 1);

  // Optional 3rd arg: rules array
  char *rules_json = NULL;
  if(json_array_get_length(args) >= 3)
  {
    JsonNode *rules_node = json_array_get_element(args, 2);
    if(rules_node && JSON_NODE_HOLDS_ARRAY(rules_node))
    {
      JsonGenerator *gen = json_generator_new();
      json_generator_set_root(gen, rules_node);
      rules_json = json_generator_to_data(gen, NULL);
      g_object_unref(gen);
    }
  }

  // Optional 4th arg: sort field, 5th arg: sort order
  // g_strdup because json_node_get_string returns pointer into parser memory
  char *sort_field = NULL;
  char *sort_order = NULL;
  if(json_array_get_length(args) >= 4)
  {
    JsonNode *n = json_array_get_element(args, 3);
    if(n && JSON_NODE_HOLDS_VALUE(n))
      sort_field = g_strdup(json_node_get_string(n));
  }
  if(json_array_get_length(args) >= 5)
  {
    JsonNode *n = json_array_get_element(args, 4);
    if(n && JSON_NODE_HOLDS_VALUE(n))
      sort_order = g_strdup(json_node_get_string(n));
  }
  g_object_unref(parser);

  GString *params_str = g_string_new("{");
  g_string_append_printf(params_str, "\"offset\":%" G_GINT64_FORMAT
                                     ",\"limit\":%" G_GINT64_FORMAT, offset, limit);
  if(rules_json)
  {
    g_string_append_printf(params_str, ",\"rules\":%s", rules_json);
    g_free(rules_json);
  }
  if(sort_field)
    g_string_append_printf(params_str, ",\"sort\":\"%s\"", sort_field);
  if(sort_order)
    g_string_append_printf(params_str, ",\"sort_order\":\"%s\"", sort_order);
  g_string_append_c(params_str, '}');

  g_free(sort_field);
  g_free(sort_order);
  char *params = g_string_free(params_str, FALSE);
  _ipc_passthrough(ctx, id, "catalog.query", params);
  g_free(params);
}

static void on_catalog_get_thumbnail(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, id, "catalogGetThumbnail requires (imgid)");
    if(parser) g_object_unref(parser);
    return;
  }

  gint64 imgid = json_array_get_int_element(args, 0);
  g_object_unref(parser);

  char *params = g_strdup_printf("{\"imgid\":%" G_GINT64_FORMAT "}", imgid);
  _ipc_passthrough(ctx, id, "catalog.get_thumbnail", params);
  g_free(params);
}

static void on_catalog_get_thumbnails(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, id, "catalogGetThumbnails requires (imgids, [size])");
    if(parser) g_object_unref(parser);
    return;
  }

  // args[0] = array of imgids, args[1] = optional size
  JsonArray *imgids = json_array_get_array_element(args, 0);
  gint64 size = 720;
  if(json_array_get_length(args) > 1
     && json_node_get_node_type(json_array_get_element(args, 1)) == JSON_NODE_VALUE)
    size = json_array_get_int_element(args, 1);
  if(size <= 0) size = 720;

  // Build params JSON: {"imgids":[...],"size":N}
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "imgids");
  json_builder_begin_array(b);
  for(guint i = 0; i < json_array_get_length(imgids); i++)
    json_builder_add_int_value(b, json_array_get_int_element(imgids, i));
  json_builder_end_array(b);
  json_builder_set_member_name(b, "size");
  json_builder_add_int_value(b, size);
  json_builder_end_object(b);

  JsonGenerator *gen = json_generator_new();
  json_generator_set_root(gen, json_builder_get_root(b));
  char *params = json_generator_to_data(gen, NULL);
  g_object_unref(gen);
  g_object_unref(b);
  g_object_unref(parser);

  _ipc_passthrough(ctx, id, "catalog.get_thumbnails", params);
  g_free(params);
}

static void *_develop_open_worker(void *arg)
{
  async_req_t *ar = arg;
  dt_webview_ctx_t *ctx = ar->ctx;

  // Parse args: [imgid, width, height]
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(ar->req, &parser);
  if(!args || json_array_get_length(args) < 3)
  {
    _return_error(ctx, ar->id, "developOpen requires (imgid, width, height)");
    if(parser) g_object_unref(parser);
    _async_req_free(ar);
    return NULL;
  }

  gint64 imgid = json_array_get_int_element(args, 0);
  gint64 width = json_array_get_int_element(args, 1);
  gint64 height = json_array_get_int_element(args, 2);
  g_object_unref(parser);

  // IPC request via transport
  char *params = g_strdup_printf("{\"imgid\":%" G_GINT64_FORMAT
                                  ",\"width\":%" G_GINT64_FORMAT
                                  ",\"height\":%" G_GINT64_FORMAT "}", imgid, width, height);
  char *error = NULL;
  char *result = dt_transport_call(ctx->transport, "develop.open", params, &error);
  g_free(params);

  if(!result)
  {
    _return_error(ctx, ar->id, error ? error : "develop.open failed");
    g_free(error);
    _async_req_free(ar);
    return NULL;
  }

  // Parse result to get shm_names and open SHM handles
  JsonParser *rparser = json_parser_new();
  if(json_parser_load_from_data(rparser, result, -1, NULL))
  {
    JsonObject *robj = json_node_get_object(json_parser_get_root(rparser));
    const char *session_id = json_object_get_string_member(robj, "session_id");
    JsonArray *shm_names = json_object_get_array_member(robj, "shm_names");
    gint64 pw = json_object_get_int_member(robj, "preview_width");
    gint64 ph = json_object_get_int_member(robj, "preview_height");

    if(session_id)
    {
      pthread_mutex_lock(&ctx->session_mutex);
      // Find empty slot
      int slot = -1;
      for(int i = 0; i < DT_WEBVIEW_MAX_SESSIONS; i++)
      {
        if(!ctx->sessions[i].active)
        {
          slot = i;
          break;
        }
      }
      if(slot >= 0)
      {
        dt_webview_shm_t *s = &ctx->sessions[slot];
        memset(s, 0, sizeof(*s));
        g_strlcpy(s->session_id, session_id, sizeof(s->session_id));
        s->width = (uint32_t)pw;
        s->height = (uint32_t)ph;

        // Store SHM names for lazy mapping (server allocates SHM on first render)
        if(shm_names && json_array_get_length(shm_names) >= 2)
        {
          const char *name0 = json_array_get_string_element(shm_names, 0);
          const char *name1 = json_array_get_string_element(shm_names, 1);
          size_t shm_size = DT_SHM_HEADER_SIZE + (size_t)pw * (size_t)ph * 4;

          g_strlcpy(s->shm_names[0], name0, sizeof(s->shm_names[0]));
          g_strlcpy(s->shm_names[1], name1, sizeof(s->shm_names[1]));
          s->shm_size[0] = shm_size;
          s->shm_size[1] = shm_size;
          // shm_ptr[0..1] remain NULL — mapped lazily on first frame read
        }
        // else: direct mode — no SHM, frame server uses transport->get_preview_frame()

        s->active = 1;
      }
      pthread_mutex_unlock(&ctx->session_mutex);
    }
  }
  g_object_unref(rparser);

  _return_ok(ctx, ar->id, result);
  g_free(result);
  _async_req_free(ar);
  return NULL;
}

static void on_develop_open(const char *id, const char *req, void *arg)
{
  _pool_dispatch(_develop_open_worker, _async_req_new(arg, id, req));
}

static void *_develop_close_worker(void *arg)
{
  async_req_t *ar = arg;
  dt_webview_ctx_t *ctx = ar->ctx;

  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(ar->req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, ar->id, "developClose requires (sessionId)");
    if(parser) g_object_unref(parser);
    _async_req_free(ar);
    return NULL;
  }

  char *session_id = _get_string_arg(args, 0);
  g_object_unref(parser);

  // Close SHM handles
  pthread_mutex_lock(&ctx->session_mutex);
  for(int i = 0; i < DT_WEBVIEW_MAX_SESSIONS; i++)
  {
    if(ctx->sessions[i].active && !strcmp(ctx->sessions[i].session_id, session_id))
    {
      for(int b = 0; b < 2; b++)
      {
        if(ctx->sessions[i].shm_ptr[b])
        {
          munmap(ctx->sessions[i].shm_ptr[b], ctx->sessions[i].shm_size[b]);
          ctx->sessions[i].shm_ptr[b] = NULL;
        }
      }
      ctx->sessions[i].active = 0;
      break;
    }
  }
  pthread_mutex_unlock(&ctx->session_mutex);

  // IPC close via transport
  char *params = g_strdup_printf("{\"session_id\":\"%s\"}", session_id);
  g_free(session_id);

  char *error = NULL;
  char *result = dt_transport_call(ctx->transport, "develop.close", params, &error);
  g_free(params);

  if(result)
    _return_ok(ctx, ar->id, result);
  else
    _return_error(ctx, ar->id, error ? error : "develop.close failed");

  g_free(result);
  g_free(error);
  _async_req_free(ar);
  return NULL;
}

static void on_develop_close(const char *id, const char *req, void *arg)
{
  _pool_dispatch(_develop_close_worker, _async_req_new(arg, id, req));
}

static void on_develop_set_params(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 3)
  {
    _return_error(ctx, id, "developSetParams requires (sessionId, op, params)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  char *op = _get_string_arg(args, 1);

  // Serialize the params object back to JSON string
  JsonNode *params_node = json_array_get_element(args, 2);
  JsonGenerator *gen = json_generator_new();
  json_generator_set_root(gen, params_node);
  char *params_str = json_generator_to_data(gen, NULL);
  g_object_unref(gen);

  // Optional 4th arg: previewOnly (boolean) — skip history write during drag
  gboolean preview_only = FALSE;
  if(json_array_get_length(args) >= 4)
    preview_only = json_array_get_boolean_element(args, 3);

  g_object_unref(parser);

  char *ipc_params;
  if(preview_only)
    ipc_params = g_strdup_printf("{\"session_id\":\"%s\",\"op\":\"%s\",\"params\":%s,\"preview_only\":true}",
                                 session_id, op, params_str);
  else
    ipc_params = g_strdup_printf("{\"session_id\":\"%s\",\"op\":\"%s\",\"params\":%s}",
                                 session_id, op, params_str);
  g_free(params_str);
  g_free(session_id);
  g_free(op);

  _ipc_passthrough(ctx, id, "develop.set_params", ipc_params);
  g_free(ipc_params);
}

static void on_develop_commit_params(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 2)
  {
    _return_error(ctx, id, "developCommitParams requires (sessionId, op)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  char *op = _get_string_arg(args, 1);
  g_object_unref(parser);

  char *ipc_params = g_strdup_printf("{\"session_id\":\"%s\",\"op\":\"%s\"}",
                                     session_id, op);
  g_free(session_id);
  g_free(op);

  _ipc_passthrough(ctx, id, "develop.commit_params", ipc_params);
  g_free(ipc_params);
}

static void on_develop_reset_params(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 2)
  {
    _return_error(ctx, id, "developResetParams requires (sessionId, op)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  char *op = _get_string_arg(args, 1);
  g_object_unref(parser);

  char *ipc_params = g_strdup_printf("{\"session_id\":\"%s\",\"op\":\"%s\"}",
                                     session_id, op);
  g_free(session_id);
  g_free(op);

  _ipc_passthrough(ctx, id, "develop.reset_params", ipc_params);
  g_free(ipc_params);
}

static void on_develop_get_params(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 2)
  {
    _return_error(ctx, id, "developGetParams requires (sessionId, op)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  char *op = _get_string_arg(args, 1);
  g_object_unref(parser);

  char *ipc_params = g_strdup_printf("{\"session_id\":\"%s\",\"op\":\"%s\"}",
                                     session_id, op);
  g_free(session_id);
  g_free(op);

  _ipc_passthrough(ctx, id, "develop.get_params", ipc_params);
  g_free(ipc_params);
}

static void on_develop_get_introspection(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 2)
  {
    _return_error(ctx, id, "developGetIntrospection requires (sessionId, op)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  char *op = _get_string_arg(args, 1);
  g_object_unref(parser);

  char *ipc_params = g_strdup_printf("{\"session_id\":\"%s\",\"op\":\"%s\"}",
                                     session_id, op);
  g_free(session_id);
  g_free(op);

  _ipc_passthrough(ctx, id, "develop.get_introspection", ipc_params);
  g_free(ipc_params);
}

static void on_develop_get_masks(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, id, "developGetMasks requires (sessionId)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  g_object_unref(parser);

  char *params = g_strdup_printf("{\"session_id\":\"%s\"}", session_id);
  g_free(session_id);

  _ipc_passthrough(ctx, id, "develop.get_masks", params);
  g_free(params);
}

static void on_develop_rename_mask(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 3)
  {
    _return_error(ctx, id, "developRenameMask requires (sessionId, formid, name)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  const gint64 formid = json_array_get_int_element(args, 1);
  char *name = _get_string_arg(args, 2);
  g_object_unref(parser);

  char *escaped_name = g_strescape(name, NULL);
  char *params = g_strdup_printf("{\"session_id\":\"%s\",\"formid\":%" G_GINT64_FORMAT ",\"name\":\"%s\"}",
                                  session_id, formid, escaped_name);
  g_free(session_id);
  g_free(name);
  g_free(escaped_name);

  _ipc_passthrough(ctx, id, "develop.rename_mask", params);
  g_free(params);
}

static void on_develop_delete_mask(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 2)
  {
    _return_error(ctx, id, "developDeleteMask requires (sessionId, formid)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  const gint64 formid = json_array_get_int_element(args, 1);
  g_object_unref(parser);

  char *params = g_strdup_printf("{\"session_id\":\"%s\",\"formid\":%" G_GINT64_FORMAT "}",
                                  session_id, formid);
  g_free(session_id);

  _ipc_passthrough(ctx, id, "develop.delete_mask", params);
  g_free(params);
}

static void on_develop_request_preview(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, id, "developRequestPreview requires (sessionId)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  g_object_unref(parser);

  char *params = g_strdup_printf("{\"session_id\":\"%s\"}", session_id);
  g_free(session_id);

  _ipc_passthrough(ctx, id, "develop.request_preview", params);
  g_free(params);
}

static void on_develop_sample_pixels(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 5)
  {
    _return_error(ctx, id, "developSamplePixels requires (sessionId, x, y, w, h)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  double x = json_array_get_double_element(args, 1);
  double y = json_array_get_double_element(args, 2);
  double w = json_array_get_double_element(args, 3);
  double h = json_array_get_double_element(args, 4);
  g_object_unref(parser);

  char *ipc_params = g_strdup_printf(
    "{\"session_id\":\"%s\",\"x\":%f,\"y\":%f,\"w\":%f,\"h\":%f}",
    session_id, x, y, w, h);
  g_free(session_id);

  _ipc_passthrough(ctx, id, "develop.sample_pixels", ipc_params);
  g_free(ipc_params);
}

static void on_develop_get_modules(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, id, "developGetModules requires (sessionId)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  g_object_unref(parser);

  char *params = g_strdup_printf("{\"session_id\":\"%s\"}", session_id);
  g_free(session_id);

  _ipc_passthrough(ctx, id, "develop.get_modules", params);
  g_free(params);
}

static void on_develop_get_history(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, id, "developGetHistory requires (sessionId)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  g_object_unref(parser);

  char *params = g_strdup_printf("{\"session_id\":\"%s\"}", session_id);
  g_free(session_id);

  _ipc_passthrough(ctx, id, "develop.get_history", params);
  g_free(params);
}

static void on_develop_select_history(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 2)
  {
    _return_error(ctx, id, "developSelectHistory requires (sessionId, historyEnd)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  int history_end = (int)json_array_get_int_element(args, 1);
  g_object_unref(parser);

  char *params = g_strdup_printf("{\"session_id\":\"%s\",\"history_end\":%d}", session_id, history_end);
  g_free(session_id);

  _ipc_passthrough(ctx, id, "develop.select_history", params);
  g_free(params);
}

static void on_develop_compress_history(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, id, "developCompressHistory requires (sessionId)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  g_object_unref(parser);

  char *params = g_strdup_printf("{\"session_id\":\"%s\"}", session_id);
  g_free(session_id);

  _ipc_passthrough(ctx, id, "develop.compress_history", params);
  g_free(params);
}

static void on_develop_truncate_history(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 2)
  {
    _return_error(ctx, id, "developTruncateHistory requires (sessionId, historyEnd)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  int history_end = (int)json_array_get_int_element(args, 1);
  g_object_unref(parser);

  char *params = g_strdup_printf("{\"session_id\":\"%s\",\"history_end\":%d}", session_id, history_end);
  g_free(session_id);

  _ipc_passthrough(ctx, id, "develop.truncate_history", params);
  g_free(params);
}

static void on_develop_delete_history(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, id, "developDeleteHistory requires (sessionId)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  g_object_unref(parser);

  char *params = g_strdup_printf("{\"session_id\":\"%s\"}", session_id);
  g_free(session_id);

  _ipc_passthrough(ctx, id, "develop.delete_history", params);
  g_free(params);
}

static void on_develop_list_presets(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 2)
  {
    _return_error(ctx, id, "developListPresets requires (sessionId, op)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  char *op = _get_string_arg(args, 1);
  g_object_unref(parser);

  char *params = g_strdup_printf("{\"session_id\":\"%s\",\"op\":\"%s\"}", session_id, op);
  g_free(session_id);
  g_free(op);

  _ipc_passthrough(ctx, id, "develop.list_presets", params);
  g_free(params);
}

static void on_develop_apply_preset(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 3)
  {
    _return_error(ctx, id, "developApplyPreset requires (sessionId, op, name)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  char *op = _get_string_arg(args, 1);
  char *name = _get_string_arg(args, 2);
  g_object_unref(parser);

  // Build JSON with proper escaping via JsonBuilder
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "session_id");
  json_builder_add_string_value(b, session_id);
  json_builder_set_member_name(b, "op");
  json_builder_add_string_value(b, op);
  json_builder_set_member_name(b, "name");
  json_builder_add_string_value(b, name);
  json_builder_end_object(b);

  JsonGenerator *gen = json_generator_new();
  JsonNode *root = json_builder_get_root(b);
  json_generator_set_root(gen, root);
  char *params = json_generator_to_data(gen, NULL);
  json_node_unref(root);
  g_object_unref(gen);
  g_object_unref(b);
  g_free(session_id);
  g_free(op);
  g_free(name);

  _ipc_passthrough(ctx, id, "develop.apply_preset", params);
  g_free(params);
}

static void on_develop_store_preset(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 3)
  {
    _return_error(ctx, id, "developStorePreset requires (sessionId, op, name, [description])");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  char *op = _get_string_arg(args, 1);
  char *name = _get_string_arg(args, 2);
  char *description = (json_array_get_length(args) > 3) ? _get_string_arg(args, 3) : g_strdup("");
  char *filters_json = (json_array_get_length(args) > 4) ? _get_string_arg(args, 4) : NULL;

  // Parse filters JSON if provided
  JsonObject *filters = NULL;
  JsonParser *filters_parser = NULL;
  if(filters_json && filters_json[0])
  {
    filters_parser = json_parser_new();
    if(json_parser_load_from_data(filters_parser, filters_json, -1, NULL))
    {
      JsonNode *fnode = json_parser_get_root(filters_parser);
      if(fnode && JSON_NODE_HOLDS_OBJECT(fnode))
        filters = json_node_get_object(fnode);
    }
  }
  g_object_unref(parser);

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "session_id");
  json_builder_add_string_value(b, session_id);
  json_builder_set_member_name(b, "op");
  json_builder_add_string_value(b, op);
  json_builder_set_member_name(b, "name");
  json_builder_add_string_value(b, name);
  json_builder_set_member_name(b, "description");
  json_builder_add_string_value(b, description);

  if(filters)
  {
    // Forward filter fields into the params object
    static const char *str_fields[] = { "model", "maker", "lens", NULL };
    for(const char **f = str_fields; *f; f++)
      if(json_object_has_member(filters, *f))
      {
        json_builder_set_member_name(b, *f);
        json_builder_add_string_value(b, json_object_get_string_member(filters, *f));
      }

    static const char *real_fields[] = {
      "iso_min", "iso_max", "exposure_min", "exposure_max",
      "aperture_min", "aperture_max", "focal_length_min", "focal_length_max", NULL
    };
    for(const char **f = real_fields; *f; f++)
      if(json_object_has_member(filters, *f))
      {
        json_builder_set_member_name(b, *f);
        json_builder_add_double_value(b, json_object_get_double_member(filters, *f));
      }

    static const char *int_fields[] = { "autoapply", "filter", "format", NULL };
    for(const char **f = int_fields; *f; f++)
      if(json_object_has_member(filters, *f))
      {
        json_builder_set_member_name(b, *f);
        // booleans come as true/false, integers as numbers
        JsonNode *node = json_object_get_member(filters, *f);
        if(JSON_NODE_HOLDS_VALUE(node) && json_node_get_value_type(node) == G_TYPE_BOOLEAN)
          json_builder_add_int_value(b, json_node_get_boolean(node) ? 1 : 0);
        else
          json_builder_add_int_value(b, json_object_get_int_member(filters, *f));
      }
  }

  json_builder_end_object(b);

  JsonGenerator *gen = json_generator_new();
  JsonNode *root = json_builder_get_root(b);
  json_generator_set_root(gen, root);
  char *params = json_generator_to_data(gen, NULL);
  json_node_unref(root);
  g_object_unref(gen);
  g_object_unref(b);
  if(filters_parser) g_object_unref(filters_parser);
  g_free(session_id);
  g_free(op);
  g_free(name);
  g_free(description);
  g_free(filters_json);

  _ipc_passthrough(ctx, id, "develop.store_preset", params);
  g_free(params);
}

static void on_develop_delete_preset(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 2)
  {
    _return_error(ctx, id, "developDeletePreset requires (op, name)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *op = _get_string_arg(args, 0);
  char *name = _get_string_arg(args, 1);
  g_object_unref(parser);

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "op");
  json_builder_add_string_value(b, op);
  json_builder_set_member_name(b, "name");
  json_builder_add_string_value(b, name);
  json_builder_end_object(b);

  JsonGenerator *gen = json_generator_new();
  JsonNode *root = json_builder_get_root(b);
  json_generator_set_root(gen, root);
  char *params = json_generator_to_data(gen, NULL);
  json_node_unref(root);
  g_object_unref(gen);
  g_object_unref(b);
  g_free(op);
  g_free(name);

  _ipc_passthrough(ctx, id, "develop.delete_preset", params);
  g_free(params);
}

static void on_develop_new_instance(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 2)
  {
    _return_error(ctx, id, "developNewInstance requires (sessionId, op, [instance], [copyParams])");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  char *op = _get_string_arg(args, 1);
  const int instance = json_array_get_length(args) > 2
    ? (int)json_array_get_int_element(args, 2) : 0;
  const gboolean copy_params = json_array_get_length(args) > 3
    && json_array_get_boolean_element(args, 3);
  g_object_unref(parser);

  char *params = g_strdup_printf(
    "{\"session_id\":\"%s\",\"op\":\"%s\",\"instance\":%d,\"copy_params\":%s}",
    session_id, op, instance, copy_params ? "true" : "false");
  g_free(session_id);
  g_free(op);

  _ipc_passthrough(ctx, id, "develop.new_instance", params);
  g_free(params);
}

static void on_develop_delete_instance(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 3)
  {
    _return_error(ctx, id, "developDeleteInstance requires (sessionId, op, instance)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  char *op = _get_string_arg(args, 1);
  const int instance = (int)json_array_get_int_element(args, 2);
  g_object_unref(parser);

  char *params = g_strdup_printf(
    "{\"session_id\":\"%s\",\"op\":\"%s\",\"instance\":%d}",
    session_id, op, instance);
  g_free(session_id);
  g_free(op);

  _ipc_passthrough(ctx, id, "develop.delete_instance", params);
  g_free(params);
}

static void on_develop_move_instance(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 4)
  {
    _return_error(ctx, id, "developMoveInstance requires (sessionId, op, instance, direction)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  char *op = _get_string_arg(args, 1);
  const int instance = (int)json_array_get_int_element(args, 2);
  char *direction = _get_string_arg(args, 3);
  g_object_unref(parser);

  char *params = g_strdup_printf(
    "{\"session_id\":\"%s\",\"op\":\"%s\",\"instance\":%d,\"direction\":\"%s\"}",
    session_id, op, instance, direction);
  g_free(session_id);
  g_free(op);
  g_free(direction);

  _ipc_passthrough(ctx, id, "develop.move_instance", params);
  g_free(params);
}

static void on_develop_rename_instance(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 4)
  {
    _return_error(ctx, id, "developRenameInstance requires (sessionId, op, instance, name)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *session_id = _get_string_arg(args, 0);
  char *op = _get_string_arg(args, 1);
  const int instance = (int)json_array_get_int_element(args, 2);
  char *name = _get_string_arg(args, 3);
  g_object_unref(parser);

  char *params = g_strdup_printf(
    "{\"session_id\":\"%s\",\"op\":\"%s\",\"instance\":%d,\"name\":\"%s\"}",
    session_id, op, instance, name);
  g_free(session_id);
  g_free(op);
  g_free(name);

  _ipc_passthrough(ctx, id, "develop.rename_instance", params);
  g_free(params);
}

/* getPreviewFrame: reads pixels directly from SHM, no IPC */

/* Minimal memory-to-memory JPEG compressor using libjpeg.
   Input: RGBA 4-byte pixels. Output: JPEG bytes in `out`.
   Returns compressed size in bytes, or 0 on failure. */
struct _jpeg_err_mgr
{
  struct jpeg_error_mgr pub;
  jmp_buf setjmp_buffer;
};

static void _jpeg_error_exit(j_common_ptr cinfo)
{
  struct _jpeg_err_mgr *err = (struct _jpeg_err_mgr *)cinfo->err;
  longjmp(err->setjmp_buffer, 1);
}

static int _jpeg_compress_rgba(const uint8_t *in, uint8_t *out,
                                const int width, const int height,
                                const size_t out_buf_size, const int quality)
{
  struct _jpeg_err_mgr jerr;
  struct jpeg_compress_struct cinfo;
  unsigned long out_size = (unsigned long)out_buf_size;

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = _jpeg_error_exit;
  if(setjmp(jerr.setjmp_buffer))
  {
    jpeg_destroy_compress(&cinfo);
    return 0;
  }

  jpeg_create_compress(&cinfo);

  /* Use libjpeg's built-in memory destination */
  uint8_t *mem_dest = out;
  jpeg_mem_dest(&cinfo, &mem_dest, &out_size);

  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE);
  if(quality > 90) cinfo.comp_info[0].v_samp_factor = 1;
  if(quality > 92) cinfo.comp_info[0].h_samp_factor = 1;

  jpeg_start_compress(&cinfo, TRUE);

  /* Strip alpha: RGBA → RGB per scanline */
  uint8_t *row = g_malloc(3 * width);
  while(cinfo.next_scanline < cinfo.image_height)
  {
    const uint8_t *src = in + cinfo.next_scanline * width * 4;
    for(int i = 0; i < width; i++)
    {
      row[3 * i + 0] = src[4 * i + 0];
      row[3 * i + 1] = src[4 * i + 1];
      row[3 * i + 2] = src[4 * i + 2];
    }
    JSAMPROW tmp[1] = { row };
    jpeg_write_scanlines(&cinfo, tmp, 1);
  }

  jpeg_finish_compress(&cinfo);
  g_free(row);
  jpeg_destroy_compress(&cinfo);

  return (int)out_size;
}

/* ── Local HTTP frame server ─────────────────────────────────────
 * Serves JPEG frames directly from SHM over HTTP, avoiding
 * base64 encoding.  The browser loads frames via:
 *   <img src="http://localhost:PORT/frame?s=SESSION&b=BUF">
 * ─────────────────────────────────────────────────────────────── */

struct dt_frame_server_t
{
  int listen_fd;
  int port;
  dt_webview_ctx_t *ctx;
  gboolean running;
  pthread_t thread;

  /* Test eval infrastructure — allows external tools (MCP server) to
   * execute JS in the webview and get results back via HTTP. */
  GMutex test_mutex;
  GCond test_cond;
  char *test_result;
  gboolean test_result_ready;
};

/* Parse a query param value: find "key=" in qs, copy value up to '&' or ' ' or end */
static int _qs_param(const char *qs, const char *key, char *out, size_t out_size)
{
  char needle[64];
  snprintf(needle, sizeof(needle), "%s=", key);
  const char *p = strstr(qs, needle);
  if(!p) return 0;
  p += strlen(needle);
  size_t i = 0;
  while(*p && *p != '&' && *p != ' ' && *p != '\r' && i < out_size - 1)
    out[i++] = *p++;
  out[i] = '\0';
  return 1;
}

static void _send_http_response(int fd, int code, const char *status,
                                const char *content_type,
                                const void *body, size_t body_len)
{
  char hdr[512];
  int hlen = snprintf(hdr, sizeof(hdr),
    "HTTP/1.1 %d %s\r\n"
    "Content-Type: %s\r\n"
    "Content-Length: %zu\r\n"
    "Cache-Control: no-store\r\n"
    "Access-Control-Allow-Origin: *\r\n"
    "Connection: close\r\n"
    "\r\n",
    code, status, content_type, body_len);
  if(write(fd, hdr, hlen) < 0) return;
  if(body && body_len > 0)
  {
    size_t written = 0;
    while(written < body_len)
    {
      ssize_t n = write(fd, (const char *)body + written, body_len - written);
      if(n <= 0) break;
      written += n;
    }
  }
}

// Lazily map a SHM buffer for a session.  Called with session_mutex held.
// Returns the mapped pointer, or NULL on failure.
static void *_ensure_shm_mapped(dt_webview_shm_t *s, int buf_idx)
{
  if(s->shm_ptr[buf_idx])
    return s->shm_ptr[buf_idx];

  if(!s->shm_names[buf_idx][0])
    return NULL;  // no SHM name (direct mode)

  int fd = shm_open(s->shm_names[buf_idx], O_RDONLY, 0);
  if(fd < 0)
    return NULL;  // server hasn't created SHM yet

  void *ptr = mmap(NULL, s->shm_size[buf_idx], PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if(ptr == MAP_FAILED)
  {
    fprintf(stderr, "[webview] mmap(%s) failed: %s\n", s->shm_names[buf_idx], strerror(errno));
    return NULL;
  }

  s->shm_ptr[buf_idx] = ptr;
  return ptr;
}

static void _handle_frame_request(int client_fd, dt_webview_ctx_t *ctx,
                                  const char *session_id, int buffer_idx,
                                  int jpeg_quality)
{
  pthread_mutex_lock(&ctx->session_mutex);
  dt_webview_shm_t *session = NULL;
  for(int i = 0; i < DT_WEBVIEW_MAX_SESSIONS; i++)
  {
    if(ctx->sessions[i].active && !strcmp(ctx->sessions[i].session_id, session_id))
    {
      session = &ctx->sessions[i];
      break;
    }
  }

  if(!session)
  {
    pthread_mutex_unlock(&ctx->session_mutex);
    _send_http_response(client_fd, 404, "Not Found", "text/plain", "session not found", 17);
    return;
  }

  int buf_idx = (buffer_idx == 0) ? 0 : 1;
  void *ptr = _ensure_shm_mapped(session, buf_idx);

  /* Direct mode: no SHM — use transport->get_preview_frame() */
  if(!ptr && ctx->transport)
  {
    pthread_mutex_unlock(&ctx->session_mutex);

    dt_transport_frame_t frame = {0};
    if(!dt_transport_get_preview_frame(ctx->transport, session_id, &frame))
    {
      _send_http_response(client_fd, 503, "Not Ready", "text/plain", "frame not ready", 15);
      return;
    }

    uint32_t w = frame.width;
    uint32_t h = frame.height;
    size_t pixel_size = (size_t)w * h * 4;

    /* BGRA → RGBA */
    uint8_t *rgba = g_malloc(pixel_size);
    for(size_t i = 0; i < pixel_size; i += 4)
    {
      rgba[i + 0] = frame.pixels[i + 2];
      rgba[i + 1] = frame.pixels[i + 1];
      rgba[i + 2] = frame.pixels[i + 0];
      rgba[i + 3] = frame.pixels[i + 3];
    }
    if(frame.owned) g_free((void *)frame.pixels);

    /* JPEG encode */
    size_t jpeg_buf_size = pixel_size + 1024;
    uint8_t *jpeg_buf = g_malloc(jpeg_buf_size);
    const int jpeg_size = _jpeg_compress_rgba(rgba, jpeg_buf, w, h, jpeg_buf_size, jpeg_quality);
    g_free(rgba);

    if(jpeg_size > 0)
      _send_http_response(client_fd, 200, "OK", "image/jpeg", jpeg_buf, jpeg_size);
    else
      _send_http_response(client_fd, 500, "Error", "text/plain", "JPEG encode failed", 18);

    g_free(jpeg_buf);
    return;
  }

  if(!ptr)
  {
    pthread_mutex_unlock(&ctx->session_mutex);
    _send_http_response(client_fd, 404, "Not Found", "text/plain", "buffer not mapped", 17);
    return;
  }

  dt_shm_header_t *header = (dt_shm_header_t *)ptr;
  if(header->magic != DT_SHM_MAGIC || header->version != DT_SHM_VERSION)
  {
    pthread_mutex_unlock(&ctx->session_mutex);
    _send_http_response(client_fd, 500, "Error", "text/plain", "bad SHM header", 14);
    return;
  }

  uint32_t ready = __atomic_load_n(&header->ready, __ATOMIC_ACQUIRE);
  if(ready != 1)
  {
    pthread_mutex_unlock(&ctx->session_mutex);
    _send_http_response(client_fd, 503, "Not Ready", "text/plain", "frame not ready", 15);
    return;
  }

  uint32_t w = header->width;
  uint32_t h = header->height;
  uint32_t stride = header->stride;
  size_t pixel_size = (size_t)stride * (size_t)h;
  uint8_t *pixels = (uint8_t *)ptr + DT_SHM_HEADER_SIZE;

  /* BGRA → RGBA */
  uint8_t *rgba = g_malloc(pixel_size);
  for(size_t i = 0; i < pixel_size; i += 4)
  {
    rgba[i + 0] = pixels[i + 2];
    rgba[i + 1] = pixels[i + 1];
    rgba[i + 2] = pixels[i + 0];
    rgba[i + 3] = pixels[i + 3];
  }
  pthread_mutex_unlock(&ctx->session_mutex);

  /* JPEG encode */
  size_t jpeg_buf_size = pixel_size + 1024;
  uint8_t *jpeg_buf = g_malloc(jpeg_buf_size);
  const int jpeg_size = _jpeg_compress_rgba(rgba, jpeg_buf, w, h, jpeg_buf_size, jpeg_quality);
  g_free(rgba);

  if(jpeg_size > 0)
    _send_http_response(client_fd, 200, "OK", "image/jpeg", jpeg_buf, jpeg_size);
  else
    _send_http_response(client_fd, 500, "Error", "text/plain", "JPEG encode failed", 18);

  g_free(jpeg_buf);
}

/** Serve raw BGRA pixels from SHM — zero conversion, zero encoding.
 *  Width/height sent as custom headers so JS can size the WebGL canvas. */
static void _handle_raw_request(int client_fd, dt_webview_ctx_t *ctx,
                                const char *session_id, int buffer_idx)
{
  const gint64 t0 = g_get_monotonic_time();
  pthread_mutex_lock(&ctx->session_mutex);
  dt_webview_shm_t *session = NULL;
  for(int i = 0; i < DT_WEBVIEW_MAX_SESSIONS; i++)
  {
    if(ctx->sessions[i].active && !strcmp(ctx->sessions[i].session_id, session_id))
    {
      session = &ctx->sessions[i];
      break;
    }
  }
  if(!session)
  {
    pthread_mutex_unlock(&ctx->session_mutex);
    _send_http_response(client_fd, 404, "Not Found", "text/plain", "session not found", 17);
    return;
  }

  int buf_idx = (buffer_idx == 0) ? 0 : 1;
  void *ptr = _ensure_shm_mapped(session, buf_idx);

  /* Direct mode: no SHM — use transport->get_preview_frame() */
  if(!ptr && ctx->transport)
  {
    pthread_mutex_unlock(&ctx->session_mutex);

    dt_transport_frame_t frame = {0};
    if(!dt_transport_get_preview_frame(ctx->transport, session_id, &frame))
    {
      _send_http_response(client_fd, 503, "Not Ready", "text/plain", "frame not ready", 15);
      return;
    }

    uint32_t w = frame.width;
    uint32_t h = frame.height;
    size_t pixel_size = (size_t)w * h * 4;

    char hdr[512];
    int hlen = snprintf(hdr, sizeof(hdr),
      "HTTP/1.1 200 OK\r\n"
      "Content-Type: application/octet-stream\r\n"
      "Content-Length: %zu\r\n"
      "X-Width: %u\r\n"
      "X-Height: %u\r\n"
      "Cache-Control: no-store\r\n"
      "Access-Control-Allow-Origin: *\r\n"
      "Access-Control-Expose-Headers: X-Width, X-Height\r\n"
      "Connection: close\r\n"
      "\r\n",
      pixel_size, w, h);
    if(write(client_fd, hdr, hlen) < 0)
    {
      if(frame.owned) g_free((void *)frame.pixels);
      return;
    }

    size_t written = 0;
    while(written < pixel_size)
    {
      ssize_t n = write(client_fd, frame.pixels + written, pixel_size - written);
      if(n <= 0) break;
      written += n;
    }
    if(frame.owned) g_free((void *)frame.pixels);

    fprintf(stderr, "[perf] raw_serve(direct): %.1f ms (%ux%u, %zu bytes)\n",
            (g_get_monotonic_time() - t0) / 1000.0, w, h, pixel_size);
    return;
  }

  if(!ptr)
  {
    pthread_mutex_unlock(&ctx->session_mutex);
    _send_http_response(client_fd, 404, "Not Found", "text/plain", "buffer not mapped", 17);
    return;
  }

  dt_shm_header_t *header = (dt_shm_header_t *)ptr;
  if(header->magic != DT_SHM_MAGIC || header->version != DT_SHM_VERSION)
  {
    pthread_mutex_unlock(&ctx->session_mutex);
    _send_http_response(client_fd, 500, "Error", "text/plain", "bad SHM header", 14);
    return;
  }

  uint32_t ready = __atomic_load_n(&header->ready, __ATOMIC_ACQUIRE);
  if(ready != 1)
  {
    pthread_mutex_unlock(&ctx->session_mutex);
    _send_http_response(client_fd, 503, "Not Ready", "text/plain", "frame not ready", 15);
    return;
  }

  uint32_t w = header->width;
  uint32_t h = header->height;
  uint32_t stride = header->stride;
  size_t pixel_size = (size_t)stride * (size_t)h;
  uint8_t *pixels = (uint8_t *)ptr + DT_SHM_HEADER_SIZE;

  /* Send raw BGRA pixels directly — WebGL shader handles B↔R swap */
  char hdr[512];
  int hlen = snprintf(hdr, sizeof(hdr),
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: application/octet-stream\r\n"
    "Content-Length: %zu\r\n"
    "X-Width: %u\r\n"
    "X-Height: %u\r\n"
    "Cache-Control: no-store\r\n"
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Expose-Headers: X-Width, X-Height\r\n"
    "Connection: close\r\n"
    "\r\n",
    pixel_size, w, h);
  if(write(client_fd, hdr, hlen) < 0)
  {
    pthread_mutex_unlock(&ctx->session_mutex);
    return;
  }

  /* Stream directly from SHM — no copy, no conversion */
  size_t written = 0;
  while(written < pixel_size)
  {
    ssize_t n = write(client_fd, pixels + written, pixel_size - written);
    if(n <= 0) break;
    written += n;
  }
  pthread_mutex_unlock(&ctx->session_mutex);

  fprintf(stderr, "[perf] raw_serve: %.1f ms (%ux%u, %zu bytes)\n",
          (g_get_monotonic_time() - t0) / 1000.0, w, h, pixel_size);
}

/* ── Test eval: execute JS in the webview from HTTP ──────────────────
 * Used by the MCP server to automate the UI for testing.
 *
 * Flow: POST /test/eval  →  webview_dispatch(webview_eval)  →  JS calls
 *       window.__testResult(json)  →  binding stores result  →  HTTP response
 */

/* Binding callback: JS calls __testResult(jsonString) to return eval results */
static void on_test_result(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  dt_frame_server_t *fs = ctx->frame_server;
  if(!fs) { webview_return(ctx->webview, id, 0, "null"); return; }

  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);

  g_mutex_lock(&fs->test_mutex);
  g_free(fs->test_result);
  if(args && json_array_get_length(args) > 0)
    fs->test_result = g_strdup(json_array_get_string_element(args, 0));
  else
    fs->test_result = g_strdup("{\"ok\":false,\"error\":\"no result\"}");
  fs->test_result_ready = TRUE;
  g_cond_signal(&fs->test_cond);
  g_mutex_unlock(&fs->test_mutex);

  if(parser) g_object_unref(parser);
  webview_return(ctx->webview, id, 0, "null");
}

/* Dispatch struct for running webview_eval on the main thread */
typedef struct _test_eval_dispatch_t
{
  dt_webview_ctx_t *ctx;
  char *js;
} _test_eval_dispatch_t;

static void _test_eval_on_main(webview_t w, void *arg)
{
  _test_eval_dispatch_t *d = arg;
  webview_eval(d->ctx->webview, d->js);
  g_free(d->js);
  g_free(d);
}

/* HTTP handler for POST /test/eval */
static void _handle_test_eval(int client_fd, dt_frame_server_t *fs, const char *body)
{
  if(!body || !*body)
  {
    const char *err = "{\"ok\":false,\"error\":\"empty code\"}";
    _send_http_response(client_fd, 400, "Bad Request", "application/json", err, strlen(err));
    return;
  }

  /* Reset result */
  g_mutex_lock(&fs->test_mutex);
  fs->test_result_ready = FALSE;
  g_free(fs->test_result);
  fs->test_result = NULL;
  g_mutex_unlock(&fs->test_mutex);

  /* Base64-encode the user code so it can be safely embedded in JS */
  gchar *b64 = g_base64_encode((const guchar *)body, strlen(body));

  /* Build JS wrapper: decode base64 → eval → call __testResult with result */
  char *js = g_strdup_printf(
    "(async()=>{"
    "try{"
    "const __code=atob('%s');"
    "const __r=await(0,eval)(__code);"
    "window.__testResult(JSON.stringify({ok:true,"
    "value:typeof __r==='undefined'?null:__r}));"
    "}catch(__e){"
    "window.__testResult(JSON.stringify({ok:false,"
    "error:__e.message,stack:__e.stack}));"
    "}})()",
    b64);
  g_free(b64);

  /* Dispatch eval to the webview's main thread */
  _test_eval_dispatch_t *d = g_new0(_test_eval_dispatch_t, 1);
  d->ctx = fs->ctx;
  d->js = js;
  webview_dispatch(fs->ctx->webview, _test_eval_on_main, d);

  /* Wait for __testResult callback (10 second timeout) */
  g_mutex_lock(&fs->test_mutex);
  gint64 deadline = g_get_monotonic_time() + 10 * G_USEC_PER_SEC;
  while(!fs->test_result_ready)
  {
    if(!g_cond_wait_until(&fs->test_cond, &fs->test_mutex, deadline))
    {
      g_mutex_unlock(&fs->test_mutex);
      const char *err = "{\"ok\":false,\"error\":\"timeout (10s)\"}";
      _send_http_response(client_fd, 504, "Timeout", "application/json", err, strlen(err));
      return;
    }
  }
  char *result = g_strdup(fs->test_result);
  g_mutex_unlock(&fs->test_mutex);

  _send_http_response(client_fd, 200, "OK", "application/json", result, strlen(result));
  g_free(result);
}

static void *_frame_server_loop(void *arg)
{
  dt_frame_server_t *fs = arg;

  while(fs->running)
  {
    int client = accept(fs->listen_fd, NULL, NULL);
    if(client < 0)
    {
      if(fs->running) perror("[frame-server] accept");
      break;
    }

    /* Read the HTTP request */
    char buf[65536];
    ssize_t n = read(client, buf, sizeof(buf) - 1);
    if(n > 0)
    {
      buf[n] = '\0';

      /* Route: POST /test/eval */
      if(strncmp(buf, "POST /test/eval", 15) == 0)
      {
        char *body = strstr(buf, "\r\n\r\n");
        if(body) body += 4;
        _handle_test_eval(client, fs, body);
        close(client);
        continue;
      }

      /* Route: GET /test/port — return port for MCP discovery */
      if(strncmp(buf, "GET /test/port", 14) == 0)
      {
        char port_json[64];
        snprintf(port_json, sizeof(port_json), "{\"port\":%d}", fs->port);
        _send_http_response(client, 200, "OK", "application/json", port_json, strlen(port_json));
        close(client);
        continue;
      }

      /* Route: GET /raw?... or GET /frame?... */
      char *query = strchr(buf, '?');
      gboolean is_raw = (strstr(buf, "GET /raw") == buf);
      if(query)
      {
        char sid[128] = {0};
        char bidx_str[16] = {0};
        char q_str[8] = {0};
        if(_qs_param(query, "s", sid, sizeof(sid))
           && _qs_param(query, "b", bidx_str, sizeof(bidx_str)))
        {
          /* Parse optional quality parameter (default: full quality) */
          int jpeg_quality = JPEG_QUALITY_FULL;
          if(ADAPTIVE_JPEG && _qs_param(query, "q", q_str, sizeof(q_str)))
          {
            int q = atoi(q_str);
            if(q >= 10 && q <= 100) jpeg_quality = q;
          }

          if(is_raw)
            _handle_raw_request(client, fs->ctx, sid, atoi(bidx_str));
          else
            _handle_frame_request(client, fs->ctx, sid, atoi(bidx_str), jpeg_quality);
        }
        else
        {
          _send_http_response(client, 400, "Bad Request", "text/plain", "missing params", 14);
        }
      }
      else
      {
        _send_http_response(client, 400, "Bad Request", "text/plain", "no query string", 15);
      }
    }
    close(client);
  }
  return NULL;
}

static dt_frame_server_t *_frame_server_start(dt_webview_ctx_t *ctx)
{
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if(fd < 0)
  {
    perror("[frame-server] socket");
    return NULL;
  }

  int opt = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0; /* OS picks a free port */

  if(bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
  {
    perror("[frame-server] bind");
    close(fd);
    return NULL;
  }

  if(listen(fd, 4) < 0)
  {
    perror("[frame-server] listen");
    close(fd);
    return NULL;
  }

  struct sockaddr_in bound;
  socklen_t len = sizeof(bound);
  getsockname(fd, (struct sockaddr *)&bound, &len);

  dt_frame_server_t *fs = g_new0(dt_frame_server_t, 1);
  fs->listen_fd = fd;
  fs->port = ntohs(bound.sin_port);
  fs->ctx = ctx;
  fs->running = TRUE;
  g_mutex_init(&fs->test_mutex);
  g_cond_init(&fs->test_cond);

  pthread_t thread;
  if(pthread_create(&thread, NULL, _frame_server_loop, fs) != 0)
  {
    perror("[frame-server] pthread_create");
    close(fd);
    g_mutex_clear(&fs->test_mutex);
    g_cond_clear(&fs->test_cond);
    g_free(fs);
    return NULL;
  }
  pthread_detach(thread);
  fs->thread = thread;

  fprintf(stderr, "[frame-server] listening on localhost:%d\n", fs->port);

  /* Write port to discoverable file for MCP server */
  char port_path[PATH_MAX];
  snprintf(port_path, sizeof(port_path), "%s/darktable_test_port", g_get_tmp_dir());
  FILE *port_file = fopen(port_path, "w");
  if(port_file)
  {
    fprintf(port_file, "%d", fs->port);
    fclose(port_file);
    fprintf(stderr, "[frame-server] test port written to %s\n", port_path);
  }
  return fs;
}

void dt_frame_server_stop(dt_frame_server_t *fs)
{
  if(!fs) return;
  fs->running = FALSE;
  close(fs->listen_fd);

  /* Clean up test eval resources */
  g_mutex_lock(&fs->test_mutex);
  g_free(fs->test_result);
  fs->test_result = NULL;
  g_mutex_unlock(&fs->test_mutex);
  g_mutex_clear(&fs->test_mutex);
  g_cond_clear(&fs->test_cond);

  /* Remove port discovery file */
  char port_path[PATH_MAX];
  snprintf(port_path, sizeof(port_path), "%s/darktable_test_port", g_get_tmp_dir());
  g_unlink(port_path);

  g_free(fs);
}

static void *_get_preview_frame_worker(void *arg)
{
  async_req_t *ar = arg;
  dt_webview_ctx_t *ctx = ar->ctx;

  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(ar->req, &parser);
  if(!args || json_array_get_length(args) < 2)
  {
    _return_error(ctx, ar->id, "getPreviewFrame requires (sessionId, frontBuffer)");
    if(parser) g_object_unref(parser);
    _async_req_free(ar);
    return NULL;
  }

  char *session_id = g_strdup(json_array_get_string_element(args, 0));
  gint64 front_buffer = json_array_get_int_element(args, 1);

  g_object_unref(parser);

  // Find session
  pthread_mutex_lock(&ctx->session_mutex);
  dt_webview_shm_t *session = NULL;
  for(int i = 0; i < DT_WEBVIEW_MAX_SESSIONS; i++)
  {
    if(ctx->sessions[i].active && !strcmp(ctx->sessions[i].session_id, session_id))
    {
      session = &ctx->sessions[i];
      break;
    }
  }

  if(!session)
  {
    pthread_mutex_unlock(&ctx->session_mutex);
    fprintf(stderr, "[webview] getPreviewFrame: session '%s' not found\n", session_id);
    _return_error(ctx, ar->id, "session not found");
    g_free(session_id);
    _async_req_free(ar);
    return NULL;
  }

  int buf_idx = (front_buffer == 0) ? 0 : 1;
  void *ptr = _ensure_shm_mapped(session, buf_idx);

  fprintf(stderr, "[webview] getPreviewFrame: session=%s buf=%d ptr=%p\n",
          session_id, buf_idx, ptr);

  /* Direct mode: no SHM — use transport->get_preview_frame() */
  if(!ptr && ctx->transport)
  {
    pthread_mutex_unlock(&ctx->session_mutex);
    fprintf(stderr, "[webview] getPreviewFrame: using transport fallback for session %s\n", session_id);

    dt_transport_frame_t frame = {0};
    if(!dt_transport_get_preview_frame(ctx->transport, session_id, &frame))
    {
      _return_error(ctx, ar->id, "frame not ready");
      g_free(session_id);
      _async_req_free(ar);
      return NULL;
    }

    uint32_t w = frame.width;
    uint32_t h = frame.height;
    size_t pixel_size = (size_t)w * h * 4;

    /* BGRA → RGBA */
    uint8_t *rgba = g_malloc(pixel_size);
    for(size_t i = 0; i < pixel_size; i += 4)
    {
      rgba[i + 0] = frame.pixels[i + 2];
      rgba[i + 1] = frame.pixels[i + 1];
      rgba[i + 2] = frame.pixels[i + 0];
      rgba[i + 3] = frame.pixels[i + 3];
    }
    if(frame.owned) g_free((void *)frame.pixels);

    size_t jpeg_buf_size = pixel_size + 1024;
    uint8_t *jpeg_buf = g_malloc(jpeg_buf_size);
    const int jpeg_size = _jpeg_compress_rgba(rgba, jpeg_buf, w, h, jpeg_buf_size, JPEG_QUALITY_FULL);
    g_free(rgba);

    if(jpeg_size > 0)
    {
      gchar *b64 = g_base64_encode(jpeg_buf, jpeg_size);
      char *result = g_strdup_printf("{\"width\":%u,\"height\":%u,\"format\":\"jpeg\",\"data\":\"%s\"}", w, h, b64);
      g_free(b64);
      _return_ok(ctx, ar->id, result);
      g_free(result);
    }
    else
    {
      _return_error(ctx, ar->id, "JPEG encode failed");
    }
    g_free(jpeg_buf);
    g_free(session_id);
    _async_req_free(ar);
    return NULL;
  }

  if(!ptr)
  {
    pthread_mutex_unlock(&ctx->session_mutex);
    fprintf(stderr, "[webview] getPreviewFrame: SHM buffer %d not mapped\n", buf_idx);
    _return_error(ctx, ar->id, "SHM buffer not mapped");
    g_free(session_id);
    _async_req_free(ar);
    return NULL;
  }

  // Validate SHM header
  dt_shm_header_t *header = (dt_shm_header_t *)ptr;
  fprintf(stderr, "[webview] getPreviewFrame: magic=0x%08x version=%u ready=%u w=%u h=%u\n",
          header->magic, header->version, header->ready, header->width, header->height);

  if(header->magic != DT_SHM_MAGIC || header->version != DT_SHM_VERSION)
  {
    pthread_mutex_unlock(&ctx->session_mutex);
    _return_error(ctx, ar->id, "invalid SHM header");
    g_free(session_id);
    _async_req_free(ar);
    return NULL;
  }

  uint32_t ready = __atomic_load_n(&header->ready, __ATOMIC_ACQUIRE);
  if(ready != 1)
  {
    pthread_mutex_unlock(&ctx->session_mutex);
    fprintf(stderr, "[webview] getPreviewFrame: frame not ready (ready=%u)\n", ready);
    _return_error(ctx, ar->id, "frame not ready");
    g_free(session_id);
    _async_req_free(ar);
    return NULL;
  }

  uint32_t w = header->width;
  uint32_t h = header->height;
  uint32_t stride = header->stride;
  size_t pixel_size = (size_t)stride * (size_t)h;
  uint8_t *pixels = (uint8_t *)ptr + DT_SHM_HEADER_SIZE;

  /* BGRA → RGBA */
  uint8_t *rgba = g_malloc(pixel_size);
  for(size_t i = 0; i < pixel_size; i += 4)
  {
    rgba[i + 0] = pixels[i + 2];
    rgba[i + 1] = pixels[i + 1];
    rgba[i + 2] = pixels[i + 0];
    rgba[i + 3] = pixels[i + 3];
  }
  pthread_mutex_unlock(&ctx->session_mutex);

  size_t jpeg_buf_size = pixel_size + 1024;
  uint8_t *jpeg_buf = g_malloc(jpeg_buf_size);
  const int jpeg_size = _jpeg_compress_rgba(rgba, jpeg_buf, w, h, jpeg_buf_size, JPEG_QUALITY_FULL);
  g_free(rgba);

  gchar *b64;
  char *result;
  if(jpeg_size > 0)
  {
    b64 = g_base64_encode(jpeg_buf, jpeg_size);
    result = g_strdup_printf("{\"width\":%u,\"height\":%u,\"format\":\"jpeg\",\"data\":\"%s\"}", w, h, b64);
    g_free(b64);
  }
  else
  {
    _return_error(ctx, ar->id, "JPEG encode failed");
    g_free(jpeg_buf);
    g_free(session_id);
    _async_req_free(ar);
    return NULL;
  }
  g_free(jpeg_buf);

  _return_ok(ctx, ar->id, result);
  g_free(result);
  g_free(session_id);
  _async_req_free(ar);
  return NULL;
}

static void on_get_preview_frame(const char *id, const char *req, void *arg)
{
  _pool_dispatch(_get_preview_frame_worker, _async_req_new(arg, id, req));
}

static void on_get_frame_port(const char *id, const char *req, void *arg)
{
  (void)req;
  dt_webview_ctx_t *ctx = arg;
  int port = ctx->frame_server ? ctx->frame_server->port : 0;
  char result[32];
  snprintf(result, sizeof(result), "%d", port);
  _return_ok(ctx, id, result);
}

/* ── Filesystem browsing (cross-platform via GLib) ───────────── */

/* Known RAW / image extensions (case-insensitive match) */
static const char *_raw_extensions[] = {
  "3fr", "ari", "arw", "bay", "braw", "cap", "cr2", "cr3",
  "crw", "dcr", "dcs", "dng", "drf", "eip", "erf", "fff",
  "gpr", "iiq", "k25", "kdc", "mdc", "mef", "mos", "mrw",
  "nef", "nrw", "obm", "orf", "pef", "ptx", "pxn", "r3d",
  "raf", "raw", "rw2", "rwl", "rwz", "sr2", "srf", "srw",
  "x3f",
  NULL
};

static const char *_image_extensions[] = {
  /* RAW formats are also valid images */
  "3fr", "ari", "arw", "bay", "braw", "cap", "cr2", "cr3",
  "crw", "dcr", "dcs", "dng", "drf", "eip", "erf", "fff",
  "gpr", "iiq", "k25", "kdc", "mdc", "mef", "mos", "mrw",
  "nef", "nrw", "obm", "orf", "pef", "ptx", "pxn", "r3d",
  "raf", "raw", "rw2", "rwl", "rwz", "sr2", "srf", "srw",
  "x3f",
  /* Common raster formats */
  "jpg", "jpeg", "tif", "tiff", "png", "exr", "hdr",
  "pfm", "pnm", "pbm", "pgm", "ppm",
  "j2k", "j2c", "jp2", "jpc",
  "avif", "heif", "heic", "hif",
  "webp",
  NULL
};

static int _is_extension_in(const char *filename, const char **ext_list)
{
  const char *dot = strrchr(filename, '.');
  if(!dot || dot == filename) return 0;
  const char *ext = dot + 1;
  for(const char **e = ext_list; *e; e++)
    if(g_ascii_strcasecmp(ext, *e) == 0) return 1;
  return 0;
}

static int _dir_has_subdirs(const char *path)
{
  GDir *d = g_dir_open(path, 0, NULL);
  if(!d) return 0;
  const gchar *name;
  while((name = g_dir_read_name(d)) != NULL)
  {
    if(name[0] == '.') continue;  // skip hidden
    gchar *child = g_build_filename(path, name, NULL);
    gboolean is_dir = g_file_test(child, G_FILE_TEST_IS_DIR);
    g_free(child);
    if(is_dir)
    {
      g_dir_close(d);
      return 1;
    }
  }
  g_dir_close(d);
  return 0;
}

/* listFolders(path) → [{name, path, hasChildren}, …] */
static void *_list_folders_worker(void *arg)
{
  async_req_t *ar = arg;
  dt_webview_ctx_t *ctx = ar->ctx;

  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(ar->req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, ar->id, "listFolders requires (path)");
    if(parser) g_object_unref(parser);
    _async_req_free(ar);
    return NULL;
  }

  const char *path = json_array_get_string_element(args, 0);

  if(!_path_is_allowed(path))
  {
    _return_error(ctx, ar->id, "path not allowed");
    g_object_unref(parser);
    _async_req_free(ar);
    return NULL;
  }

  GError *err = NULL;
  GDir *d = g_dir_open(path, 0, &err);
  if(!d)
  {
    _return_error(ctx, ar->id, err ? err->message : "cannot open directory");
    g_clear_error(&err);
    g_object_unref(parser);
    _async_req_free(ar);
    return NULL;
  }

  JsonBuilder *builder = json_builder_new();
  json_builder_begin_array(builder);

  const gchar *name;
  while((name = g_dir_read_name(d)) != NULL)
  {
    if(name[0] == '.') continue;  // skip hidden

    gchar *fullpath = g_build_filename(path, name, NULL);

    if(!g_file_test(fullpath, G_FILE_TEST_IS_DIR))
    {
      g_free(fullpath);
      continue;
    }

    json_builder_begin_object(builder);
    json_builder_set_member_name(builder, "name");
    json_builder_add_string_value(builder, name);
    json_builder_set_member_name(builder, "path");
    json_builder_add_string_value(builder, fullpath);
    json_builder_set_member_name(builder, "hasChildren");
    json_builder_add_boolean_value(builder, _dir_has_subdirs(fullpath));
    json_builder_end_object(builder);

    g_free(fullpath);
  }
  g_dir_close(d);

  json_builder_end_array(builder);
  JsonNode *root = json_builder_get_root(builder);
  JsonGenerator *gen = json_generator_new();
  json_generator_set_root(gen, root);
  char *json = json_generator_to_data(gen, NULL);
  g_object_unref(gen);
  json_node_unref(root);
  g_object_unref(builder);
  g_object_unref(parser);

  _return_ok(ctx, ar->id, json);
  g_free(json);
  _async_req_free(ar);
  return NULL;
}

static void on_list_folders(const char *id, const char *req, void *arg)
{
  _pool_dispatch(_list_folders_worker, _async_req_new(arg, id, req));
}

/* Helper to add files from a single directory to the builder array */
static void _collect_files(JsonBuilder *builder, const char *dir_path,
                           int recursive, int ignore_non_raw)
{
  GDir *d = g_dir_open(dir_path, 0, NULL);
  if(!d) return;

  const char **ext_list = ignore_non_raw ? _raw_extensions : _image_extensions;

  const gchar *name;
  while((name = g_dir_read_name(d)) != NULL)
  {
    if(name[0] == '.') continue;  // skip hidden

    gchar *fullpath = g_build_filename(dir_path, name, NULL);

    if(g_file_test(fullpath, G_FILE_TEST_IS_DIR))
    {
      if(recursive)
        _collect_files(builder, fullpath, recursive, ignore_non_raw);
      g_free(fullpath);
      continue;
    }

    /* Check file extension */
    if(!_is_extension_in(name, ext_list))
    {
      g_free(fullpath);
      continue;
    }

    GStatBuf st;
    if(g_stat(fullpath, &st) != 0)
    {
      g_free(fullpath);
      continue;
    }

    json_builder_begin_object(builder);
    json_builder_set_member_name(builder, "filename");
    json_builder_add_string_value(builder, name);
    json_builder_set_member_name(builder, "fullpath");
    json_builder_add_string_value(builder, fullpath);
    json_builder_set_member_name(builder, "modified");
    json_builder_add_int_value(builder, (gint64)st.st_mtime);
    json_builder_set_member_name(builder, "alreadyImported");
    json_builder_add_boolean_value(builder, FALSE);
    json_builder_end_object(builder);

    g_free(fullpath);
  }
  g_dir_close(d);
}

/** Query the server for which paths are already imported.
 *  Returns a GHashTable (set) of imported fullpath strings.
 *  Caller must g_hash_table_unref() the result. */
static GHashTable *_check_imported_paths(dt_webview_ctx_t *ctx, JsonArray *files_array)
{
  GHashTable *set = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);

  const guint len = json_array_get_length(files_array);
  if(len == 0 || ctx->socket_fd < 0) return set;

  /* Build params: {"paths":["...", ...]} */
  JsonBuilder *pb = json_builder_new();
  json_builder_begin_object(pb);
  json_builder_set_member_name(pb, "paths");
  json_builder_begin_array(pb);
  for(guint i = 0; i < len; i++)
  {
    JsonObject *file = json_array_get_object_element(files_array, i);
    const char *fp = json_object_get_string_member(file, "fullpath");
    if(fp) json_builder_add_string_value(pb, fp);
  }
  json_builder_end_array(pb);
  json_builder_end_object(pb);

  JsonNode *pnode = json_builder_get_root(pb);
  JsonGenerator *pgen = json_generator_new();
  json_generator_set_root(pgen, pnode);
  char *params_json = json_generator_to_data(pgen, NULL);
  g_object_unref(pgen);
  json_node_unref(pnode);
  g_object_unref(pb);

  char *error = NULL;
  char *result = dt_transport_call(ctx->transport, "catalog.check_imported", params_json, &error);
  g_free(params_json);

  if(result)
  {
    /* Parse {"imported":["path1","path2",...]} */
    JsonParser *rp = json_parser_new();
    if(json_parser_load_from_data(rp, result, -1, NULL))
    {
      JsonObject *robj = json_node_get_object(json_parser_get_root(rp));
      if(robj && json_object_has_member(robj, "imported"))
      {
        JsonArray *imported = json_object_get_array_member(robj, "imported");
        for(guint i = 0; i < json_array_get_length(imported); i++)
        {
          const char *p = json_array_get_string_element(imported, i);
          if(p) g_hash_table_add(set, g_strdup(p));
        }
      }
    }
    g_object_unref(rp);
    g_free(result);
  }
  else
  {
    fprintf(stderr, "[webview] check_imported failed: %s\n", error ? error : "unknown");
    g_free(error);
  }

  return set;
}

/* listFiles(path, recursive, ignoreNonRaw) → [{filename, fullpath, modified, alreadyImported}, …] */
static void *_list_files_worker(void *arg)
{
  async_req_t *ar = arg;
  dt_webview_ctx_t *ctx = ar->ctx;

  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(ar->req, &parser);
  if(!args || json_array_get_length(args) < 3)
  {
    _return_error(ctx, ar->id, "listFiles requires (path, recursive, ignoreNonRaw)");
    if(parser) g_object_unref(parser);
    _async_req_free(ar);
    return NULL;
  }

  const char *path = json_array_get_string_element(args, 0);

  if(!_path_is_allowed(path))
  {
    _return_error(ctx, ar->id, "path not allowed");
    g_object_unref(parser);
    _async_req_free(ar);
    return NULL;
  }

  gboolean recursive = json_array_get_boolean_element(args, 1);
  gboolean ignore_non_raw = json_array_get_boolean_element(args, 2);

  /* Phase 1: collect files from the filesystem */
  JsonBuilder *builder = json_builder_new();
  json_builder_begin_array(builder);
  _collect_files(builder, path, recursive, ignore_non_raw);
  json_builder_end_array(builder);

  JsonNode *root = json_builder_get_root(builder);
  g_object_unref(builder);

  /* Phase 2: check which files are already imported via IPC */
  JsonArray *files_array = json_node_get_array(root);
  GHashTable *imported_set = _check_imported_paths(ctx, files_array);

  /* Phase 3: patch alreadyImported flags and serialize */
  for(guint i = 0; i < json_array_get_length(files_array); i++)
  {
    JsonObject *file = json_array_get_object_element(files_array, i);
    const char *fp = json_object_get_string_member(file, "fullpath");
    gboolean is_imported = fp && g_hash_table_contains(imported_set, fp);
    json_object_set_boolean_member(file, "alreadyImported", is_imported);
  }
  g_hash_table_unref(imported_set);

  JsonGenerator *gen = json_generator_new();
  json_generator_set_root(gen, root);
  char *json = json_generator_to_data(gen, NULL);
  g_object_unref(gen);
  json_node_unref(root);
  g_object_unref(parser);

  _return_ok(ctx, ar->id, json);
  g_free(json);
  _async_req_free(ar);
  return NULL;
}

static void on_list_files(const char *id, const char *req, void *arg)
{
  _pool_dispatch(_list_files_worker, _async_req_new(arg, id, req));
}

/* ── Environment helpers ─────────────────────────────────────── */

static void on_get_home_path(const char *id, const char *req, void *arg)
{
  (void)req;
  dt_webview_ctx_t *ctx = arg;
  const char *home = g_get_home_dir();
  if(home)
  {
    JsonNode *node = json_node_new(JSON_NODE_VALUE);
    json_node_set_string(node, home);
    JsonGenerator *gen = json_generator_new();
    json_generator_set_root(gen, node);
    char *json = json_generator_to_data(gen, NULL);
    g_object_unref(gen);
    json_node_unref(node);
    _return_ok(ctx, id, json);
    g_free(json);
  }
  else
  {
    _return_ok(ctx, id, "\"/\"");
  }
}

/* ── Collection values via IPC ─────────────────────────────────── */

static void on_catalog_get_collection_values(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, id, "catalogGetCollectionValues requires (property[, filter])");
    if(parser) g_object_unref(parser);
    return;
  }

  const char *property = json_array_get_string_element(args, 0);
  const char *filter = "";
  if(json_array_get_length(args) >= 2)
    filter = json_array_get_string_element(args, 1);

  /* Escape strings for JSON */
  JsonNode *pnode = json_node_new(JSON_NODE_VALUE);
  json_node_set_string(pnode, property);
  JsonNode *fnode = json_node_new(JSON_NODE_VALUE);
  json_node_set_string(fnode, filter);
  JsonGenerator *gen = json_generator_new();
  json_generator_set_root(gen, pnode);
  char *prop_esc = json_generator_to_data(gen, NULL);
  json_generator_set_root(gen, fnode);
  char *filt_esc = json_generator_to_data(gen, NULL);
  g_object_unref(gen);
  json_node_unref(pnode);
  json_node_unref(fnode);

  char *params = g_strdup_printf("{\"property\":%s,\"filter\":%s}", prop_esc, filt_esc);
  g_free(prop_esc);
  g_free(filt_esc);
  g_object_unref(parser);

  _ipc_passthrough(ctx, id, "catalog.get_collection_values", params);
  g_free(params);
}

/* Existing filmrolls/tags endpoints are already served via server routes.
   We add simple passthrough bindings for them. */

static void on_catalog_get_filmrolls(const char *id, const char *req, void *arg)
{
  (void)req;
  _ipc_passthrough(arg, id, "catalog.get_filmrolls", "{}");
}

static void on_catalog_get_tags(const char *id, const char *req, void *arg)
{
  (void)req;
  _ipc_passthrough(arg, id, "catalog.get_tags", "{}");
}

/* ── Platform info ─────────────────────────────────────────────── */

static void on_get_platform_info(const char *id, const char *req, void *arg)
{
  (void)req;
  dt_webview_ctx_t *ctx = arg;
#if defined(__APPLE__)
  _return_ok(ctx, id, "{\"os\":\"macos\"}");
#elif defined(_WIN32)
  _return_ok(ctx, id, "{\"os\":\"windows\"}");
#else
  _return_ok(ctx, id, "{\"os\":\"linux\"}");
#endif
}

/* ── Import images via IPC ────────────────────────────────────── */

static void *_import_images_worker(void *arg)
{
  async_req_t *ar = arg;
  dt_webview_ctx_t *ctx = ar->ctx;

  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(ar->req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, ar->id, "importImages requires (paths)");
    if(parser) g_object_unref(parser);
    _async_req_free(ar);
    return NULL;
  }

  /* args[0] is an array of fullpath strings */
  JsonArray *paths = json_array_get_array_element(args, 0);

  /* Validate all paths before forwarding */
  for(guint i = 0; i < json_array_get_length(paths); i++)
  {
    const char *p = json_array_get_string_element(paths, i);
    if(p && !_path_is_allowed(p))
    {
      _return_error(ctx, ar->id, "path not allowed");
      g_object_unref(parser);
      _async_req_free(ar);
      return NULL;
    }
  }

  /* Build params: {"paths":["...", ...]} */
  JsonBuilder *pb = json_builder_new();
  json_builder_begin_object(pb);
  json_builder_set_member_name(pb, "paths");
  json_builder_begin_array(pb);
  for(guint i = 0; i < json_array_get_length(paths); i++)
  {
    const char *p = json_array_get_string_element(paths, i);
    if(p) json_builder_add_string_value(pb, p);
  }
  json_builder_end_array(pb);
  json_builder_end_object(pb);

  JsonNode *pnode = json_builder_get_root(pb);
  JsonGenerator *pgen = json_generator_new();
  json_generator_set_root(pgen, pnode);
  char *params_json = json_generator_to_data(pgen, NULL);
  g_object_unref(pgen);
  json_node_unref(pnode);
  g_object_unref(pb);

  char *error = NULL;
  char *result = dt_transport_call(ctx->transport, "catalog.import", params_json, &error);
  g_free(params_json);

  if(result)
  {
    _return_ok(ctx, ar->id, result);
    g_free(result);
  }
  else
  {
    _return_error(ctx, ar->id, error ? error : "import failed");
    g_free(error);
  }

  g_object_unref(parser);
  _async_req_free(ar);
  return NULL;
}

static void on_import_images(const char *id, const char *req, void *arg)
{
  _pool_dispatch(_import_images_worker, _async_req_new(arg, id, req));
}

/* ── Copy & import images via IPC ─────────────────────────────── */

static void *_copy_import_images_worker(void *arg)
{
  async_req_t *ar = arg;
  dt_webview_ctx_t *ctx = ar->ctx;

  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(ar->req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, ar->id, "copyAndImportImages requires (paths)");
    if(parser) g_object_unref(parser);
    _async_req_free(ar);
    return NULL;
  }

  /* args[0] is an array of fullpath strings */
  JsonArray *paths = json_array_get_array_element(args, 0);

  /* Validate all paths before forwarding */
  for(guint i = 0; i < json_array_get_length(paths); i++)
  {
    const char *p = json_array_get_string_element(paths, i);
    if(p && !_path_is_allowed(p))
    {
      _return_error(ctx, ar->id, "path not allowed");
      g_object_unref(parser);
      _async_req_free(ar);
      return NULL;
    }
  }

  /* Build params: {"paths":["...", ...]} */
  JsonBuilder *pb = json_builder_new();
  json_builder_begin_object(pb);
  json_builder_set_member_name(pb, "paths");
  json_builder_begin_array(pb);
  for(guint i = 0; i < json_array_get_length(paths); i++)
  {
    const char *p = json_array_get_string_element(paths, i);
    if(p) json_builder_add_string_value(pb, p);
  }
  json_builder_end_array(pb);
  json_builder_end_object(pb);

  JsonNode *pnode = json_builder_get_root(pb);
  JsonGenerator *pgen = json_generator_new();
  json_generator_set_root(pgen, pnode);
  char *params_json = json_generator_to_data(pgen, NULL);
  g_object_unref(pgen);
  json_node_unref(pnode);
  g_object_unref(pb);

  char *error = NULL;
  char *result = dt_transport_call(ctx->transport, "catalog.copy_import", params_json, &error);
  g_free(params_json);

  if(result)
  {
    _return_ok(ctx, ar->id, result);
    g_free(result);
  }
  else
  {
    _return_error(ctx, ar->id, error ? error : "copy_import failed");
    g_free(error);
  }

  g_object_unref(parser);
  _async_req_free(ar);
  return NULL;
}

static void on_copy_import_images(const char *id, const char *req, void *arg)
{
  _pool_dispatch(_copy_import_images_worker, _async_req_new(arg, id, req));
}

/* ── File thumbnail (embedded EXIF preview) via IPC ───────────── */

static void on_get_file_thumbnail(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, id, "getFileThumbnail requires (path)");
    if(parser) g_object_unref(parser);
    return;
  }

  const char *path = json_array_get_string_element(args, 0);

  if(!_path_is_allowed(path))
  {
    _return_error(ctx, id, "path not allowed");
    g_object_unref(parser);
    return;
  }

  /* Escape path for JSON */
  JsonNode *node = json_node_new(JSON_NODE_VALUE);
  json_node_set_string(node, path);
  JsonGenerator *gen = json_generator_new();
  json_generator_set_root(gen, node);
  char *escaped = json_generator_to_data(gen, NULL);
  g_object_unref(gen);
  json_node_unref(node);

  char *params = g_strdup_printf("{\"path\":%s}", escaped);
  g_free(escaped);
  g_object_unref(parser);

  _ipc_passthrough(ctx, id, "catalog.get_file_thumbnail", params);
  g_free(params);
}

/* ── Image actions ────────────────────────────────────────────── */

static void _image_action_passthrough(const char *id, const char *req, void *arg,
                                      const char *method)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    if(parser) g_object_unref(parser);
    _return_error(ctx, id, "Missing imgids argument");
    return;
  }

  /* Build params: {"imgids": [...], ...extra} */
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "imgids");
  json_builder_add_value(b, json_node_copy(json_array_get_element(args, 0)));

  /* Optional second argument (direction for rotate) */
  if(json_array_get_length(args) > 1)
  {
    json_builder_set_member_name(b, "direction");
    json_builder_add_int_value(b, (gint64)json_array_get_int_element(args, 1));
  }

  json_builder_end_object(b);
  JsonNode *root = json_builder_get_root(b);
  JsonGenerator *gen = json_generator_new();
  json_generator_set_root(gen, root);
  char *params = json_generator_to_data(gen, NULL);
  g_object_unref(gen);
  json_node_unref(root);
  g_object_unref(b);
  g_object_unref(parser);

  _ipc_passthrough(ctx, id, method, params);
  g_free(params);
}

static void on_image_remove(const char *id, const char *req, void *arg)
{ _image_action_passthrough(id, req, arg, "catalog.image_remove"); }

static void on_image_delete(const char *id, const char *req, void *arg)
{ _image_action_passthrough(id, req, arg, "catalog.image_delete"); }

static void on_image_duplicate(const char *id, const char *req, void *arg)
{ _image_action_passthrough(id, req, arg, "catalog.image_duplicate"); }

static void on_image_rotate(const char *id, const char *req, void *arg)
{ _image_action_passthrough(id, req, arg, "catalog.image_rotate"); }

static void on_image_group(const char *id, const char *req, void *arg)
{ _image_action_passthrough(id, req, arg, "catalog.image_group"); }

static void on_image_ungroup(const char *id, const char *req, void *arg)
{ _image_action_passthrough(id, req, arg, "catalog.image_ungroup"); }

static void on_image_copy_local(const char *id, const char *req, void *arg)
{ _image_action_passthrough(id, req, arg, "catalog.image_copy_local"); }

static void on_image_resync_local(const char *id, const char *req, void *arg)
{ _image_action_passthrough(id, req, arg, "catalog.image_resync_local"); }

static void on_image_refresh_exif(const char *id, const char *req, void *arg)
{ _image_action_passthrough(id, req, arg, "catalog.image_refresh_exif"); }

/* Metadata/monochrome: forward single JSON object arg directly */
static void _json_object_passthrough(const char *id, const char *req, void *arg,
                                     const char *method)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    if(parser) g_object_unref(parser);
    _return_error(ctx, id, "Missing argument");
    return;
  }

  JsonNode *obj_node = json_array_get_element(args, 0);
  JsonGenerator *gen = json_generator_new();
  json_generator_set_root(gen, obj_node);
  char *params = json_generator_to_data(gen, NULL);
  g_object_unref(gen);
  g_object_unref(parser);

  _ipc_passthrough(ctx, id, method, params);
  g_free(params);
}

static void on_metadata_paste(const char *id, const char *req, void *arg)
{ _json_object_passthrough(id, req, arg, "catalog.metadata_paste"); }

static void on_metadata_clear(const char *id, const char *req, void *arg)
{ _json_object_passthrough(id, req, arg, "catalog.metadata_clear"); }

static void on_image_set_monochrome(const char *id, const char *req, void *arg)
{ _json_object_passthrough(id, req, arg, "catalog.image_set_monochrome"); }

/* Move/copy to folder: args = [imgids[], path] */
static void _image_folder_action(const char *id, const char *req, void *arg,
                                 const char *method)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 2)
  {
    if(parser) g_object_unref(parser);
    _return_error(ctx, id, "Missing imgids or path argument");
    return;
  }

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "imgids");
  json_builder_add_value(b, json_node_copy(json_array_get_element(args, 0)));
  json_builder_set_member_name(b, "path");
  json_builder_add_string_value(b, json_array_get_string_element(args, 1));
  json_builder_end_object(b);
  JsonNode *root = json_builder_get_root(b);
  JsonGenerator *gen = json_generator_new();
  json_generator_set_root(gen, root);
  char *params = json_generator_to_data(gen, NULL);
  g_object_unref(gen);
  json_node_unref(root);
  g_object_unref(b);
  g_object_unref(parser);

  _ipc_passthrough(ctx, id, method, params);
  g_free(params);
}

static void on_image_move(const char *id, const char *req, void *arg)
{ _image_folder_action(id, req, arg, "catalog.image_move"); }

static void on_image_copy_to(const char *id, const char *req, void *arg)
{ _image_folder_action(id, req, arg, "catalog.image_copy_to"); }


/* ── Window titlebar actions ──────────────────────────────────── */

static void on_window_start_drag(const char *id, const char *req, void *arg)
{
  (void)req;
  dt_webview_ctx_t *ctx = arg;
  dt_titlebar_start_drag(ctx->webview);
  _return_ok(ctx, id, "null");
}

static void on_window_zoom(const char *id, const char *req, void *arg)
{
  (void)req;
  dt_webview_ctx_t *ctx = arg;
  dt_titlebar_zoom(ctx->webview);
  _return_ok(ctx, id, "null");
}

/* ── Native folder picker (via nativefiledialog-extended) ─────── */

static void on_pick_folder(const char *id, const char *req, void *arg)
{
  (void)req;
  dt_webview_ctx_t *ctx = arg;

  nfdchar_t *out_path = NULL;
  nfdresult_t result = NFD_PickFolder(&out_path, NULL);

  if(result == NFD_OKAY)
  {
    /* Return the path as a JSON string */
    JsonNode *node = json_node_new(JSON_NODE_VALUE);
    json_node_set_string(node, out_path);
    JsonGenerator *gen = json_generator_new();
    json_generator_set_root(gen, node);
    char *json = json_generator_to_data(gen, NULL);
    g_object_unref(gen);
    json_node_unref(node);

    _return_ok(ctx, id, json);
    g_free(json);
    NFD_FreePath(out_path);
  }
  else if(result == NFD_CANCEL)
  {
    _return_ok(ctx, id, "null");
  }
  else
  {
    _return_error(ctx, id, NFD_GetError());
  }
}

static void on_config_get(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, id, "configGet requires (key)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *key = _get_string_arg(args, 0);
  g_object_unref(parser);

  char *ipc_params = g_strdup_printf("{\"key\":\"%s\"}", key);
  g_free(key);

  _ipc_passthrough(ctx, id, "config.get", ipc_params);
  g_free(ipc_params);
}

static void on_config_set(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 2)
  {
    _return_error(ctx, id, "configSet requires (key, value)");
    if(parser) g_object_unref(parser);
    return;
  }

  char *key = _get_string_arg(args, 0);
  char *value = _get_string_arg(args, 1);
  g_object_unref(parser);

  // Validation (dt_conf_key_exists) happens server-side in _handle_config_set
  char *ipc_params = g_strdup_printf("{\"key\":\"%s\",\"value\":\"%s\"}", key, value);
  g_free(key);
  g_free(value);

  _ipc_passthrough(ctx, id, "config.set", ipc_params);
  g_free(ipc_params);
}

void dt_webview_register_bindings(dt_webview_ctx_t *ctx)
{
  /* Initialize worker thread pool */
  _binding_pool_init();

  /* Initialize NFD once */
  NFD_Init();

  /* Set up transport: if not already created (direct mode sets it before calling us),
   * create an IPC transport wrapping the socket */
  if(!ctx->transport)
  {
    ctx->transport = dt_transport_ipc_new(ctx->socket_fd);

    /* Transfer server process ownership to transport (for unified shutdown) */
    if(ctx->server_pid > 0)
    {
      dt_transport_ipc_set_server_pid(ctx->transport, ctx->server_pid);
      ctx->server_pid = -1;  /* transport owns it now */
    }

    /* Create event-aware IPC context with reader thread (owned by transport) */
    dt_ipc_context_t *ipc_ctx = dt_ipc_context_new(ctx->socket_fd, _on_server_event, ctx);
    if(!ipc_ctx)
      fprintf(stderr, "[webview] WARNING: failed to create IPC context, falling back to legacy IPC\n");
    else
      dt_transport_ipc_set_context(ctx->transport, ipc_ctx);

    /* Socket is now owned by the transport */
    ctx->socket_fd = -1;
  }
  else
  {
    /* Direct mode: register event callback on the transport so server events
     * (like preview_ready) reach the webview's JS layer */
    dt_transport_set_event_callback(ctx->transport, _on_server_event, ctx);
  }

  /* Start local HTTP server for zero-copy frame delivery */
  ctx->frame_server = _frame_server_start(ctx);

  webview_bind(ctx->webview, "ping", on_ping, ctx);
  webview_bind(ctx->webview, "catalogQuery", on_catalog_query, ctx);
  webview_bind(ctx->webview, "catalogGetThumbnail", on_catalog_get_thumbnail, ctx);
  webview_bind(ctx->webview, "catalogGetThumbnails", on_catalog_get_thumbnails, ctx);
  webview_bind(ctx->webview, "developOpen", on_develop_open, ctx);
  webview_bind(ctx->webview, "developClose", on_develop_close, ctx);
  webview_bind(ctx->webview, "developSetParams", on_develop_set_params, ctx);
  webview_bind(ctx->webview, "developCommitParams", on_develop_commit_params, ctx);
  webview_bind(ctx->webview, "developResetParams", on_develop_reset_params, ctx);
  webview_bind(ctx->webview, "developGetParams", on_develop_get_params, ctx);
  webview_bind(ctx->webview, "developRequestPreview", on_develop_request_preview, ctx);
  webview_bind(ctx->webview, "developSamplePixels", on_develop_sample_pixels, ctx);
  webview_bind(ctx->webview, "developGetModules", on_develop_get_modules, ctx);
  webview_bind(ctx->webview, "developGetHistory", on_develop_get_history, ctx);
  webview_bind(ctx->webview, "developSelectHistory", on_develop_select_history, ctx);
  webview_bind(ctx->webview, "developCompressHistory", on_develop_compress_history, ctx);
  webview_bind(ctx->webview, "developTruncateHistory", on_develop_truncate_history, ctx);
  webview_bind(ctx->webview, "developDeleteHistory", on_develop_delete_history, ctx);
  webview_bind(ctx->webview, "developListPresets", on_develop_list_presets, ctx);
  webview_bind(ctx->webview, "developApplyPreset", on_develop_apply_preset, ctx);
  webview_bind(ctx->webview, "developStorePreset", on_develop_store_preset, ctx);
  webview_bind(ctx->webview, "developDeletePreset", on_develop_delete_preset, ctx);
  webview_bind(ctx->webview, "developNewInstance", on_develop_new_instance, ctx);
  webview_bind(ctx->webview, "developDeleteInstance", on_develop_delete_instance, ctx);
  webview_bind(ctx->webview, "developMoveInstance", on_develop_move_instance, ctx);
  webview_bind(ctx->webview, "developRenameInstance", on_develop_rename_instance, ctx);
  webview_bind(ctx->webview, "developGetIntrospection", on_develop_get_introspection, ctx);
  webview_bind(ctx->webview, "developGetMasks", on_develop_get_masks, ctx);
  webview_bind(ctx->webview, "developRenameMask", on_develop_rename_mask, ctx);
  webview_bind(ctx->webview, "developDeleteMask", on_develop_delete_mask, ctx);
  webview_bind(ctx->webview, "getPreviewFrame", on_get_preview_frame, ctx);
  webview_bind(ctx->webview, "getFramePort", on_get_frame_port, ctx);
  webview_bind(ctx->webview, "pickFolder", on_pick_folder, ctx);
  webview_bind(ctx->webview, "listFolders", on_list_folders, ctx);
  webview_bind(ctx->webview, "listFiles", on_list_files, ctx);
  webview_bind(ctx->webview, "getHomePath", on_get_home_path, ctx);
  webview_bind(ctx->webview, "importImages", on_import_images, ctx);
  webview_bind(ctx->webview, "copyAndImportImages", on_copy_import_images, ctx);
  webview_bind(ctx->webview, "getFileThumbnail", on_get_file_thumbnail, ctx);
  webview_bind(ctx->webview, "getPlatformInfo", on_get_platform_info, ctx);
  webview_bind(ctx->webview, "catalogGetCollectionValues", on_catalog_get_collection_values, ctx);
  webview_bind(ctx->webview, "catalogGetFilmrolls", on_catalog_get_filmrolls, ctx);
  webview_bind(ctx->webview, "catalogGetTags", on_catalog_get_tags, ctx);
  webview_bind(ctx->webview, "configGet", on_config_get, ctx);
  webview_bind(ctx->webview, "configSet", on_config_set, ctx);

  /* Image actions */
  webview_bind(ctx->webview, "imageRemove", on_image_remove, ctx);
  webview_bind(ctx->webview, "imageDelete", on_image_delete, ctx);
  webview_bind(ctx->webview, "imageDuplicate", on_image_duplicate, ctx);
  webview_bind(ctx->webview, "imageRotate", on_image_rotate, ctx);
  webview_bind(ctx->webview, "imageGroup", on_image_group, ctx);
  webview_bind(ctx->webview, "imageUngroup", on_image_ungroup, ctx);
  webview_bind(ctx->webview, "imageCopyLocal", on_image_copy_local, ctx);
  webview_bind(ctx->webview, "imageResyncLocal", on_image_resync_local, ctx);
  webview_bind(ctx->webview, "imageRefreshExif", on_image_refresh_exif, ctx);
  webview_bind(ctx->webview, "metadataPaste", on_metadata_paste, ctx);
  webview_bind(ctx->webview, "metadataClear", on_metadata_clear, ctx);
  webview_bind(ctx->webview, "imageSetMonochrome", on_image_set_monochrome, ctx);
  webview_bind(ctx->webview, "imageMove", on_image_move, ctx);
  webview_bind(ctx->webview, "imageCopyTo", on_image_copy_to, ctx);

  /* Test automation — internal binding for returning eval results */
  webview_bind(ctx->webview, "__testResult", on_test_result, ctx);
}

void dt_webview_register_window_bindings(dt_webview_ctx_t *ctx)
{
  webview_bind(ctx->webview, "windowStartDrag", on_window_start_drag, ctx);
  webview_bind(ctx->webview, "windowZoom", on_window_zoom, ctx);
}
